from transformers import Dinov2Model, Dinov2PreTrainedModel, DPTForSemanticSegmentation, DPTConfig, Dinov2Config
from transformers.modeling_outputs import SemanticSegmenterOutput
import torch 

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        return self.classifier(embeddings)

class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(config.hidden_size, 74, 37, config.num_labels)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
        # use frozen features
        outputs = self.dinov2(pixel_values,
                                output_hidden_states=output_hidden_states,
                                output_attentions=output_attentions)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:,1:,:]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            # print(logits.shape, labels.shape)
            logits = torch.nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Custom Wrapper to Integrate DINOv2 with DPT
class DPTDinov2ForSemanticSegmentation(torch.nn.Module):
    def __init__(self, num_labels = 40, backbone = "facebook/dinov2-base"):
        super().__init__()
        dinov2_config = Dinov2Config(reshape_hidden_states=True).from_pretrained(backbone)
        # dinov2_config = Dinov2Config(backbone = backbone, reshape_hidden_states=True, use_pretrained_backbone = True)
        # dinov2_backbone = Dinov2Model(dinov2_config)
        dinov2_backbone = Dinov2Model.from_pretrained(backbone, reshape_hidden_states=True)
        self.backbone = dinov2_backbone
        if backbone=="facebook/dinov2-base": 
            self.indices = (2,5,8,11) 
        elif backbone == "facebook/dinov2-giant":
            self.indices = (9, 19, 29, 39)
        
        config = DPTConfig(
            num_labels=num_labels,  
            ignore_index=0,  
            semantic_loss_ignore_index = 0,
            is_hybrid=False,  
            backbone_out_indices=self.indices,  
            backbone_config = dinov2_config
        )

        # Load the DPT segmentation model
        dpt_model = DPTForSemanticSegmentation(config)

        self.neck = dpt_model.neck
        self.head = dpt_model.head
        # self.auxiliary_head = dpt_model.auxiliary_head

    def forward(self, pixel_values, labels=None):
        features = self.backbone(pixel_values, output_hidden_states = True)
        features = [features.hidden_states[i] for i in self.indices]

        h, w = pixel_values.shape[-2:]
        if h!=w: # In a non square image 
            if(h%14!=0 or w%14!=0):
                raise ValueError("Height and width must be divisible by the patch size (14).")
            
            patch_height = h//14
            patch_width = w//14
            features = self.neck(features, patch_height = patch_height, patch_width = patch_width)
        else:
            features = self.neck(features)
        
        logits = self.head(features)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            # print(logits.shape, labels.shape)
            logits = torch.nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
        )

