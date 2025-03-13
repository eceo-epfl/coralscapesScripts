import numpy as np

import torch 
from torch.optim.lr_scheduler import PolynomialLR

from tqdm import tqdm 

import segmentation_models_pytorch as smp
from transformers import ConvNextV2Config, ConvNextV2Model, UperNetConfig, UperNetForSemanticSegmentation, SegformerForSemanticSegmentation, Mask2FormerForUniversalSegmentation, AutoModel, Mask2FormerConfig
from peft import LoraConfig, get_peft_model

from coralscapesScripts.segmentation.models.dinov2 import Dinov2ForSemanticSegmentation, DPTDinov2ForSemanticSegmentation
from coralscapesScripts.datasets.preprocess import get_preprocessor, preprocess_batch, get_windows
from coralscapesScripts.segmentation.loss import get_loss_fn


def get_batch_predictions(data, model, device, loss_fn = None):
    """
    Get batch predictions from the model and optionally compute the loss.
    Parameters:
    data (tuple, list, or dict): The input data for the model. It can be a tuple or list containing inputs and labels,
                                    or a dictionary with keys "pixel_values", "mask_labels", and "class_labels".
    model (torch.nn.Module): The model to use for generating predictions.
    device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
    loss_fn (callable, optional): The loss function to use for computing the loss. Default is None.
    Returns:
    tuple: A tuple containing:
        - outputs (torch.Tensor or dict): The model's predictions.
        - loss (torch.Tensor or None): The computed loss if loss_fn is provided or if the model's output contains a loss attribute, otherwise None.
    """

    loss = None
    if(isinstance(data,tuple) or isinstance(data,list)):
        inputs, labels = data
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
 
        if(loss_fn):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        else:
            outputs = model(inputs, labels = labels)

    if("pixel_values" and "mask_labels" in data):
        outputs = model(
            pixel_values=data["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in data["mask_labels"]],
            class_labels=[labels.to(device) for labels in data["class_labels"]], 
        )

    if("pixel_values" and "labels" in data):
        outputs = model(
            pixel_values=data["pixel_values"].to(device),
            labels=data["labels"].to(device),
        )

    if(isinstance(outputs, dict) and "out" in outputs):
        outputs = outputs["out"]

    if(hasattr(outputs, "loss")):
        loss = outputs.loss

    return outputs, loss 

class Benchmark_Run:
    """
    Benchmark_Run class for running segmentation model benchmarks.
    Attributes:
        run_name (str): Name of the benchmark run.
        model_name (str): Name of the model to be used.
        N_classes (int): Number of classes for segmentation.
        device (torch.device): Device to run the model on (default is "cuda").
        model_kwargs (dict): Dictionary containing model keyword arguments.
        model_checkpoint (str): Path to the model checkpoint.
        lora_kwargs (dict): Dictionary containing LoRA keyword arguments.
        training_hyperparameters (dict): Dictionary containing training hyperparameters.
    Methods:
        __init__(run_name: str, model_name: str, N_classes: int, device: torch.device = "cuda", training_hyperparameters: dict = {...}):
            Initializes the Benchmark_Run class with the given parameters.
        train_epoch(train_loader):
            Trains the model for one epoch.
        validation_epoch(val_loader):
            Validates the model for one epoch.
    """

    def __init__(self, run_name: str, model_name: str, N_classes: int, 
                 device:torch.device = "cuda", 
                 model_kwargs:dict = {},
                 model_checkpoint: str = None,
                 lora_kwargs:dict = None,
                 training_hyperparameters:dict = {"epochs": 50, 
                                               "optimizer": {"type": torch.optim.SGD, "lr": 0.001, "momentum": 0.9, "weight_decay": 1e-4}, 
                                               "scheduler": {"type": PolynomialLR, "power": 0.9},
                                               "loss": {"type": "cross_entropy", "reduction": "mean"},
                                               "preprocessor": None
                                                }):

        self.run_name = run_name
        self.N_classes = N_classes
        self.device = device

        self.model_name = model_name
        self.model_checkpoint = model_checkpoint
        self.model_kwargs = model_kwargs
        self.lora_kwargs = lora_kwargs
        self.training_hyperparameters = training_hyperparameters

        ## Load preprocessor if necessary
        self.preprocessor = get_preprocessor(training_hyperparameters.preprocessor)

        ## Load model
        if(model_name == "deeplabv3+resnet50"):
            self.model = smp.DeepLabV3Plus(encoder_name="resnet50", classes=N_classes)

        elif(model_name == "unet++resnet50"):
            self.model = smp.UnetPlusPlus(encoder_name="resnet50", classes=N_classes)

        elif(model_name == "segformer-mit-b2"):
            pretrained_model_name = "nvidia/mit-b2"
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name, id2label={i:i for i in range(0, N_classes)}, 
                semantic_loss_ignore_index = 0, ignore_mismatched_sizes=True
            )

        elif(model_name == "segformer-mit-b5"):
            pretrained_model_name = "nvidia/mit-b5"
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name, id2label={i:i for i in range(0, N_classes)}, 
                semantic_loss_ignore_index = 0, ignore_mismatched_sizes=True
            )

        elif(model_name == "dpt-dinov2-base"):
            self.model = DPTDinov2ForSemanticSegmentation(num_labels=N_classes, 
                                                    backbone = "facebook/dinov2-base")
            
        elif(model_name == "dpt-dinov2-giant"):
            self.model = DPTDinov2ForSemanticSegmentation(num_labels=N_classes, 
                                                    backbone = "facebook/dinov2-giant")

        elif(model_name == "linear-dinov2-base"):
            self.model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", 
                                                                       id2label={i:i for i in range(0, N_classes)}, 
                                                                       num_labels=N_classes)
        else:
            raise ValueError("Model not found")

        ## PEFT model
        if(lora_kwargs):
            lora_config = LoraConfig(
                                    r = lora_kwargs.r,
                                    lora_alpha = lora_kwargs.lora_alpha,
                                    target_modules = ["query", "value"],
                                    lora_dropout = 0.1,
                                    bias = "lora_only",
                                    modules_to_save = lora_kwargs.modules_to_save,
                                     )
            self.model = get_peft_model(self.model, lora_config)

        self.model = self.model.to(device)

        ## Load model checkpoint if necessary
        if(model_checkpoint):
            checkpoint = torch.load(model_checkpoint)
            if("model_state_dict" in checkpoint):
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(torch.load(model_checkpoint))

        ## Load training procedure: optimizer, scheduler, loss
        optimizer = training_hyperparameters.optimizer.pop("type", None)
        lr_multiplier = training_hyperparameters.optimizer.pop("lr_multiplier", None)
        if(lr_multiplier):
            backbone_regex = training_hyperparameters.optimizer.pop("backbone_regex", None)
            backbone_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if backbone_regex in name:  # Adjust this to match the actual backbone param names
                    backbone_params.append(param)
                else:
                    other_params.append(param)

            lr = training_hyperparameters.optimizer.pop("lr", None)

            self.optimizer = eval(optimizer)([
                {"params": backbone_params, "lr": lr * lr_multiplier},
                {"params": other_params, "lr": lr}
            ], **training_hyperparameters.optimizer)

        else:
            self.optimizer = eval(optimizer)(params = self.model.parameters(), **training_hyperparameters.optimizer)

        if(training_hyperparameters.scheduler):
            scheduler = training_hyperparameters.scheduler.pop("type", None)
            self.scheduler = eval(scheduler)(self.optimizer, **training_hyperparameters.scheduler)

        if(training_hyperparameters.loss):
            loss = training_hyperparameters.loss.pop("type", None)
            self.loss = get_loss_fn(loss, training_hyperparameters.loss)
        else:
            self.loss = None # In the case of HF models, the loss is already included in the model

        self.eval = training_hyperparameters.eval

    def train_epoch(self, train_loader):
        running_loss = 0.

        for i, data in tqdm(enumerate(train_loader)):
            self.optimizer.zero_grad() 
            preprocessed_batch = preprocess_batch(data, self.preprocessor)
            outputs, loss = get_batch_predictions(preprocessed_batch, self.model, self.device, loss_fn = self.loss)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / (i+1)

    def validation_epoch(self, val_loader):
        running_loss = 0.

        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader)):   
                if(len(data[0])==1): # If the batch size is one we skip 
                    i = i-1
                    continue
                if(self.eval and self.eval.sliding_window):
                    input_windows, label_windows = get_windows(data, self.eval.window, self.eval.stride, self.eval.window_target, self.eval.stride_target)
                    input_windows = input_windows.view(-1, *input_windows.shape[-3:])
                    label_windows = label_windows.view(-1, *label_windows.shape[-2:])                    
                    data = (input_windows, label_windows)
                preprocessed_batch = preprocess_batch(data, self.preprocessor)
                outputs, loss = get_batch_predictions(preprocessed_batch, self.model, self.device, loss_fn = self.loss)
                running_loss += loss.item()

        return running_loss / (i+1)    
    
    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )


def predict(inputs, benchmark_run, output_size = None, window_dims = None):
    """
    Perform prediction using the given model and inputs.
    Args:
        inputs (torch.Tensor or dict): The input data for the model. It can be a tensor for CNN & DINO models or a dictionary containing "pixel_values" for Segformer models.
        benchmark_run (object): An object containing the model, device, and evaluation settings.
        output_size (tuple, optional): The desired output size for the prediction. If None, the output size will be the same as the input size.
        window_dims (tuple, optional): The dimensions of the sliding window for evaluation. Required if sliding window evaluation is enabled.
    Returns:
        numpy.ndarray: The predicted output as a numpy array.
    """

    if(torch.is_tensor(inputs)): #For CNN & DINO models
        if(output_size is None):
            output_size = inputs.shape[-2:]
        inputs = inputs.to(benchmark_run.device).float()
        outputs = benchmark_run.model(inputs)
        if(hasattr(outputs, "logits")): #For DINO models
            outputs = outputs.logits
            outputs = torch.nn.functional.interpolate(outputs, size=output_size, mode="bilinear", align_corners=False)
        outputs = outputs.argmax(dim=1)

    elif("pixel_values" in inputs):  #For Segformer models
        original_images = inputs["pixel_values"]
        target_sizes = [
            (image.shape[1], image.shape[2]) for image in original_images
        ]
        outputs = benchmark_run.model(
            pixel_values=inputs["pixel_values"].to(benchmark_run.device),
        )
        outputs = benchmark_run.preprocessor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
        outputs = torch.stack(outputs)

    if(isinstance(outputs, dict) and "out" in outputs):
        outputs = outputs["out"]

    if(benchmark_run.eval and benchmark_run.eval.sliding_window):
        outputs = outputs.view(*window_dims, benchmark_run.eval.window, benchmark_run.eval.stride)                  
        outputs = torch.cat([torch.cat(list(row), dim=-1) for row in list(outputs)], dim=-2)

    return outputs[0].cpu().numpy()