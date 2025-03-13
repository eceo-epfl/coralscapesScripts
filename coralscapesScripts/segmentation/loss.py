import torch 

import segmentation_models_pytorch as smp

class CombinedCrossEntropyDiceLoss(torch.nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5, ignore_index=0, reduction = "mean", **loss_kwargs):
        """
        Initializes the class of a combined CrossEntropy and Dice Loss.
        Args:
            weight_ce (float, optional): Weight for the CrossEntropy loss component. Defaults to 0.5.
            weight_dice (float, optional): Weight for the Dice loss component. Defaults to 0.5.
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Defaults to 0.
            loss_params (dict, optional): Additional parameters for the Dice loss. Defaults to an empty dictionary.
        """

        super(CombinedCrossEntropyDiceLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction = reduction)
        self.dice_loss = smp.losses.DiceLoss(mode = "multiclass", ignore_index = ignore_index, **loss_kwargs)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (N, C, H, W) where C is the number of classes.
            targets: Tensor of shape (N, H, W) with class indices.

        Returns:
            Combined loss (weighted sum of CrossEntropy and Dice Loss)
        """
        # Cross Entropy Loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Dice Loss
        dice_loss = self.dice_loss(logits, targets)
        
        # Combine losses
        total_loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss
        return total_loss

def get_loss_fn(loss_name, loss_params):
    if loss_name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(ignore_index = 0, **loss_params)
    elif loss_name == 'dice':
        return smp.losses.DiceLoss(mode = "multiclass", ignore_index = 0, **loss_params)
    elif loss_name == 'cross_entropy+dice':
        return CombinedCrossEntropyDiceLoss(ignore_index = 0, **loss_params)
