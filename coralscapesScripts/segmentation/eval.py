import numpy as np 

import torch 

from torchmetrics.classification import Accuracy, JaccardIndex
from torchmetrics.segmentation import MeanIoU

from coralscapesScripts.visualization import denormalize_image, color_label
from coralscapesScripts.datasets.preprocess import preprocess_batch, get_windows

def get_batch_predictions_eval(data, model, device, preprocessor = None):
    """
    Generate batch predictions for evaluation using different types of models.
    Args:
        data (tuple, list, or dict): Input data for the model. It can be:
            - A tuple or list containing inputs and labels for CNN & DINO models.
            - A dictionary containing "pixel_values" and "labels" for Segformer models.
        model (torch.nn.Module): The model used for generating predictions.
        device (torch.device): The device (CPU or GPU) on which the computation will be performed.
        preprocessor (optional): A preprocessor object for post-processing the outputs of Segformer models.
    Returns:
        torch.Tensor: The predicted outputs from the model.
    """

    if(isinstance(data,tuple) or isinstance(data,list)): #For CNN & DINO models
        inputs, labels = data 
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        outputs = model(inputs)
        if(hasattr(outputs, "logits")): #For DINO models
            outputs = outputs.logits
            outputs = torch.nn.functional.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        outputs = outputs.argmax(dim=1)

    if("pixel_values" and "labels" in data):  #For Segformer models
        target_sizes = [
            (image.shape[1], image.shape[2]) for image in data["labels"]
        ]
        outputs = model(
            pixel_values=data["pixel_values"].to(device),
        )
        outputs = preprocessor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
        outputs = torch.stack(outputs)

    if(isinstance(outputs, dict) and "out" in outputs):
        outputs = outputs["out"]

    return outputs

class Evaluator:
    def __init__(self, N_classes: int, device:torch.device = "cuda", metric_dict: dict = None, preprocessor = None, eval_params: dict = None):
        """
        Initializes the evaluation class with specified metrics. Currently usable for classification and semantic segmentation tasks.
        Args:
            N_classes (int): Number of classes for the classification task.
            device (torch.device, optional): Device to run the evaluation on. Defaults to "cuda".
            metric_dict (dict, optional): Dictionary containing the metrics to be computed. If None, default metrics are used: accuracy and mean_iou.
            preprocessor (object, optional): Preprocessor object for post-processing the segmentation outputs. Defaults to None.
            eval_params (dict, optional): Evaluation parameters for sliding window inference. Defaults to None.
        """
        self.device = device
        self.N_classes = N_classes
        if(metric_dict):
            self.metric_dict = metric_dict
        else:
            self.metric_dict = {
                                "accuracy": Accuracy(task="multiclass" if N_classes > 2 else "binary", num_classes=int(N_classes), ignore_index = 0).to(device),
                                "mean_iou": JaccardIndex(task="multiclass" if N_classes > 2 else "binary", num_classes=int(N_classes), ignore_index = 0).to(device)
                                }
        self.preprocessor = preprocessor
        self.eval = eval_params

    def evaluate_model(self, dataloader: torch.utils.data.dataloader.DataLoader, model: torch.nn.Module, split = "val"):
        """
        Evaluates the given model using the provided dataloader and computes the metrics.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing the validation data.
            model (torch.nn.Module): The model to be evaluated.
            split (str, optional): The split of the data to be evaluated. Defaults to "val".
        Returns:
            dict: A dictionary containing the computed metric results.
        Notes:
            - The model is set to evaluation mode during the evaluation process.
            - The data is transferred to the appropriate device before making predictions.
            - The predictions are obtained by applying argmax on the model outputs.
            - The metrics are updated and computed based on the predictions and true labels.
        """

        model.eval()
        metric_results = {}
        with torch.no_grad():
            for i, vdata in enumerate(dataloader):
                __, vlabels = vdata
                vlabels = vlabels.to(self.device).long()
                
                if(split!="train" and self.eval and self.eval.sliding_window):
                    input_windows, label_windows = get_windows(vdata, self.eval.window, self.eval.stride, self.eval.window_target, self.eval.stride_target)
                    n_vertical, n_horizontal, batch_size = input_windows.shape[:3]
                    input_windows = input_windows.view(-1, *input_windows.shape[-3:])
                    label_windows = label_windows.view(-1, *label_windows.shape[-2:])                    
                    vdata = (input_windows, label_windows)

                preprocessed_batch = preprocess_batch(vdata, self.preprocessor)
                voutputs = get_batch_predictions_eval(preprocessed_batch, model, self.device, preprocessor=self.preprocessor)

                if(split!="train" and self.eval and self.eval.sliding_window):
                    if(self.eval.window_target):
                      voutputs = voutputs.view(n_vertical, n_horizontal, batch_size, self.eval.window_target, self.eval.stride_target)     
                    else:
                      voutputs = voutputs.view(n_vertical, n_horizontal, batch_size, self.eval.window, self.eval.stride)                  
             
                    voutputs = torch.cat([torch.cat(list(row), dim=-1) for row in list(voutputs)], dim=-2)

                ## Update metrics
                for metric in self.metric_dict.values():
                    metric.update(voutputs, vlabels)
        
        ## Compute metrics
        for metric_name in self.metric_dict:
            metric_results[metric_name] = self.metric_dict[metric_name].compute().cpu().numpy()
            if(metric_results[metric_name].ndim==0):
                metric_results[metric_name] = metric_results[metric_name].item()
            self.metric_dict[metric_name].reset()
        
        return metric_results


    def evaluate_image(self, dataloader: torch.utils.data.dataloader.DataLoader, model: torch.nn.Module, split = "val", epoch = 0):
        """
        Evaluates the given model on one image of the dataloader.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing the validation data.
            model (torch.nn.Module): The model to be evaluated.
            split (str, optional): The split of the data to be evaluated. Defaults to "val".
            epoch (int, optional): The epoch number for logging purposes. Defaults to 0.
        Returns:
            dict: A dictionary containing the computed metric results.
        Notes:
            - The model is set to evaluation mode during the evaluation process.
            - The data is transferred to the appropriate device before making predictions.
            - The predictions are obtained by applying argmax on the model outputs.
            - The metrics are updated and computed based on the predictions and true labels.
        """
        
        model.eval()
        with torch.no_grad():
            vdata = next(iter(dataloader))
            vinputs, vlabels = vdata
            vlabels = vlabels.to(self.device).long()

            if(split!="train" and self.eval and self.eval.sliding_window):
                input_windows, label_windows = get_windows(vdata, self.eval.window, self.eval.stride, self.eval.window_target, self.eval.stride_target)
                n_vertical, n_horizontal, batch_size = input_windows.shape[:3]
                input_windows = input_windows.view(-1, *input_windows.shape[-3:])
                label_windows = label_windows.view(-1, *label_windows.shape[-2:])                    
                vdata = (input_windows, label_windows)

            preprocessed_batch = preprocess_batch(vdata, self.preprocessor)
            voutputs = get_batch_predictions_eval(preprocessed_batch, model, self.device, preprocessor=self.preprocessor)

            if(split!="train" and self.eval and self.eval.sliding_window):
                if(self.eval.window_target):
                    voutputs = voutputs.view(n_vertical, n_horizontal, batch_size, self.eval.window_target, self.eval.stride_target)     
                else:
                    voutputs = voutputs.view(n_vertical, n_horizontal, batch_size, self.eval.window, self.eval.stride)       
                voutputs = torch.cat([torch.cat(list(row), dim=-1) for row in list(voutputs)], dim=-2)

        image_counter = epoch%(5*5)//5 #Due to log_epochs being 5 and rotating 5 images
        image_counter = image_counter%len(vinputs) #In case we use a smaller batch size

        image = vinputs[image_counter].cpu().numpy()
        label = vlabels[image_counter].cpu().numpy()
        pred = voutputs[image_counter].cpu().numpy()

        return image, label, pred