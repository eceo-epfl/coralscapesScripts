import numpy as np 

import torch 

import wandb
import os 
import json 
import time

from coralscapesScripts.visualization import denormalize_image, color_label, color_correctness

class Logger:
    def __init__(self, project, benchmark_run, config = None, log_epochs = 5, log_checkpoint = 50, checkpoint_dir = "."):
        self.config = config
        self.log_epochs = log_epochs
        self.log_checkpoint = log_checkpoint
        self.checkpoint_dir = checkpoint_dir
        if(project is not None):
            self.logger = wandb.init(project=project, name = benchmark_run.run_name, config=config)
            self.logger.config.update({"N_classes": benchmark_run.N_classes})
        else:
            self.logger = None

    def log(self, log_dict, step):
        if(self.logger is not None):
            self.logger.log(log_dict, step = step)

    def log_image_predictions(self, image, label, pred, epoch, split = "train"):
        """
        Evaluates the given model using the provided dataloader and computes the metrics.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing the validation data.
            model (torch.nn.Module): The model to be evaluated.
        Returns:
            dict: A dictionary containing the computed metric results.
        Notes:
            - The model is set to evaluation mode during the evaluation process.
            - The data is transferred to the appropriate device before making predictions.
            - The predictions are obtained by applying argmax on the model outputs.
            - The metrics are updated and computed based on the predictions and true labels.
        """
        
        if(self.logger is not None):
            log_dict = None
                                
            image = denormalize_image(image).transpose(1,2,0)
            correctness = color_correctness(label, pred)

            label = color_label(label)/255.
            pred = color_label(pred)/255.

            if(image.shape[-2:] == label.shape[-2:]):
                log_dict = {
                    f"{split}/imgs/image": wandb.Image(image, caption="Image"),
                    f"{split}/imgs/label": wandb.Image(0.4 * image/255. + 0.6 * label, caption="Label"),
                    f"{split}/imgs/prediction": wandb.Image(0.4 * image/255. + 0.6 * pred, caption="Prediction"),
                    f"{split}/imgs/correctness": wandb.Image(0.4 * image/255. + 0.6 * correctness, caption="Correctness"),
                        }
            else:
                log_dict = {
                    f"{split}/imgs/image": wandb.Image(image, caption="Image"),
                    f"{split}/imgs/label": wandb.Image(0.6 * label, caption="Label"),
                    f"{split}/imgs/prediction": wandb.Image(0.6 * pred, caption="Prediction"),
                    f"{split}/imgs/correctness": wandb.Image(0.6 * correctness, caption="Correctness"),
                        }
            self.logger.log(log_dict, step=epoch)

def save_benchmark_run(benchmark_run, benchmark_metrics):
    """
    Save the attributes of a benchmark run and its metrics to a JSON file.
    Args:
        benchmark_run (object): An object containing the attributes of the benchmark run.
            Expected attributes:
                - run_name (str): The name of the benchmark run.
                - model_name (str): The name of the model used in the benchmark run.
                - N_classes (int): The number of classes in the dataset.
                - device (str): The device used for the benchmark run (e.g., 'cpu' or 'cuda').
                - training_hyperparameters (str): The hyperparameters used for training.
        benchmark_metrics (dict): A dictionary containing the metrics of the benchmark run.
            Expected keys:
                - validation_loss (float): The validation loss.
                - validation_mean_iou (float): The mean Intersection over Union (IoU) on the validation set.
                - validation_mean_accuracy (float): The accuracy on the validation set.
                - best_epoch (int): The epoch number with the best performance.
    Returns:
        None
    """

    benchmark_run_attributes = {
        'run_name': benchmark_run.run_name,
        'model_name': benchmark_run.model_name,
        'N_classes': int(benchmark_run.N_classes),
        'device': str(benchmark_run.device),
        'training_hyperparameters': str(benchmark_run.training_hyperparameters),
        "validation_loss": str(np.round(benchmark_metrics["validation_loss"], 3)),
        "validation_mean_iou": str(np.round(benchmark_metrics["validation_mean_iou"], 3)),
        "validation_mean_accuracy": str(np.round(benchmark_metrics["validation_mean_accuracy"], 3)),
        "best_epoch": int(benchmark_metrics["best_epoch"])
        }
    
    # Create directory if it does not exist
    if not os.path.exists('benchmark_run_results'):
        os.makedirs('benchmark_run_results')
    # Save attributes to a JSON file
    with open(f'benchmark_run_results/{benchmark_run.run_name}.json', 'w') as json_file:
        json.dump(benchmark_run_attributes, json_file, indent=4)

def save_model_checkpoint(benchmark_run, epoch, loss, vloss, val_mean_iou, val_mean_accuracy, logger, final_checkpoint = False):
    timestamp = time.strftime('%Y%m%d%H')

    checkpoint = {
        # Model and training state
        "model_state_dict": benchmark_run.model.state_dict(),
        "optimizer_state_dict": benchmark_run.optimizer.state_dict(),
        "epoch": epoch,
        
        # # Hyperparameters and config
        # "config": logger.config,  # Your YAML/argparse config dict
        
        # Performance metrics
        "loss": loss,
        "val_metrics": {
            "validation_loss": vloss,
            "validation_mean_iou": val_mean_iou,
            "validation_mean_accuracy": val_mean_accuracy,
        },
        
        # Additional metadata (optional)
        "timestamp": timestamp
    }    
    
    model_path = f'{logger.checkpoint_dir}/model_checkpoints/{benchmark_run.run_name}/model_{timestamp}'
    if(epoch%logger.log_checkpoint==0 and epoch>0):
        model_path = f'{logger.checkpoint_dir}/model_checkpoints/{benchmark_run.run_name}/model_epoch{epoch}'
    if(final_checkpoint):
        model_path = f'{logger.checkpoint_dir}/model_checkpoints/{benchmark_run.run_name}/model_epoch{epoch}_final'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(checkpoint, model_path)
