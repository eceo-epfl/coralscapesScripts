import numpy as np 

from coralscapesScripts.segmentation.eval import Evaluator
from coralscapesScripts.logger import save_model_checkpoint

import time

def launch_benchmark(train_loader, val_loader, test_loader, benchmark_run, logger = None, start_epoch:int = None, end_epoch:int = None):
    """
    Launches the benchmarking process for a segmentation model.
    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        benchmark_run (object): An object containing the model, training hyperparameters, 
                                scheduler, device, and other necessary attributes for benchmarking.
        logger (wandb): Weights and Biases logger for tracking and visualizing metrics.
        start_epoch (int): The epoch number to start training from. If None, training starts from epoch 0.
        end_epoch (int): The epoch number to end training at. If None, training ends at the number of epochs specified in the training hyperparameters.
    Returns:
        dict: A dictionary containing the best validation loss, mean IoU, mean accuracy, 
              and the epoch number at which these best metrics were achieved.
    """

    best_val_mean_iou = 0.
    best_epoch = -1

    if(start_epoch is None):
        start_epoch = 0
        end_epoch = benchmark_run.training_hyperparameters["epochs"]

    if hasattr(benchmark_run, "preprocessor"):
        evaluator = Evaluator(N_classes = benchmark_run.N_classes, device = benchmark_run.device, preprocessor = benchmark_run.preprocessor, eval_params=benchmark_run.eval)  
    else:
        evaluator = Evaluator(N_classes = benchmark_run.N_classes, device = benchmark_run.device, eval_params=benchmark_run.eval)  

    for epoch in range(start_epoch, end_epoch):
        t = time.time()
        print('EPOCH {}:'.format(epoch + 1))
        benchmark_run.model.train(True)
        if hasattr(benchmark_run.optimizer, 'train'): # For use with ScheduleFree optimizers
            benchmark_run.optimizer.train()

        train_loss = benchmark_run.train_epoch(train_loader)
        
        if hasattr(benchmark_run, "scheduler"):
            benchmark_run.scheduler.step()
        print('LOSS train {}'.format(train_loss))

        benchmark_run.model.eval()
        if hasattr(benchmark_run.optimizer, 'train'): # For use with ScheduleFree optimizers
            benchmark_run.optimizer.eval()
        val_loss = benchmark_run.validation_epoch(val_loader)

        print('LOSS valid {}'.format(val_loss))

        if(logger):
            logger.log({
                            "train/loss": train_loss, 
                            "validation/loss": val_loss, 
                            "train/time_taken": time.time() - t
                            }, 
                        step=epoch)

            if(epoch%logger.log_epochs==0 or epoch == end_epoch-1):

                train_metric_results = evaluator.evaluate_model(train_loader, benchmark_run.model, split = "train")
                logger.log(
                        {f"train/{metric_name}":metric for metric_name, metric in train_metric_results.items()},
                        step=epoch)
                print("Train metrics")
                print(train_metric_results)

                metric_results = evaluator.evaluate_model(val_loader, benchmark_run.model, split = "validation")
                logger.log(
                        {f"validation/{metric_name}":metric for metric_name, metric in metric_results.items()},
                        step=epoch)
                print("Validation metrics")
                print(metric_results)

                if(logger.logger):
                    logger.log_image_predictions(*evaluator.evaluate_image(train_loader, benchmark_run.model, split = "train", epoch = epoch), epoch, split = "train")
                    logger.log_image_predictions(*evaluator.evaluate_image(val_loader, benchmark_run.model, split = "validation", epoch = epoch), epoch, split = "validation")

                if(test_loader is not None):
                    test_metric_results = evaluator.evaluate_model(test_loader, benchmark_run.model, split = "test")
                    logger.log(
                            {f"test/{metric_name}":metric for metric_name, metric in test_metric_results.items()},
                            step=epoch)
                    print("Test metrics")
                    print(test_metric_results)

                    if(logger.logger):
                        logger.log_image_predictions(*evaluator.evaluate_image(test_loader, benchmark_run.model, split = "test", epoch = epoch), epoch, split = "test")

            # Track best performance, and save the model's state
            if best_val_mean_iou < metric_results["mean_iou"]:
                best_vloss = val_loss
                best_val_mean_iou = metric_results["mean_iou"]
                best_val_mean_accuracy = metric_results["accuracy"]
                best_epoch = epoch

                save_model_checkpoint(benchmark_run, epoch, train_loss, 
                                      val_loss, best_val_mean_iou, best_val_mean_accuracy, 
                                      logger)

            if (epoch%logger.log_checkpoint==0) and epoch>0:
                best_vloss = val_loss
                best_val_mean_iou = metric_results["mean_iou"]
                best_val_mean_accuracy = metric_results["accuracy"]
                best_epoch = epoch

                save_model_checkpoint(benchmark_run, epoch, train_loss, 
                                      val_loss, best_val_mean_iou, best_val_mean_accuracy, 
                                      logger, final_checkpoint=False)
                
            if (epoch==(end_epoch-1)) and epoch>0:
                best_vloss = val_loss
                best_val_mean_iou = metric_results["mean_iou"]
                best_val_mean_accuracy = metric_results["accuracy"]
                best_epoch = epoch

                save_model_checkpoint(benchmark_run, epoch, train_loss, 
                                      val_loss, best_val_mean_iou, best_val_mean_accuracy, 
                                      logger, final_checkpoint=True)
                
    results_dict = {"validation_loss": best_vloss,
                    "validation_mean_iou": best_val_mean_iou,
                    "validation_mean_accuracy": best_val_mean_accuracy,
                    "best_epoch": best_epoch}
    
    return results_dict

