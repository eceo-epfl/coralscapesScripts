import numpy as np 

import torch 

from transformers import SegformerImageProcessor, Mask2FormerImageProcessor, DPTImageProcessor, AutoImageProcessor

def get_windows_inference(inputs, window_size, stride):
    """
    Extracts sliding windows from the input tensor for inference.
    Args:
        inputs (torch.Tensor): The input tensor with shape (batch_size, channels, height, width).
        window_size (int): The size of the sliding window.
        stride (int): The stride of the sliding window.
    Returns:
        torch.Tensor: A tensor containing the extracted windows with shape 
                      (num_windows_y, num_windows_x, batch_size, channels, window_size, window_size),
                      where num_windows_y and num_windows_x are the number of windows along the height and width dimensions, respectively.
    """

    # Get the dimensions of the tensor
    _, _, height, width = inputs.shape
    # Initialize a list to store the windows
    input_windows = []
    # Iterate over the tensor with a sliding window
    for y in range(0, height, stride):
        input_windows_row = []
        for x in range(0, width, stride):
            input_window = inputs[:, :, y:y+window_size, x:x+window_size]
            input_windows_row.append(input_window)
        input_windows.append(input_windows_row)

    input_windows = torch.stack([torch.stack(row, dim=0) for row in input_windows], dim=0)
    return input_windows

def get_windows(batch, window_size, stride, window_size_label = None, stride_label = None):
    """
    Extracts sliding windows from input and label tensors.
    Args:
        batch (tuple): A tuple containing input and label tensors. 
                        The input tensor should have the shape (batch_size, channels, height, width).
                        The label tensor should have the shape (batch_size, height, width).
        window_size (int): The size of the sliding window for the input tensor.
        stride (int): The stride of the sliding window for the input tensor.
        window_size_label (int, optional): The size of the sliding window for the label tensor. 
                                            If None, the same window size as the input tensor is used. Default is None.
        stride_label (int, optional): The stride of the sliding window for the label tensor. 
                                        If None, the same stride as the input tensor is used. Default is None.
    Returns:
        tuple: A tuple containing two tensors:
                - input_windows (torch.Tensor): A tensor of shape (num_windows_y, num_windows_x, batch_size, channels, window_size, window_size)
                                                containing the extracted windows from the input tensor.
                - label_windows (torch.Tensor): A tensor of shape (num_windows_y, num_windows_x, batch_size, window_size, window_size)
                                                containing the extracted windows from the label tensor.
    """
    # Get the dimensions of the tensor
    inputs, labels = batch
    _, _, height, width = inputs.shape
    # Initialize a list to store the windows
    input_windows = []
    label_windows = []
    # Iterate over the tensor with a sliding window
    for y in range(0, height, stride):
        input_windows_row = []
        label_windows_row = []
        for x in range(0, width, stride):
            input_window = inputs[:, :, y:y+window_size, x:x+window_size]
            label_window = labels[:, y:y+window_size, x:x+window_size]
            input_windows_row.append(input_window)
            label_windows_row.append(label_window)
        input_windows.append(input_windows_row)
        label_windows.append(label_windows_row)

    if(window_size_label is not None and stride_label is not None):
        height, width = labels.shape[-2:]
        label_windows = []
        # Iterate over the tensor with a sliding window
        for y in range(0, height, stride_label):
            label_windows_row = []
            for x in range(0, width, stride_label):
                label_window = labels[:, y:y+window_size_label, x:x+window_size_label]
                label_windows_row.append(label_window)
            label_windows.append(label_windows_row)

    input_windows = torch.stack([torch.stack(row, dim=0) for row in input_windows], dim=0)
    label_windows = torch.stack([torch.stack(row, dim=0) for row in label_windows], dim=0)
    return input_windows, label_windows

def get_preprocessor(model_name):
    """
    Returns a preprocessor object for the specified model.
    Args:
        model_name (str): The name of the model for which the preprocessor is required.
    Returns:
        preprocessor: A huggingface preprocessor configured for the specified model.
    """
    default_preprocessor_kwargs = {"ignore_index": 0, 
                                   "do_reduce_labels": False,
                                   "do_resize": False, 
                                   "do_rescale": False, 
                                   "do_normalize": False,
                                   "num_labels": 40}
    if(model_name is None): # For models that don't need a preprocessor 
        return None
    if(model_name == "segformer"):
        preprocessor = SegformerImageProcessor(**default_preprocessor_kwargs)
    elif(model_name == "dpt"):
        preprocessor = DPTImageProcessor(**default_preprocessor_kwargs)
    else:
        raise ValueError("Model not found")
    return preprocessor

def collate_fn_hf_with_preprocessor(preprocessor):
    """
    Creates a collate function for handling batches of segmentation data with a given preprocessor.
    Args:
        preprocessor (callable): A function or callable object that preprocesses the images and segmentation maps.
    Returns:
        callable: A collate function that takes a batch of SegmentationDataInput and returns a dictionary with preprocessed data.
    The returned collate function performs the following steps:
        1. Extracts transformed images and segmentation maps from the batch.
        2. Applies the preprocessor to the transformed images and segmentation maps.
        3. Adds the segmentation maps to the preprocessed batch.
    """

    def collate_fn_hf(batch) -> dict:
        transformed_images = [sample[0] for sample in batch]
        transformed_segmentation_maps = [sample[1] for sample in batch]

        preprocessed_batch = preprocessor(
            transformed_images,
            segmentation_maps=transformed_segmentation_maps,
            return_tensors="pt",
        )
        preprocessed_batch["images"] = np.array(transformed_images)
        preprocessed_batch["masks"] = np.array(transformed_segmentation_maps)
        return preprocessed_batch
    return collate_fn_hf
   
def preprocess_batch(data, preprocessor = None):
    """
    Preprocess a batch of data using the provided preprocessor.
    Args:
        data (tuple): A tuple containing inputs and labels. 
                        - inputs: A list of images.
                        - labels: A list of segmentation maps corresponding to the images.
        preprocessor (callable, optional): A function or callable object that takes inputs and segmentation maps 
                                            and returns preprocessed data. If None, the data is returned as is.
    Returns:
        preprocessed_batch: The preprocessed batch of data if a preprocessor is provided, otherwise the original data.
    """

    if(preprocessor):
        inputs, labels = data
        preprocessed_batch = preprocessor(
            [image for image in inputs],
            segmentation_maps=[label for label in labels],
            return_tensors="pt",
        )
    else:
        preprocessed_batch = data
    return preprocessed_batch

def preprocess_inference(inference_image, transform, benchmark_run):
    """
    Preprocess an inference image for model input.
    Args:
        inference_image (numpy.ndarray): The image to be processed for inference.
        transform (callable): A transformation function to apply to the image.
        benchmark_run (object): An object containing evaluation and preprocessing configurations.
    Returns:
        tuple: A tuple containing:
            - preprocessed_batch (torch.Tensor): The preprocessed image or batch of images.
            - window_dims (tuple or None): The dimensions of the sliding windows if used, otherwise None.
    """
    
    if transform:
        transformed = transform(image=inference_image)
        inference_image = transformed["image"].transpose(2, 0, 1)  
        
    inference_image = torch.tensor(inference_image).unsqueeze(0)

    window_dims = None 

    if(benchmark_run.eval and benchmark_run.eval.sliding_window):
        input_windows = get_windows_inference(inference_image, 
                                              benchmark_run.eval.window, 
                                              benchmark_run.eval.stride)
        window_dims = input_windows.shape[:3]
        input_windows = input_windows.view(-1, *input_windows.shape[-3:])
        inference_image = input_windows
    
    if(benchmark_run.preprocessor):
        preprocessed_batch = benchmark_run.preprocessor(
            [image for image in inference_image],
            return_tensors="pt",
        )
    else:
        preprocessed_batch = inference_image
        
    return preprocessed_batch, window_dims