import numpy as np
import torch 

from tqdm import tqdm

def calculate_weights(dataset, const = 2000000):
    """
    Calculate class weights for a given dataset.
    This function computes the weights for each class in the dataset based on 
    the frequency of each class label. The weights are inversely proportional 
    to the square root of the class frequency, adjusted by a constant value.
    Args:
        dataset (Dataset): The dataset object which contains images and labels. 
                            It should have an attribute `N_classes` indicating 
                            the number of classes.
        const (int, optional): A constant value added to the class frequency 
                                to avoid division by zero and to smooth the weights. 
                                Default is 2000000.
    Returns:
        torch.Tensor: A tensor of weights for each class, normalized by the mean weight.
    """

    label_counts = {}
    label_counts = {i: 0 for i in range(dataset.N_classes)}
    for i in tqdm(range(len(dataset))):
        image, label = dataset[i]
        unique_labels = np.unique(label, return_counts=True)
        for label_id, count in zip(*unique_labels):
            label_counts[label_id] += int(count)

    weights = np.zeros(dataset.N_classes)
    for index, count in label_counts.items():
        weights[index] = 1 / np.sqrt(count + const)
    weight = torch.tensor(weights).float()
    weight /= weight.mean()

    return weight