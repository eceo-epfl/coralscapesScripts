import numpy as np 

from matplotlib import pyplot as plt

from coralscapesScripts.datasets.dataset import Coralscapes

id_to_color_map = lambda x: Coralscapes.train_id_to_color[x]
    
def denormalize_image(image, mean = np.array([0.485, 0.456, 0.406]), std = np.array([0.229, 0.224, 0.225])):
    """
    Denormalizes an image that was previously normalized using the given mean and standard deviation.
    Args:
        image (numpy.ndarray): The normalized image to be denormalized. Expected shape is (C, H, W).
        mean (numpy.ndarray, optional): The mean used for normalization. Default is np.array([0.485, 0.456, 0.406]) as used in ImageNet.
        std (numpy.ndarray, optional): The standard deviation used for normalization. Default is np.array([0.229, 0.224, 0.225]) as used in ImageNet.
    Returns:
        numpy.ndarray: The denormalized image with pixel values in the range [0, 255] and dtype uint8.
    """

    unnormalized_image = (image * std[:, None, None]) + mean[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    return unnormalized_image

def color_label(label):
    label_colors = np.array([[id_to_color_map(pixel) for pixel in row] for row in np.array(label)])
    return label_colors

def color_correctness(label, pred):
    semseg = np.zeros((pred.shape[0], pred.shape[1], 3))
    semseg[pred==label] = np.array([0, 0, 1])
    semseg[pred!=label] = np.array([1, 0, 0])
    semseg[label==0] = 1
    return semseg

def show_samples(dataset, denormalize = True, n: int = 2):
    """
    Display n sample images and their corresponding segmentation maps from a dataset.
    Parameters:
    -----------
    dataset : Dataset
        The dataset from which to retrieve the samples. The dataset should support
        indexing and return either a tuple (image, label) or an object with attributes
        `transformed_image` and `transformed_segmentation_map`.
    denormalize : bool, optional
        If True, the images will be denormalize before displaying. Default is True.
    n : int, optional
        The number of samples to display. Default is 3.
    """

    if n > len(dataset):
        raise ValueError("n is larger than the dataset size")

    fig, ax = plt.subplots(n, 2, figsize=(10, 3 * n))

    for i in range(n):
        image, label = dataset.__getitem__(i)
        label_colors = color_label(label)
        if(denormalize and np.min(image)<0):
            image = denormalize_image(image)
        image = image.transpose(1, 2, 0)

        ax[i, 0].imshow(image)
        ax[i, 0].set_title("Image")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(image)
        ax[i, 1].imshow(label_colors, alpha=0.4)
        ax[i, 1].set_title("Segmentation Map")
        ax[i, 1].axis("off")

    plt.tight_layout()
    plt.show()