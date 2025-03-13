import numpy as np
from PIL import Image 

from torch.utils.data import Dataset

import os 
import json
from collections import namedtuple

class Coralscapes(Dataset):
    # Based on https://github.com/mcordts/cityscapesScripts
    # CoralscapesClass = namedtuple('CoralscapesClass', ['name', 'id', 'train_id', 'category', 'category_id', 'ignore_in_eval', 'color'])
    # classes = []
    # classes.append(CoralscapesClass('unlabeled', 0, 0, 'placeholder', 0, True, (255, 255, 255)))
    # for class_ in coralscapes_classes.keys():
    #     classes.append(CoralscapesClass(class_, coralscapes_classes[class_], coralscapes_classes[class_], "placeholder", 0, False, coralscapes_colors[class_]))

    # train_id_to_color = np.array([c.color for c in classes])

    def __init__(self, root = "../../coralscapes", split='train', transform=None, transform_target=True):
        """
        Initialize the dataset.
        Args:
            root (str): Root directory of the dataset.
            split (str, optional): The dataset split, one of 'train', 'test', or 'val'. Default is 'train'.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default is None.
            transform_target (bool, optional): Whether to also transform the segmentation mask, as opposed to only the input image. Default is True.

        Attributes:
            root (str): Expanded user path of the root directory.
            mode (str): Mode of the dataset, set to 'gtFine' which contains the semantic segmentation labels.
            images_dir (str): Directory path for images.
            targets_dir (str): Directory path for target annotations.
            transform (callable): Transform function for images.
            transform_target (bool): Whether to transform the mask.
            N_classes (int): Number of classes in the dataset.
            id2label (dict): Mapping of class IDs to class names.
            label2id (dict): Mapping of class names to class IDs.
            split (str): The dataset split.
            images (list): List of image file paths.
            targets (list): List of target file paths.
        """
        global coralscapes_classes
        global coralscapes_colors

        with open(f'{root}/classes.json', 'r') as file:
            coralscapes_classes = json.load(file)
            coralscapes_classes = dict(sorted(coralscapes_classes.items(), key=lambda item: item[1]))

        with open(f'{root}/colors.json', 'r') as file:
            coralscapes_colors = json.load(file)

        Coralscapes.CoralscapesClass = namedtuple('CoralscapesClass', ['name', 'id', 'train_id', 'category', 'category_id', 'ignore_in_eval', 'color'])
        Coralscapes.classes = []
        Coralscapes.classes.append(Coralscapes.CoralscapesClass('unlabeled', 0, 0, 'placeholder', 0, True, (255, 255, 255)))
        for class_ in coralscapes_classes.keys():
            Coralscapes.classes.append(Coralscapes.CoralscapesClass(class_, coralscapes_classes[class_], coralscapes_classes[class_], "placeholder", 0, False, coralscapes_colors[class_]))

        Coralscapes.train_id_to_color = np.array([c.color for c in Coralscapes.classes])

        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.images_dir = os.path.join(self.root, 'leftImg8bit/', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform
        self.transform_target = transform_target
        self.N_classes = int(1+np.sum([dataset_class.ignore_in_eval==False for dataset_class in self.classes]))
        self.id2label = {dataset_class.id:dataset_class.name for dataset_class in self.classes}
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_file_name = file_name.replace("leftImg8bit", "gtFine")
                self.targets.append(os.path.join(target_dir, target_file_name))

    def __getitem__(self, index):
        """
        Retrieve and transform an image and its corresponding segmentation map by index.
        Args:
            index (int): Index of the image and target to retrieve.
        Returns:
            tuple: A tuple containing:
            - image (numpy.ndarray): The transformed image.
            - target (numpy.ndarray): The transformed segmentation map.
        """

        image = np.array(Image.open(self.images[index]).convert('RGB'))
        target = np.array(Image.open(self.targets[index]))
        
        if self.transform:
            if self.transform_target:
                transformed = self.transform(image=image, mask=target)
                image = transformed["image"].transpose(2, 0, 1)  
                target = transformed["mask"]
            else:
                transformed = self.transform(image=image)
                image = transformed["image"].transpose(2, 0, 1)  
        return image, target

    def __len__(self):
        return len(self.images)

