import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np


def tinyimagenet_get_datasets(data_dir):
    """
    Load the Tiny ImageNet dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) # using mean and std of original imagenet dataset

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(56),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.CenterCrop(56),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset

