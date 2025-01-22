from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
Loads datasets and dataloaders via the torchvision library 
and Pytorch DataLoaders
"""

def download_MNIST():
    """
    Downloads the dataset from the torch datasets library to the data folder
    if they aren't already dowloaded.

    Returns:
        A tuple containing the training and testing datasets.
    """
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transforms.ToTensor(), #turns images into pytorch tensors
        target_transform=None,
        download=True,
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transforms.ToTensor(),
        target_transform=None,
        download=True,
    )
    return train_dataset, test_dataset

def make_dataloaders(
        train_dataset: datasets,
        test_dataset: datasets,
        batch_size: int,
        shuffle: bool):
    """
    Creating pytorch dataloaders.

    Args:
        train_dataset: the training dataset
        test_dataset: the testing dataset
        batch_size: the number of images and labels per training batch
        num_workers: number of subprocesses per dataloader
        shuffle: whether to shuffle the data or not
    
    Returns:
        A tuple containing the training and testing dataloaders
    """

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False
    )
    return train_dataloader, test_dataloader