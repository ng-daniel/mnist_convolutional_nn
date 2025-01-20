from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# loading datasets and dataloaders

# downloads the dataset from the pytorch torchvision library
# into training and testing sets
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

# creating training and testing dataloaders of batch size 32
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,
)