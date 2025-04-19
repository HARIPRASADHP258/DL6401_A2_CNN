import os
from torchvision import datasets, transforms
from torch.utils.data import random_split

# Function to define data augmentation and preprocessing transforms
def data_transforms(config):
    # Apply augmentation if enabled in config
    if config.get('use_augmentation', False):
        train_transform = transforms.Compose([
            transforms.Resize(256),                       # Resize image to 256x256
            transforms.RandomCrop(224),                   # Randomly crop to 224x224
            transforms.RandomHorizontalFlip(),            # Random horizontal flip
            transforms.RandomVerticalFlip(p=0.3),         # Random vertical flip with 30% probability
            transforms.ToTensor(),                        # Convert image to PyTorch tensor
            transforms.Normalize(mean=config['mean'], std=config['std'])  # Normalize using given mean and std
        ])
    else:
        # Basic preprocessing without augmentation
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),                   # Center crop to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std'])
        ])

    # Validation/test transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    return train_transform, val_transform

# Function to create training, validation, and test datasets
def create_datasets(config):
    train_transform, val_transform = data_transforms(config)

    # Load training data with appropriate transform
    train_data = datasets.ImageFolder(os.path.join(config['data_dir'], 'train'), transform=train_transform)

    # Load validation/test data (usually from 'val' folder)
    test_data = datasets.ImageFolder(os.path.join(config['data_dir'], 'val'), transform=val_transform)

    # Split training data into train and validation sets (80-20 split)
    val_size = int(0.2 * len(train_data))
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    return train_data, val_data, test_data
