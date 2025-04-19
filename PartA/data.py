from torchvision import transforms, datasets  # For image transformations and dataset loading
from torch.utils.data import random_split  # For splitting dataset into training and validation
import os  # For path operations

# Function to define image transformations
def data_transforms(config):
    # If data augmentation is enabled in config
    if config.get('use_augmentation', False):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomVerticalFlip(0.3),  # Random vertical flip with 30% probability
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize(config['mean'], config['std'])  # Normalize with given mean and std
        ])
    else:
        # Only resizing and normalization (no augmentation)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])
    
    # Validation/test transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std'])
    ])

    return train_transform, val_transform  # Return both transforms

# Function to create and split datasets
def create_datasets(config):
    train_transform, val_transform = data_transforms(config)  # Get transforms based on config

    # Load training data from 'train' folder with appropriate transform
    train_data = datasets.ImageFolder(os.path.join(config['data_dir'], 'train'), transform=train_transform)

    # Load validation/test data from 'val' folder with val_transform
    test_data = datasets.ImageFolder(os.path.join(config['data_dir'], 'val'), transform=val_transform)

    # Split 20% of training data to create validation dataset
    val_size = int(0.2 * len(train_data))
    train_data, val_data = random_split(train_data, [len(train_data) - val_size, val_size])

    return train_data, val_data, test_data  # Return all datasets
