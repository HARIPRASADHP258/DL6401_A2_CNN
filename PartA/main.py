import os  # Provides functions to interact with the operating system
import wandb  # Imports Weights & Biases for experiment tracking
import torch  # Imports PyTorch for deep learning
import torch.nn as nn  # Imports PyTorch's neural network module
from torch.utils.data import DataLoader  # Imports DataLoader to load data in batches
from model import Conv_NN  # Imports the custom CNN model from the model file
from data import create_datasets  # Imports function to create train/val/test datasets
from train_utils import train_epoch, evaluate  # Imports training and evaluation functions
from config import base_config  # Imports base configuration dictionary

wandb.login(key="999fe4f321204bd8f10135f3e40de296c23050f9")  # Logs in to wandb using the API key

def train(config, run=None):  # Defines a train function with config and optional wandb run
    config = {**base_config, **config}  # Merges base config with the passed config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sets device to GPU if available, else CPU

    if run is None:
        run = wandb.init(project="inaturalist-classify", config=config)  # Initializes wandb run if not already passed

    # Create training, validation, and test datasets
    train_data, val_data, test_data = create_datasets(config)
    
    # Create DataLoaders to load data in batches
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])
    test_loader = DataLoader(test_data, batch_size=config['batch_size'])

    # Initialize model and move it to the selected device
    model = Conv_NN(config).to(device)
    
    # Define the optimizer (Adam) and loss function (CrossEntropy)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_model_path = 0.0, "best_model.pth"  # Initialize best validation accuracy and model save path

    for epoch in range(config['epochs']):  # Loop over number of epochs
        # Train the model for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config['gradient_accumulation_steps'])
        # Evaluate the model on the validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Log metrics to wandb
        run.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                 "val_loss": val_loss, "val_acc": val_acc})

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        # Print training progress
        print(f"Epoch {epoch+1}/{config['epochs']} - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Evaluate the best model on the test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")  # Print final test accuracy
    run.log({"test_loss": test_loss, "test_acc": test_acc})  # Log test results to wandb

    wandb.finish()  # Finish the wandb run
    return best_model_path  # Return the path to the best saved model
