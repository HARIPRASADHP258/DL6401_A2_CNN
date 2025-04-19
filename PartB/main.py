# Import required libraries
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import custom modules
from config import base_config
from data import create_datasets
from model import load_model
from train_utils import train_epoch, evaluate

# Login to with (W&B)  API key
wandb.login(key="999fe4f321204bd8f10135f3e40de296c23050f9")

# Function to train a model using sweep parameters
def train_sweep():
    with wandb.init() as run:
        # Combine base config with sweep config
        sweep_config = dict(wandb.config)
        config = {**base_config, **sweep_config}

        # Set a descriptive name for this run
        run.name = (
            f"model_{config['model_name']}-"
            f"freeze_{config['freeze_percentage']}-"
            f"bs_{config['batch_size']}-"
            f"lr_{config['learning_rate']}-"
            f"ep_{config['epochs']}-"
            f"wd_{config['weight_decay']}-"
            f"aug_{config['use_augmentation']}"
        )

        # Create datasets and data loaders
        train_data, val_data, test_data = create_datasets(config)
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=config['batch_size'], num_workers=2, pin_memory=True)

        # Load the model and move it to GPU if available
        model = load_model(config['model_name'], num_classes=len(train_data.dataset.classes), freeze_percentage=config['freeze_percentage'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Define optimizer, loss function, and scaler for mixed precision training
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        best_val_acc = 0.0  # To keep track of the best validation accuracy
        best_model_path = "best_model.pth"  # File to save the best model

        # Training loop
        for epoch in range(config['epochs']):
            # Train for one epoch
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler, config['gradient_accumulation_steps']
            )
            
            # Evaluate on validation set
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            # Log results to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

            # Save the model if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)

            # Print progress
            print(f"Epoch {epoch+1}/{config['epochs']} - "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # Evaluate the final model on test data
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.2f}%")

        # Log test results and best model path
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        wandb.log({"best_model_path": best_model_path})

# Run the sweep when this script is executed
if __name__ == "__main__":
    # Define hyperparameter sweep configuration
    sweep_config = {
        'method': 'bayes',  # Use Bayesian optimization
        'name': 'transfer-learning-sweep',  # Sweep name
        'metric': {'goal': 'maximize', 'name': 'val_acc'},  # Goal is to maximize validation accuracy
        'parameters': {  # List of hyperparameters to tune
            'model_name': {'values': ['resnet50', 'efficientnet_v2_s', 'vgg16', 'googlenet']},
            'freeze_percentage': {'values': [0.7, 0.8, 0.85, 0.9]},
            'learning_rate': {'values': [1e-5, 5e-5, 1e-4]},
            'batch_size': {'values': [32, 64]},
            'epochs': {'values': [10, 15]},
            'weight_decay': {'values': [0.001, 0.005]},
            'use_augmentation': {'values': [False]}
        }
    }

    # Create the sweep and start running it
    sweep_id = wandb.sweep(sweep_config, project="inaturalist-transfer-learning")
    wandb.agent(sweep_id, function=train_sweep, count=15)  # Run 15 experiments
    wandb.finish()
