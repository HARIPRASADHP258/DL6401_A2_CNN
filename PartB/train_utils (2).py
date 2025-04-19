import torch
# AMP (Automatic Mixed Precision) tools for faster training with less memory
from torch.cuda.amp import autocast, GradScaler

# Helper function to compute number of correct predictions in a batch
def update_accuracy_stats(outputs, labels):
    _, predicted = torch.max(outputs, 1)  # Get predicted class with highest score
    correct = (predicted == labels).sum().item()  # Count correct predictions
    total = labels.size(0)  # Total samples in batch
    return correct, total

# Function to train the model for one epoch
def train_epoch(model, loader, criterion, optimizer, device, scaler, accumulation_steps=1):
    model.train()  # Set model to training mode
    running_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()  # Reset gradients

    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU

        # Use mixed precision for faster training
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss for gradient accumulation

        # Scale the loss and backpropagate
        scaler.scale(loss).backward()

        # Perform optimizer step after 'accumulation_steps' mini-batches
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # Update weights
            scaler.update()         # Update scaler for next iteration
            optimizer.zero_grad()   # Reset gradients

        # Update running loss and accuracy
        running_loss += loss.item() * accumulation_steps * inputs.size(0)
        batch_correct, batch_total = update_accuracy_stats(outputs, labels)
        correct += batch_correct
        total += batch_total

    # Return average loss and accuracy for the epoch
    return running_loss / total, 100.0 * correct / total

# Function to evaluate the model on validation or test data
def evaluate(model, loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        with autocast(device_type='cuda', dtype=torch.float16):  # Use mixed precision
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                batch_correct, batch_total = update_accuracy_stats(outputs, labels)
                correct += batch_correct
                total += batch_total

    # Return average loss and accuracy
    return running_loss / total, 100.0 * correct / total
