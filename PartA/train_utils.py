import torch  # PyTorch library
from torch.cuda.amp import autocast  # For mixed precision (faster evaluation on GPU)

# Function to update accuracy statistics
def update_accuracy_stats(outputs, labels):
    _, predicted = torch.max(outputs, 1)  # Get predicted class (index of max logit)
    correct = (predicted == labels).sum().item()  # Count how many predictions were correct
    return correct, labels.size(0)  # Return correct count and total count

# Function to train the model for one epoch
def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1):
    model.train()  # Set model to training mode
    total_loss, correct, total = 0, 0, 0  # Initialize tracking variables
    optimizer.zero_grad()  # Clear any existing gradients

    for i, (inputs, labels) in enumerate(loader):  # Loop over mini-batches
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU or CPU
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss for gradient accumulation
        loss.backward()  # Backward pass (compute gradients)

        # Perform optimizer step every 'accumulation_steps' batches
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # Update model weights
            optimizer.zero_grad()  # Reset gradients

        total_loss += loss.item() * accumulation_steps  # Accumulate total loss
        c, t = update_accuracy_stats(outputs, labels)  # Update accuracy stats
        correct += c
        total += t

    # Return average loss and accuracy for the epoch
    return total_loss / len(loader), 100.0 * correct / total

# Function to evaluate model performance (no gradient updates)
def evaluate(model, loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss, correct, total = 0, 0, 0  # Initialize tracking variables

    with torch.no_grad():  # Disable gradient calculation (saves memory)
        with autocast(device_type='cuda'):  # Use mixed precision for faster evaluation on GPU
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                total_loss += loss.item()  # Accumulate loss
                c, t = update_accuracy_stats(outputs, labels)  # Update accuracy stats
                correct += c
                total += t

    # Return average loss and accuracy
    return total_loss / len(loader), 100.0 * correct / total
