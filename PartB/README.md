# DL6401 Assignment 2 - Part B: Convolutional Neural Networks

This section of the repository contains code for Part B of DL6401 Assignment 2, focusing on image classification using Convolutional Neural Networks (CNNs).

## Files in This Directory

- `config.py`: Defines training configurations and hyperparameters.
- `data.py`: Contains functions for data loading and preprocessing.
- `main.py`: Main script to train and test the CNN model.
- `model.py`: Defines the CNN architecture.
- `train_utils.py`: Helper functions for training workflows.
- `README.md`: Project overview and instructions.

## Setup Instructions

### Requirements

- Python 3.6+
- PyTorch
- torchvision
- numpy, matplotlib 

### Installation

Clone the repository and navigate to PartB:

```bash
git clone https://github.com/HARIPRASADHP258/DL6401_A2_CNN.git
cd DL6401_A2_CNN/PartB
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually install required libraries.

## Model Description

The CNN is defined in `model.py` and includes:

- Multiple convolutional layers with ReLU activations
- Max pooling layers
- Fully connected layers
- Dropout for regularization

## Running the Model

Train the model using:

```bash
python "sweep.py"
```

Ensure the data loading and preprocessing functions in `data.py` are correctly configured for your dataset.

## Utilities

`train_utils.py` includes useful functions such as:

- Accuracy calculation
- Model checkpointing
- Training/validation tracking



