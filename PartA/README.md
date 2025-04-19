# DL6401 Assignment 2 - Part A: Convolutional Neural Networks

This repository contains the implementation of Part A of Assignment 2 for the DL6401 course. The focus is on building and training Convolutional Neural Networks (CNNs) for image classification tasks.

##  Directory Structure

- `config.py`: Contains configuration settings and hyperparameters for training.
- `data.py`: Handles data loading and preprocessing.
- `main.py`: The main script to initiate training and evaluation.
- `model.py`: Defines the CNN architecture.
- `sweep.py`: Implements hyperparameter tuning using sweep methods.
- `train_utils.py`: Utility functions to support the training process.
- `README.md`: This file.



### Prerequisites

Ensure you have the following installed:

- Python 3.6 or higher
- PyTorch
- torchvision
- Other dependencies as required

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/HARIPRASADHP258/DL6401_A2_CNN.git
   cd DL6401_A2_CNN/PartA
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: If `requirements.txt` is not present, manually install the necessary packages.*

##  Model Overview

The CNN model is defined in `model.py` and is designed for image classification tasks. It includes:

- Multiple convolutional layers
- Activation functions (e.g., ReLU)
- Pooling layers
- Fully connected layers
- Dropout for regularization

##  Training and Evaluation

To train the model:

```bash
python sweep.py
```

This script will:

- Load and preprocess the dataset using `data.py`
- Initialize the model from `model.py`
- Train the model using configurations from `config.py`
- Evaluate the model's performance

## Hyperparameter Tuning

The `sweep.py` script allows for hyperparameter tuning:

```bash
python sweep.py
```

This will perform a sweep over specified hyperparameters to optimize model performance.

##  Utilities

The `train_utils.py` file contains helper functions to facilitate the training process, such as:

- Accuracy calculations
- Model saving/loading
- Logging training progress


