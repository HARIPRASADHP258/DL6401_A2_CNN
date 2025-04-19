import torch  # PyTorch main library
import torch.nn as nn  # For building neural network layers
import torch.nn.functional as F  # For activation functions and other utilities
from config import base_config  # Importing default configuration dictionary

# Define a Convolutional Neural Network class
class Conv_NN(nn.Module):
    def __init__(self, config):  # Constructor takes a config dictionary
        super().__init__()  # Initialize the parent class
        self.config = {**base_config, **config}  # Merge base config with the passed config

        # Make sure there are 5 conv filters and filter sizes specified
        assert len(self.config['conv_filters']) == 5
        assert len(self.config['filter_sizes']) == 5

        in_channels = 3  # Input image has 3 channels (RGB)
        # Lists to hold convolution, batchnorm, and pooling layers
        self.conv_layers, self.bn_layers, self.pool_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        # Create 5 convolutional blocks
        for i in range(5):
            out_channels = self.config['conv_filters'][i]  # Number of filters
            kernel_size = self.config['filter_sizes'][i]  # Size of the filter
            # Add a convolutional layer with padding to keep size
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            self.conv_layers.append(conv)  # Add to conv layers list
            # Add batchnorm layer if enabled
            self.bn_layers.append(nn.BatchNorm2d(out_channels) if self.config['use_batchnorm'] else None)
            self.pool_layers.append(nn.MaxPool2d(2, 2))  # Add max pooling layer
            in_channels = out_channels  # Update for the next layer

        # Estimate the size after convolutions to create dense layers
        with torch.no_grad():  # No gradient computation needed
            dummy = torch.rand(1, 3, 224, 224)  # Create dummy input
            for i in range(5):
                dummy = self.conv_layers[i](dummy)  # Apply conv
                if self.bn_layers[i]: dummy = self.bn_layers[i](dummy)  # Apply batchnorm
                dummy = self.pool_layers[i](self._get_activation(dummy))  # Apply activation and pooling
            self.flattened_size = dummy.view(1, -1).size(1)  # Flatten and get the feature size

        # Fully connected (dense) layers
        self.fc1 = nn.Linear(self.flattened_size, self.config['dense_neurons'])  # First dense layer
        self.fc2 = nn.Linear(self.config['dense_neurons'], 10)  # Output layer (10 classes)
        # Dropout for regularization if specified
        self.dropout = nn.Dropout(self.config['dense_dropout']) if self.config.get('dense_dropout') else None

    # Helper function to get the selected activation function
    def _get_activation(self, x):
        act = self.config.get('activation', 'relu')  # Get activation name from config
        return {
            'relu': F.relu, 'leaky_relu': lambda x: F.leaky_relu(x, 0.1),
            'elu': F.elu, 'silu': F.silu, 'mish': F.mish,
            'gelu': F.gelu, 'swish': lambda x: x * torch.sigmoid(x)
        }.get(act, F.relu)(x)  # Apply the selected activation (default: relu)

    # Helper function to pass input through all convolutional blocks
    def _forward_convs(self, x):
        for i in range(5):
            x = self.conv_layers[i](x)  # Apply conv
            if self.bn_layers[i]: x = self.bn_layers[i](x)  # Apply batchnorm if used
            x = self.pool_layers[i](self._get_activation(x))  # Apply activation and pooling
        return x  # Return processed output

    # Main forward function for the model
    def forward(self, x):
        x = self._forward_convs(x)  # Pass through convolutional layers
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self._get_activation(self.fc1(x))  # First dense layer + activation
        if self.dropout: x = self.dropout(x)  # Apply dropout if enabled
        return F.log_softmax(self.fc2(x), dim=1)  # Final output with log-softmax (for classification)
