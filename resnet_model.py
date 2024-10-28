from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import functional as F
import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity mapping)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if stride != 1 or in_channels != out_channels else None
        self.bn_skip = nn.BatchNorm2d(out_channels) if self.skip_connection else None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply skip connection if dimensions differ
        if self.skip_connection:
            identity = self.bn_skip(self.skip_connection(x))

        out += identity
        return F.relu(out)
    

class ResNetClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ResNetClassifier, self).__init__()
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)
        
        # Update layer4 with modified bottleneck blocks
        self.model.layer4 = nn.Sequential(
            BottleneckBlock(in_channels=256, out_channels=256, stride=2),
            BottleneckBlock(in_channels=256, out_channels=num_features, stride=1)
        )

        # Modify the first convolutional layer to accept 1 input channel instead of 3
        # The original conv1 has in_channels=3; we change this to in_channels=1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,   # Grayscale images have 1 channel
            out_channels=64, # Keep the same number of output channels
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Modify the fully connected layer to fit the output of layer4
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)