import torch
from torch import nn

# NOTE: This will be the network architecture.


class Net(nn.Module):
    def __init__(self, nClasses):
        super(Net, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Convolutional layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Fully connected layer 1
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        # Fully connected layer 2
        self.fc2 = nn.Linear(256, nClasses)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Activation function
        self.relu = nn.ReLU()
        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.pool(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
