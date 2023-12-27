import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            4608, 512
        )  # Adjusted the input size based on the flattened tensor size
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 128)
        self.batchnorm6 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))

        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))

        x = self.flatten(x)

        x = F.relu(self.batchnorm5(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm6(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
