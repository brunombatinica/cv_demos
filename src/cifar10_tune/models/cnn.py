import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#CIFAR10 inputs are [batchsize, 3, 32, 32]

class CNN(nn.Module):
    def __init__(self, conv_layers: int = None, hidden_dim: int = None, dropout: float = None):
        '''
        CNN model
        args:
            conv_layers: int, number of conv layers
            hidden_dim: int, hidden dimension
            dropout: float, dropout rate
        '''
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1), # [3, 32, 32] -> [32, 32, 32] #kernel 3 and padding 1 to keep the size
            nn.MaxPool2d(kernel_size=2, stride=2), # [32, 32, 32] -> [32, 16, 16] #maxpooling to reduce the size 2 on each axis
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            *(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ) for _ in range(conv_layers - 1))
        )
        # size out output will depend on number of conv layers and hidden dim 
        # hidden dim * 32 * 32 / 4^(conv_layers)
        self.fc1 = nn.Linear(int((hidden_dim*32*32)/(4**conv_layers)), 10) 
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x




