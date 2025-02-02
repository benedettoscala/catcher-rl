import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNQNetwork(nn.Module):
    def __init__(self, grid_size=20, action_size=3, in_channels=2):
        """
        Parametri:
        - grid_size: dimensione della griglia (es: 20)
        - action_size: numero di azioni (es: 3: LEFT, STAY, RIGHT)
        """
        super(CNNQNetwork, self).__init__()
        # Convoluzioni
        # Input: (batch_size, 2, grid_size, grid_size)
        print("in_channels: ", in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Pooling per ridurre dimensioni
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calcoliamo la dimensione dopo due conv2d e due pool2d
        # 1° conv+pool: input (2, grid_size, grid_size) --> output (16, grid_size/2, grid_size/2)
        # 2° conv+pool: input (16, grid_size/2, grid_size/2) --> output (32, grid_size/4, grid_size/4)
        conv_output_size = 32 * (grid_size // 4) * (grid_size // 4)

        # Layer fully connected
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        """
        Forward pass. 
        x ha dimensioni [batch_size, 2, grid_size, grid_size].
        """
        # Passaggio nelle convoluzioni + ReLU + pooling
        x = F.relu(self.conv1(x))  # out: (16, grid_size, grid_size)
        x = self.pool(x)          # out: (16, grid_size/2, grid_size/2)

        x = F.relu(self.conv2(x))  # out: (32, grid_size/2, grid_size/2)
        x = self.pool(x)           # out: (32, grid_size/4, grid_size/4)

        # Flatten per passare ai layer fully connected
        x = x.view(x.size(0), -1)   # out: (batch_size, 32*(grid_size/4)*(grid_size/4))

        # Due layer fully connected
        x = F.relu(self.fc1(x))     # out: (batch_size, 256)
        x = self.fc2(x)             # out: (batch_size, action_size)

        return x
    
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(QNetwork, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

