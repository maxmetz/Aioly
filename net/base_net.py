import torch
import torch.nn as nn
import torch.nn.functional as F

class CuiNet(nn.Module):
    def __init__(self, input_dims, mean,std):
        super(CuiNet, self).__init__()
        
        # Layers dimensions
        self.conv1d_dims = input_dims
        self.k_number = 1
        self.k_width = 5
        self.k_stride = 1
        self.fc1_dims = 36
        self.fc2_dims = 18
        self.fc3_dims = 12
        self.out_dims = 1
        self.mean = nn.Parameter(torch.tensor(mean).float())
        self.std = nn.Parameter(torch.tensor(std).float())
        
        # Convolutional layer
        self.conv1d = nn.Conv1d(1,1, kernel_size= 5 , stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv1d_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.out = nn.Linear(self.fc3_dims, self.out_dims)
        
        # Initialize weights with He normal
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Reshape input
        x = (x-self.mean)/self.std
        # Convolutional layer with ELU activation
        x = F.elu(self.conv1d(x))
        
        # Flatten the output from conv1d
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ELU activation
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        
        # Output layer with linear activation
        x = self.out(x)
        
        return x



