import torch
import torch.nn as nn
import torch.nn.functional as F

class CuiNet(nn.Module):
    def __init__(self, input_dims, mean,std,out_dims=1):
        super(CuiNet, self).__init__()
        
        # Layers dimensions
        self.conv1d_dims = input_dims-4         # size of spectrum - (kernel_size-1) is the size of the spectrum after first conv1D
        self.k_number = 1
        self.k_width = 5
        self.k_stride = 1
        self.fc1_dims = 36
        self.fc2_dims = 18
        self.fc3_dims = 12
        self.out_dims = out_dims
        self.mean = nn.Parameter(torch.tensor(mean).float(),requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(),requires_grad=False)
        
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


    
    
class ConvBlock1D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBlock1D, self).__init__()

        # 1d convolution
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # batchnorm
        self.batchnorm1d = nn.BatchNorm1d(out_channels)

        # relu layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batchnorm1d(x)
        x = self.relu(x)
        return x
    
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1 = ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock1D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock1D(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out
   

class DeepSpectraCNN(nn.Module):
    def __init__(self, input_dim,mean,std,dropout=0.5,out_dims=1):
        super(DeepSpectraCNN, self).__init__()
        self.conv1d_dims = input_dim
        self.dropout=dropout
        self.mean = nn.Parameter(torch.tensor(mean).float(),requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(),requires_grad=False)
        self.out_dims=out_dims
        kernel_size = 7
        stride = 3
        padding = 3
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 8, kernel_size=kernel_size, stride=stride, padding=padding)
        # Inception modules
        self.inception2 = InceptionModule(8, 4)
        self.inception3 = InceptionModule(16,4)
        # Other layers
        self.flatten = nn.Flatten()
        flat_dim=((input_dim + 2 * padding - kernel_size) // stride + 1)
        self.fc1 = nn.Linear(16*flat_dim,64)
        self.fc2 = nn.Linear(64, self.out_dims)

    def forward(self, x):
        x = (x-self.mean)/self.std
        x = F.relu(self.conv1(x))
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.flatten(x)       
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x


class FullyConvNet(nn.Module):
    def __init__(self, input_dims, mean, std, out_dims=1):
        super(FullyConvNet, self).__init__()

        # Layers dimensions
        self.input_dims = input_dims
     
        self.out_dims = out_dims
        self.mean = nn.Parameter(torch.tensor(mean).float(), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(), requires_grad=False)

        # Convolutional layer
        self.conv1d_1 = nn.Conv1d(1, 2, kernel_size=9, stride=1)
        self.avg_1 = nn.AvgPool1d(2)
        self.conv1d_2 = nn.Conv1d(2, 2, kernel_size=7, stride=1)
        self.avg_2 = nn.AvgPool1d(2)
        self.conv1d_3 = nn.Conv1d(2, 4, kernel_size=7, stride=1)
        self.avg_3 = nn.AvgPool1d(2)
        self.conv1d_4 = nn.Conv1d(4, 8, kernel_size=5, stride=1)
        self.avg_4 = nn.AvgPool1d(2)
        self.conv1d_5 = nn.Conv1d(8, 12, kernel_size=3, stride=1)
        self.dp = nn.Dropout(0.8)
        self.head = nn.Conv1d(12, out_dims, kernel_size=1, stride=1)

    def forward(self, x):
        # Reshape input
        x = (x - self.mean) / self.std
        # Convolutional layer with ELU activation
        x = F.relu(self.conv1d_1(x))
        x = self.avg_1(x)
        x = F.relu(self.conv1d_2(x))
        x = self.avg_2(x)
        x = F.relu(self.conv1d_3(x))
        x = self.avg_3(x)
        x = F.relu(self.conv1d_4(x))
        x = self.avg_4(x)
        x = F.relu(self.conv1d_5(x))
        x = self.dp(x)
        x = self.head(x)
        x = F.adaptive_avg_pool1d(x,(1))
        return x[...,0]