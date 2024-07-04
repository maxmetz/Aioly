##### TRAIN MODEL WITH ADAM OPTIMIZER #########

###############################################################################
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from net.base_net import DeepSpectraCNN
from torch.utils.data import Dataset , DataLoader, random_split
from data.load_dataset import SoilSpectralDataSet
from torcheval import metrics
###############################################################################

# Load and split Data, define training parameters

###############################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_epochs = 1000
BATCH=256
LR=0.0001

data_path="C:\\00_aioly\\sources_projects\\OSSL_project\\data\\datasets\\ossl\\ossl_all_L1_v1.2.csv"
spectral_data = SoilSpectralDataSet(data_path=data_path,dataset_type="mir")
spec_dims=spectral_data.spec_dims


train_size = int(0.9 * len(spectral_data))
val_size = len(spectral_data) - train_size
train_dataset, val_dataset = random_split(spectral_data, [train_size, val_size])

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False,num_workers=0)
###############################################################################

# compute mean and std of dataset , Standardize the data

###############################################################################
mean = np.zeros(spec_dims) 
std = np.zeros(spec_dims)

for inputs, targets in train_loader:
    mean += np.sum(np.array(inputs),axis = 0)
mean /= len(train_loader.dataset)


for inputs, targets in train_loader:
    std += np.sum((np.array(inputs)-mean)**2,axis = 0)
    
std /= len(train_loader.dataset)
std = std 
###############################################################################

# Load Model and set training 

###############################################################################
model = DeepSpectraCNN(spec_dims, mean = mean,std = std)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.003/2)
criterion = nn.MSELoss(reduction='none')
criterion_test = nn.MSELoss()

print(model)
model = model.to(device)

###############################################################################

# Training loop

###############################################################################


for epoch in range(num_epochs):
    
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        
        inputs = inputs.to(device,non_blocking=True).float()
        targets = targets.to(device,non_blocking=True).float()
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs[:,None])  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    out = []
    tar = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            tar += targets
            inputs = inputs.to(device,non_blocking=True).float()
            targets = targets.to(device,non_blocking=True).float()
            outputs = model(inputs[:,None])
            out += outputs.detach().cpu()
            loss = criterion(outputs, targets) 
            val_loss += loss.item() * inputs.size(0)
    
    val_loss /= len(val_loader.dataset)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

plt.scatter(out,tar)
R2 = metrics.R2Score()
R2.update(torch.tensor(out),torch.tensor(tar))
R2.compute()