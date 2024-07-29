################################### IMPORTS ###################################
import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset , DataLoader, random_split
from torcheval import metrics
from net.base_net import DeepSpectraCNN
from data.load_dataset import SoilSpectralDataSet
from utils.training import train
###############################################################################


if __name__ == "__main__":

################################### SEEDING ###################################
    # Set seed for reproducibility (for dataset splitting)
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
###############################################################################
    
################################# SET DEVICE ##################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###############################################################################
    
############################# DEFINE TRAINING PARAMS ##########################
    num_epochs = 1000
    BATCH = 1024
    LR = 0.0001
    save_interval = 50  # Save model every 10 epochs
###############################################################################

############################# LOAD DATA ##########################    
    name_model ="_DeepSpectra_OSSL_"  
    data_path="C:\\00_aioly\\sources_projects\\OSSL_project\\data\\datasets\\ossl\\ossl_all_L1_v1.2.csv"
    save_path = os.path.dirname(data_path) + f'\\models\\{name_model}\\'+ name_model
    
    y_labels = ["oc_usda.c729_w.pct", "na.ext_usda.a726_cmolc.kg"]
 
    # Load dataset and create DataLoader with seed
    spectral_data = SoilSpectralDataSet(data_path=data_path, dataset_type="mir", y_labels=y_labels)
    spec_dims=spectral_data.spec_dims
    
    train_size = int(0.9 * len(spectral_data))
    val_size = len(spectral_data) - train_size
    train_dataset, val_dataset = random_split(spectral_data, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=0)
###############################################################################

#####################    Standardize the data  ################################
    mean = np.zeros(spec_dims) 
    std = np.zeros(spec_dims)


    for inputs, targets in train_loader:
        mean += np.sum(np.array(inputs),axis = 0)                          
    mean /= len(train_loader.dataset)
  
    for inputs, targets in train_loader:
        
        std += np.sum((np.array(inputs)-mean)**2,axis = 0)        
    std /= len(train_loader.dataset)
###############################################################################

    model = DeepSpectraCNN(spec_dims, mean = mean,std = std,out_dims=len(y_labels))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.003/2)
    criterion = nn.MSELoss()
    criterion_test = nn.MSELoss()
    print(model)
   
    
    train(model, optimizer, criterion, train_loader, val_loader, num_epochs, save_path=save_path, save_interval=save_interval)