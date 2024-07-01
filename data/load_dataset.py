import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class SoilSpectralDataSet(Dataset):
    def __init__(self, dataset_type="visnir", data_path=None, preprocessing=None):
        if data_path==None:
            rel_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(rel_dir, 'data/datasets/oss/ossl_all_L1_v1.2.csv')
            
        self.data_path = data_path
        self.oss_data_path = data_path
        self.preprocessing =preprocessing
      
        
        
        try:
           data_raw = pd.read_csv(self.data_path, low_memory=False)
        except FileNotFoundError:
           raise ValueError(f"File not found: {self.data_path}")
        Y = np.array(data_raw.filter(regex = "oc.usda.c729"))
        mask = ~np.isnan(Y)[:,0]
        

        if dataset_type == "mir":
            mir = np.array(data_raw.filter(regex = "mir").filter(regex = "abs"))
            mask_ = ~(np.isnan(mir[:,0])[mask])            
            self.X = mir[mask][mask_]
            self.Y = Y[mask][mask_]

        if dataset_type == "nir":
            nir = np.array(data_raw.filter(regex = "visnir").filter(regex = "ref"))
            mask_ = ~(np.isnan(nir[:,0])[mask])            
            self.X = nir[mask][mask_]
            self.Y = Y[mask][mask_]


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        spectral_data = self.X[index, :-1]
        label = self.Y[index]
        return spectral_data, np.log(label+1)


# if __name__ == '__main__':


#     spectral_data = SoilSpectralDataSet("mir")
#     spectral_loader = DataLoader(spectral_data, shuffle=True, batch_size=32, num_workers=0)


#     for X, Y in spectral_loader:
#         print(X.shape)
#         print(Y.shape)
#         plt.plot(X.T)
#         break