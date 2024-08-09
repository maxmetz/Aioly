import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

class SpectralDataset(Dataset):
    def __init__(self, data_path, y_labels="oc.usda.c729", dataset_type="visnir"):
        self.data_path = data_path
        self.y_labels = y_labels
        self.dataset_type = dataset_type
        self.data_raw = self.load_data()
        self.X, self.Y = self.process_data()
        
    def load_data(self):
        # Load data from the CSV file
        try:
            data_raw = pd.read_csv(self.data_path, low_memory=False)
            print(f"Data loaded successfully from {self.data_path}")
            return data_raw
        except FileNotFoundError:
            raise ValueError(f"File not found: {self.data_path}")
        
    def process_data(self):
        # Extract target variables and create mask for NaN values
        Y = np.array(self.data_raw.filter(regex="|".join(self.y_labels)))
        mask = ~np.isnan(Y).any(axis=1)

        # Choose spectral type based on dataset_type
        if self.dataset_type == "mir":
            spectrum_type = "mir"
            filter_keyword = "abs"
        elif self.dataset_type == "nir":
            spectrum_type = "visnir"
            filter_keyword = "ref"
        else:
            raise ValueError("dataset_type must be either 'mir' or 'nir'")

        # Extract spectrum data
        spectral_data = np.array(self.data_raw.filter(regex=spectrum_type).filter(regex=filter_keyword))
        
        # Apply mask
        X = spectral_data[mask]
        Y = Y[mask]

        return X, Y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def get_train_val_data(data_path, y_labels, dataset_type, test_size=0.2, random_seed=42):
       
        dataset = SpectralDataset(data_path, y_labels, dataset_type)
        
        num_samples = len(dataset)
        num_val_samples = int(test_size * num_samples)
        num_train_samples = num_samples - num_val_samples

        
        train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples],
                                                    generator=torch.Generator().manual_seed(random_seed))

        X_train = torch.stack([train_dataset[i][0] for i in range(num_train_samples)])
        Y_train = torch.stack([train_dataset[i][1] for i in range(num_train_samples)])
        X_val = torch.stack([val_dataset[i][0] for i in range(num_val_samples)])
        Y_val = torch.stack([val_dataset[i][1] for i in range(num_val_samples)])

        return X_train, Y_train, X_val, Y_val


