import os
from data.load_dataset_atonce import  SpectralDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from net.chemtools.PLS import PLS
from utils.testing import ccc,r2_score
import pickle

if __name__ == "__main__":
    
    #########################################################################################################
    user = os.environ.get('USERNAME')
    if user == 'fabdelghafo':
        data_path = "C:\\00_aioly\\GitHub\datasets\\ossl\\ossl_all_L1_v1.2.csv"
    else:
        data_path = "/home/metz/deepchemometrics/Aioly/data/dataset/oss/ossl_all_L1_v1.2.csv"

    y_labels = ["oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg","clay.tot_usda.a334_w.pct","k.ext_usda.a725_cmolc.kg","ph.h2o_usda.a268_index"]  
    dataset_type = "mir"
    
    
    base_path = os.path.join(os.path.dirname(data_path), 'figures', 'pls_multi', f'data_{dataset_type}')
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams['font.family'] = 'Times New Roman'

    # Instantiate the dataset
    dataset = SpectralDataset(data_path, y_labels, dataset_type)
    spec_dims=dataset.spec_dims

    #########################################################################################################
    
    wavelength = dataset.get_spectral_dimensions()
    num_samples = 200 
    X_train=dataset.X_train
    X_val=dataset.X_val
    Y_train=dataset.Y_train
    Y_val=dataset.Y_val
    #########################################################################################################
    ncomp=50
    pls =PLS(ncomp=ncomp)
    #########################################################################################################

    