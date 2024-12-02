
from data.load_dataset_atonce import  SpectralDataset
import pandas as pd
import os
if __name__ == "__main__":
    
    #########################################################################################################
    
    base_path= "C:\\00_aioly\\GitHub\\datasets\\ossl"
    data_path = os.path.join(base_path, "ossl_all_L1_v1.2.csv")
    
    dataset_type = "nir"
    all_y_labels = ["oc_usda.c729_w.pct", "na.ext_usda.a726_cmolc.kg", "clay.tot_usda.a334_w.pct", "k.ext_usda.a725_cmolc.kg", "ph.h2o_usda.a268_index"]  
    for y_labels in  all_y_labels:
        y_label_dir = y_labels[:5].replace(".", "_")
        dataset_dir = os.path.join(base_path, dataset_type,y_label_dir)
        os.makedirs(dataset_dir, exist_ok=True)

        
        x_train_csv_path = os.path.join(dataset_dir, "X_train.csv")
        x_test_csv_path = os.path.join(dataset_dir, "X_test.csv")
        y_train_csv_path = os.path.join(dataset_dir, "Y_train.csv")
        y_test_csv_path = os.path.join(dataset_dir, "Y_test.csv")

        
        dataset = SpectralDataset(data_path, [y_labels], dataset_type)
        spec_dims = dataset.spec_dims
        wavelength = dataset.get_spectral_dimensions()
        
        X_train=dataset.X_train
        X_test=dataset.X_val
        Y_train=dataset.Y_train
        Y_test=dataset.Y_val
                
        
        for target_label in [y_labels]:
            target_index = y_labels.index(target_label)
            Y_train_subset = Y_train[:, target_index].unsqueeze(1)
            Y_test_subset = Y_test[:, target_index].unsqueeze(1)
            
        X_train_np = X_train.numpy()  
        Y_train_np = Y_train.numpy()
        X_test_np = X_test.numpy()
        Y_test_np = Y_test.numpy()


        X_train_df = pd.DataFrame(X_train_np, columns=wavelength)
        X_test_df = pd.DataFrame(X_test_np, columns=wavelength)
        Y_train_df = pd.DataFrame(Y_train_np, columns=[y_labels])
        Y_test_df = pd.DataFrame(Y_test_np, columns=[y_labels])
        
        X_train_df.to_csv(x_train_csv_path, index=False)
        X_test_df.to_csv(x_test_csv_path, index=False)
        Y_train_df.to_csv(y_train_csv_path, index=False)
        Y_test_df.to_csv(y_test_csv_path, index=False)
       
        
   