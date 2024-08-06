import torch
import torcheval.metrics
import numpy as np
import os

class EarlyStopping:
    def __init__(self, patience=10, delta_factor=0.00025, save_path=None):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.last_loss = np.Inf
        
        self.delta_factor = delta_factor
        
        if save_path is None:
            rel_dir = os.path.dirname(os.path.abspath(__file__))
            self.save_path = os.path.join(rel_dir, 'best_model.pth')
        else:
            self.save_path=save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, val_loss, model, epoch):
        # Compute the adaptive delta
        delta = self.delta_factor * self.last_loss if self.last_loss < np.Inf else 0
        if val_loss < self.last_loss - delta:
            self.counter = 0        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_checkpoint(val_loss, model, epoch)
        
        self.last_loss=val_loss
      

    def save_checkpoint(self, val_loss, model, epoch):

        checkpoint_path = f'{self.save_path}_best_epoch_{epoch}.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        self.val_loss_min = val_loss
            
            
            
###############################################################################
def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, save_path=None, save_interval=10,early_stop=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    
 
    early_stopping = EarlyStopping(save_path=save_path) if early_stop and save_path else None
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    R2 = [torcheval.metrics.R2Score() for _ in range(model.out_dims)]
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = torch.zeros(model.out_dims, device=device)
        for inputs, targets in train_loader:
            
            inputs = inputs.to(device,non_blocking=True).float()
            targets = targets.to(device,non_blocking=True).float()
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs[:,None])  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            torch.mean(loss).backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.mean(dim=0) * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)       
           
        # Validation loop
        model.eval()
        val_loss = torch.zeros(model.out_dims, device=device)
        out = []
        tar = []
        with torch.no_grad():
            for inputs, targets in val_loader:
               
                inputs = inputs.to(device,non_blocking=True).float()
                targets = targets.to(device,non_blocking=True).float()
                outputs = model(inputs[:,None])
                loss = criterion(outputs, targets) 
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        if epoch == 0 : 
            ref_val = val_loss
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
      
        if save_path and (epoch + 1) % save_interval == 0 and val_loss < ref_val :
            ref_val = val_loss
            epoch_save_path = save_path + f'_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), epoch_save_path)
            print(f'Model saved at epoch {epoch + 1} to {epoch_save_path}')

    if save_path:
        if val_loss < ref_val : 
            final_save_path = save_path + '_best.pth'
            torch.save(model.state_dict(), final_save_path)
            print(f"Final model saved at {final_save_path}")
            
        else : 
            final_save_path = save_path + '_best.pth'
            torch.save(torch.load(epoch_save_path), final_save_path)
            print(f"Final model saved at {final_save_path}")
            
###############################################################################       