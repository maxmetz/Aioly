import torch
import torcheval.metrics
import numpy as np
import os

class EarlyStopping:
    def __init__(self, patience=10, delta_factor=0.00025, save_path='checkpoint'):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.last_loss = np.Inf
        self.save_path = save_path
        self.delta_factor = delta_factor
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, val_loss, model, epoch):
        # Compute the adaptive delta
        delta = self.delta_factor * self.last_loss if self.last_loss < np.Inf else 0
        print(delta)
        if val_loss < self.last_loss - delta:
            self.counter = 0
            
        else:
            self.counter += 1
            print(self.counter)
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_checkpoint(val_loss, model, epoch)
        
        self.last_loss=val_loss
      

    def save_checkpoint(self, val_loss, model, epoch):

        checkpoint_path = f'{self.save_path}_best_epoch_{epoch}.pt'
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
    
    R2 = torcheval.metrics.R2Score()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            
            inputs = inputs.to(device,non_blocking=True).float()
            targets = targets.to(device,non_blocking=True).float()
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs[:,None])  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            torch.mean(loss).backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

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
        val_losses.append(val_loss)
        
        
        all_outputs = torch.cat(out, dim=0)
        all_targets = torch.cat(tar, dim=0)
        R2.update(all_outputs, all_targets)
        r2_score = R2.compute()
        val_r2_scores.append(r2_score.item()) 
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        
         # Check early stopping
        if early_stop:
            early_stopping(val_loss, model, epoch + 1)
            if early_stopping.early_stop:
                break
               
      
        if save_path and (epoch + 1) % save_interval == 0:
            epoch_save_path = save_path + f'_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), epoch_save_path)
            print(f'Model saved at epoch {epoch + 1} to {epoch_save_path}')

    if save_path:
        final_save_path = save_path + '_final.pth'
        torch.save(model.state_dict(), final_save_path)
        print(f"Final model saved at {final_save_path}")
        
    return train_losses, val_losses,val_r2_scores
###############################################################################       