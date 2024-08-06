import torch


###############################################################################
def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, save_path=None, save_interval=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    
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
      
        if save_path and (epoch + 1) % save_interval == 0:
            epoch_save_path = save_path + f'_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), epoch_save_path)
            print(f'Model saved at epoch {epoch + 1} to {epoch_save_path}')

    if save_path:
        final_save_path = save_path + '_final.pth'
        torch.save(model.state_dict(), final_save_path)
        print(f"Final model saved at {final_save_path}")
###############################################################################       