import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
from torch.utils.data import ConcatDataset, DataLoader
class ConvBlock(nn.Module):
    """Robust convolution block with BatchNorm and optional dropout"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class RobustUNet(nn.Module):
    """
    Minimal yet robust U-Net with:
    - Proper skip connections
    - Batch normalization for stability
    - Optional dropout for regularization
    - Residual connections in bottleneck
    - Proper weight initialization
    """
    def __init__(self, in_ch: int, out_ch: int = 1, features: int = 64, dropout: float = 0.1):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_ch, features, dropout)
        self.enc2 = ConvBlock(features, features*2, dropout)
        self.enc3 = ConvBlock(features*2, features*4, dropout)
        self.enc4 = ConvBlock(features*4, features*8, dropout)
        
        # Bottleneck with residual connection
        self.bottleneck = nn.Sequential(
            ConvBlock(features*8, features*16, dropout),
            ConvBlock(features*16, features*16, dropout)  # Extra depth
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(features*16, features*8, 2, stride=2)
        self.dec4 = ConvBlock(features*16, features*8, dropout)  # Skip connection doubles channels
        
        self.up3 = nn.ConvTranspose2d(features*8, features*4, 2, stride=2)
        self.dec3 = ConvBlock(features*8, features*4, dropout)
        
        self.up2 = nn.ConvTranspose2d(features*4, features*2, 2, stride=2)
        self.dec2 = ConvBlock(features*4, features*2, dropout)
        
        self.up1 = nn.ConvTranspose2d(features*2, features, 2, stride=2)
        self.dec1 = ConvBlock(features*2, features, dropout)
        
        # Final output with 1x1 conv
        self.final = nn.Conv2d(features, out_ch, 1)
        
        self.pool = nn.MaxPool2d(2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization for training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder path with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)


# Drop-in replacement for your existing SimpleCNN
def create_robust_model(in_ch: int, out_ch: int = 1, features: int = 32):
    """
    Factory function that creates a robust model.
    Drop-in replacement for SimpleCNN with same interface.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels  
        features: Base feature count (will be scaled appropriately)
    """
    return RobustUNet(in_ch=in_ch, out_ch=out_ch, features=features, dropout=0.1)


# Enhanced Trainer with better loss function and learning rate scheduling
class EnhancedTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _combined_loss(self, pred, target, alpha=0.7):
        """Robust loss combining MSE and L1"""
        l1_loss = F.l1_loss(pred, target)
        mse_loss = F.mse_loss(pred, target)
        return alpha * mse_loss + (1 - alpha) * l1_loss
    
    def fit(self, train_loader, test_loader=None, epochs=50, lr=1e-4):
        """Enhanced training with better defaults and scheduling"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8
        )

        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20
        
        history = {'train_loss': [], 'test_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for xb, yb, _ in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(xb)
                mask = ~torch.isnan(yb)
                if mask.sum() > 0:
                    loss = self._combined_loss(pred[mask], yb[mask])
                else:
                    loss = torch.tensor(0.0, device=self.device)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            test_loss = 0.0
            if test_loader:
                self.model.eval()
                with torch.no_grad():
                    for xb, yb, _ in test_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        pred = self.model(xb)
                        # Only compute loss on non-NaN values
                        mask = ~torch.isnan(yb)
                        if mask.sum() > 0:
                            test_loss += self._combined_loss(pred[mask], yb[mask]).item()
                
                test_loss /= len(test_loader)
                history['test_loss'].append(test_loss)
                scheduler.step(test_loss)
                
                # Early stopping
                if test_loss < best_loss:
                    best_loss = test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}")
        
        return history
    
    def evaluate(self, loader):
        """Evaluate model on dataset"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for xb, yb, _ in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                mask = ~torch.isnan(yb)
                if mask.sum() > 0:
                    total_loss += self._combined_loss(pred[mask], yb[mask]).item()
        
        return total_loss / len(loader)
    
    def predict_batch(self, loader, max_batches=None):
        """Generate predictions for visualization"""
        self.model.eval()
        preds, trues, times = [], [], []
        
        with torch.no_grad():
            for i, (xb, yb, t) in enumerate(loader):
                if max_batches and i >= max_batches:
                    break
                
                xb = xb.to(self.device)
                pred = self.model(xb)
                
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())
                times.append(t)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # Squeeze channel dimension if it's 1
        if preds.shape[1] == 1:
            preds = preds[:, 0]
        if trues.shape[1] == 1:
            trues = trues[:, 0]
            
        return preds, trues, times


# Updated test function using robust U-Net
#Handles multiple variables, but sea ice concentration by default
def run_robust_test(ds, variables: list = ['siconc'], target_var: str = 'siconc',
                    batch_size: int = 8, epochs: int = 50):
    """
    Multi-variable U-Net test function.
    
    Args:
        ds_path: Path to dataset (can be combined .nc file)
        variables: List of input variables ['siconc', 'w_speed', 'temp', ...]
        target_var: Target variable to predict
        batch_size, epochs: Training parameters
    """
    from models.data import get_dataloaders
    from models.utils import mae, rmse
    import matplotlib.pyplot as plt
    
    print('=== Multi-Variable U-Net Test ===')
    print(f'Input variables: {variables}')
    print(f'Target variable: {target_var}')



    #I want to load this for multiple files within the folder so that xb, yb is combined

    nc_files = glob.glob(f"{ds}/*.nc")
    train_datasets, test_datasets = [], []

    for file in nc_files:
        print(file)
        train_loader, test_loader = get_dataloaders(file, variables, target_var,
                                                    lead=0, batch_size=batch_size,
                                                    test_fraction=0.2, num_workers=0,
                                                    time_split=True, normalize=True,
                                                    fill_na=0.0, add_mask=True)
        train_datasets.append(train_loader.dataset)
        test_datasets.append(test_loader.dataset)

    # Combine all datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_test_dataset = ConcatDataset(test_datasets)

    # Create DataLoaders
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)
    print("I'm here and I combined all datasets")
    # Get input channels
    sample_batch = next(iter(train_loader))
    xb, yb, t = sample_batch


    # Check if training and test data still have NaN values and substitute with 0
    for loader in [train_loader, test_loader]:
        for xb, yb, _ in loader:
            xb[torch.isnan(xb)] = 0
            yb[torch.isnan(yb)] = 0
        print(xb, yb)

    in_ch = xb.shape[1]
    print(f'Input channels: {in_ch}, Batch shape: {xb.shape}')
    
    # Create robust model (drop-in replacement)
    model = create_robust_model(in_ch=in_ch, out_ch=1, features=32)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')
    
    # Enhanced training
    trainer = EnhancedTrainer(model)
    history = trainer.fit(train_loader, test_loader, epochs=epochs, lr=1e-4)
    
    # Evaluation
    test_loss = trainer.evaluate(test_loader)
    preds, trues, times = trainer.predict_batch(test_loader, max_batches=3)
    
    print(f'Test loss: {test_loss:.6f}')
    print(f'MAE: {mae(preds, trues):.6f}')
    print(f'RMSE: {rmse(preds, trues):.6f}')
    
    # Quick visualization
    if len(preds) > 0:
        idx = 0
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = axes[0].imshow(trues[idx], cmap='Blues_r', origin='lower', vmin=0, vmax=1)
        axes[0].set_title('True')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(preds[idx], cmap='Blues_r', origin='lower', vmin=0, vmax=1)
        axes[1].set_title('Predicted')
        plt.colorbar(im2, ax=axes[1])
        
        diff = preds[idx] - trues[idx]
        im3 = axes[2].imshow(diff, cmap='RdBu_r', origin='lower')
        axes[2].set_title('Difference')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()

        # Calculate baseline persistence error (use last available data as prediction)
    persistence_mse = np.mean((trues[1:] - trues[:-1])**2)  # Simple persistence baseline

    # Plot MSE vs epochs with baseline
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training MSE', linewidth=2)
    plt.plot(history['test_loss'], label='Test MSE', linewidth=2) 
    plt.axhline(float(persistence_mse), color='red', linestyle='--', label=f'Persistence Baseline ({persistence_mse:.4f})', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Model Performance vs Persistence Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

        # Arctic region plotting (vertical stacking)
    if len(preds) > 0:
        arctic_slice = slice(-80, None)  # Top 80 pixels for Arctic
        idx = 0
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # True Arctic
        im1 = axes[0].imshow(trues[idx][arctic_slice, :], cmap='Blues_r', origin='lower', vmin=0, vmax=1)
        axes[0].set_title('True (Arctic)')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Predicted Arctic
        im2 = axes[1].imshow(preds[idx][arctic_slice, :], cmap='Blues_r', origin='lower', vmin=0, vmax=1)
        axes[1].set_title('Predicted (Arctic)')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Difference Arctic
        diff = preds[idx][arctic_slice, :] - trues[idx][arctic_slice, :]
        im3 = axes[2].imshow(diff, cmap='RdBu_r', origin='lower', vmin=-0.5, vmax=0.5)
        axes[2].set_title('Difference (Arctic)')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        plt.show()
    
    return model, history


if __name__ == "__main__":
    # Example usage - just replace your existing function call
    ds_path = "your_dataset_path.nc"
    model, history = run_robust_test(ds_path, batch_size=8, epochs=50)