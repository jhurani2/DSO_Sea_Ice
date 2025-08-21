import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers.drop import DropPath
from timm.layers.helpers import to_2tuple
from timm.layers.weight_init import trunc_normal_
from typing import cast


def window_partition(x, window_size):
    """Partition into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        x = self.relative_position_index
        x = torch.tensor(self.relative_position_index)
        relative_position_bias = self.relative_position_bias_table[x.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, dropout=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x, H // self.patch_size, W // self.patch_size


class SeaIceSwinTransformer(nn.Module):
    """
    Swin Transformer for Sea Ice Concentration Prediction
    - Multi-variable input support (siconc, wind, temperature, etc.)
    - Hierarchical feature learning
    - Self-attention for spatial dependencies
    - Regression head for continuous prediction
    """
    
    def __init__(self, in_ch=5, out_ch=1, img_size=224, patch_size=4, embed_dim=96, 
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size

        # Make sure window size divides patch grid
        patches_per_dim = img_size // patch_size
        if patches_per_dim % window_size != 0:
            # Adjust window size to fit
            window_size = self._find_best_window_size(patches_per_dim)
            print(f"Adjusted window_size to {window_size} for img_size {img_size}")

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_ch, embed_dim=embed_dim)
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.patches_resolution = patches_resolution

        # Build layers
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=int(embed_dim * 2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])][i]
                ) for i in range(depths[i_layer])
            ])
            
            # Patch merging layer
            if i_layer < self.num_layers - 1:
                downsample = nn.Sequential(
                    nn.LayerNorm(int(embed_dim * 2 ** i_layer)),
                    nn.Linear(int(embed_dim * 2 ** i_layer), int(embed_dim * 2 ** (i_layer + 1)), bias=False)
                )
                self.layers.append(nn.ModuleDict({'blocks': layer, 'downsample': (downsample)}))
            else:
                downsample = None

        # Regression head for sea ice concentration
        final_dim = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = nn.LayerNorm(final_dim)
        
        # Upsampling path to reconstruct spatial resolution
        self.upsample_layers = nn.ModuleList()
        for i in range(self.num_layers - 1, 0, -1):
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(int(embed_dim * 2 ** i), int(embed_dim * 2 ** (i-1)), 
                                     kernel_size=2, stride=2),
                    nn.LayerNorm(int(embed_dim * 2 ** (i-1))),
                    nn.GELU()
                )
            )
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim//2, out_ch, kernel_size=2, stride=2),
            nn.Sigmoid()  # For sea ice concentration [0,1]
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _find_best_window_size(self, patches_per_dim):
    # Find the best window size that divides the patch grid
    # Try window sizes from 7 down to 1
        for ws in [7, 5, 3, 1]:
            if patches_per_dim % ws == 0:
                return ws
        return 1  # Fallback

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Resize input to expected size if needed
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Patch embedding
        x, Hp, Wp = self.patch_embed(x)
        
        # Swin Transformer layers
        features = []
        for layer in self.layers:
            md = cast(nn.ModuleDict, layer)
            blocks = cast(nn.ModuleList, md['blocks'])
            for blk in blocks:
                x = blk(x, Hp, Wp)
            features.append(x)
            down = md['downsample']
            if not isinstance(down, nn.Identity):
                x = down(x)
                Hp, Wp = Hp // 2, Wp // 2
        
        # Final norm
        x = self.norm(x)
        
        # Reshape to spatial format
        x = x.transpose(1, 2).view(B, -1, Hp, Wp)
        
        # Upsample to original resolution
        for upsample in self.upsample_layers:
            x = upsample(x)

        # Final prediction
        x = self.head(x)
        
        # Resize back to original dimensions if needed
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x


# Drop-in replacement factory function
def create_swin_model(in_ch: int, out_ch: int = 1, img_size: int = 224, features: int = 96):
    """
    Factory function for Swin Transformer sea ice model.
    Drop-in replacement for create_robust_model.
    
    Args:
        in_ch: Input channels (number of variables)
        out_ch: Output channels
        img_size: Input image size (assumes square)
        features: Base embedding dimension
    """
    return SeaIceSwinTransformer(
        in_ch=in_ch, 
        out_ch=out_ch, 
        img_size=img_size,
        embed_dim=features,
        depths=[2, 2, 4, 2],  # Lighter than original
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.1
    )


# Enhanced Trainer with attention visualization
class SwinTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _combined_loss(self, pred, target, alpha=0.7):
        """Robust loss combining MSE and L1"""
        l1_loss = F.l1_loss(pred, target)
        mse_loss = F.mse_loss(pred, target)
        return alpha * mse_loss + (1 - alpha) * l1_loss
    
    def fit(self, train_loader, test_loader=None, epochs=50, lr=2e-5):
        """Enhanced training with better defaults for Transformer"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=0.05,
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr*0.01
        )

        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 25
        
        history = {'train_loss': [], 'test_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for xb, yb, _ in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(xb)
                
                # Handle NaN values
                mask = ~torch.isnan(yb)
                if mask.sum() > 0:
                    loss = self._combined_loss(pred[mask], yb[mask])
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
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
                        mask = ~torch.isnan(yb)
                        if mask.sum() > 0:
                            test_loss += self._combined_loss(pred[mask], yb[mask]).item()
                
                test_loss /= len(test_loader)
                history['test_loss'].append(test_loss)
                
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
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}, lr={scheduler.get_last_lr()[0]:.2e}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, lr={scheduler.get_last_lr()[0]:.2e}")
            
            scheduler.step()
        
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


# Updated test function using Swin Transformer
def run_swin_test(ds_path: str, variables: list, batch_size: int = 4, epochs: int = 50, img_size: int = 224):
    """
    Test function for Swin Transformer sea ice prediction.
    
    Args:
        ds_path: Path to combined dataset
        variables: List of variable names ['siconc', 'w_speed', 'temp', ...]
        batch_size: Batch size (smaller for transformer)
        epochs: Training epochs
        img_size: Input image size
    """
    from models.data import get_dataloaders
    from models.utils import mae, rmse
    import matplotlib.pyplot as plt
    
    print('=== Swin Transformer Sea Ice Test ===')
    print(f'Variables: {variables}')
    
    train_loader, test_loader = get_dataloaders(
        ds_path, variables, 'siconc',
        lead=0, batch_size=batch_size,
        test_fraction=0.2, num_workers=0,
        time_split=True, normalize=True,
        fill_na=0.0, add_mask=True
    )
    

    # Get input channels and determine image size
    sample_batch = next(iter(train_loader))
    xb, yb, t = sample_batch
    in_ch = xb.shape[1]
    
    # Get actual image dimensions
    actual_h, actual_w = xb.shape[2], xb.shape[3]
    print(f'Actual image size: {actual_h}x{actual_w}')
    
    # Auto-determine appropriate image size if not provided
    if img_size is None:
        # Find largest dimension divisible by 32 (for 4 downsample stages)
        max_dim = max(actual_h, actual_w)
        img_size = ((max_dim + 31) // 32) * 32  # Round up to nearest multiple of 32
        print(f'Auto-selected img_size: {img_size}')
    
    # Handle NaN values
    for loader in [train_loader, test_loader]:
        for xb_batch, yb_batch, _ in loader:
            xb_batch[torch.isnan(xb_batch)] = 0
            yb_batch[torch.isnan(yb_batch)] = 0
    
    print(f'Input channels: {in_ch}, Batch shape: {xb.shape}')
    
    # Create Swin Transformer model
    model = create_swin_model(in_ch=in_ch, out_ch=1, img_size=img_size, features=96)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')
    
    # Training
    trainer = SwinTrainer(model)
    history = trainer.fit(train_loader, test_loader, epochs=epochs, lr=2e-5)
    
    # Evaluation
    test_loss = trainer.evaluate(test_loader)
    preds, trues, times = trainer.predict_batch(test_loader, max_batches=3)
    
    print(f'Test loss: {test_loss:.6f}')
    print(f'MAE: {mae(preds, trues):.6f}')
    print(f'RMSE: {rmse(preds, trues):.6f}')
    
    # Enhanced visualization
    if len(preds) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i in range(min(2, len(preds))):
            # True vs Predicted vs Difference
            im1 = axes[i, 0].imshow(trues[i], cmap='Blues_r', origin='lower', vmin=0, vmax=1)
            axes[i, 0].set_title(f'True (Sample {i+1})')
            plt.colorbar(im1, ax=axes[i, 0])
            
            im2 = axes[i, 1].imshow(preds[i], cmap='Blues_r', origin='lower', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Predicted (Sample {i+1})')
            plt.colorbar(im2, ax=axes[i, 1])
            
            diff = preds[i] - trues[i]
            im3 = axes[i, 2].imshow(diff, cmap='RdBu_r', origin='lower', 
                                  vmin=-0.5, vmax=0.5)
            axes[i, 2].set_title(f'Difference (Sample {i+1})')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.tight_layout()
        plt.show()
        
        # Loss curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        if history['test_loss']:
            plt.plot(history['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        # Scatter plot of predictions vs truth
        plt.subplot(1, 2, 2)
        true_flat = trues.flatten()
        pred_flat = preds.flatten()
        plt.scatter(true_flat, pred_flat, alpha=0.5, s=1)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        plt.xlabel('True Sea Ice Concentration')
        plt.ylabel('Predicted Sea Ice Concentration')
        plt.title('Predictions vs Truth')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return model, history


if __name__ == "__main__":
    # Example usage with multiple variables
    ds_path = "combined_data.nc"
    variables = ['siconc', 'w_speed', 'temp', 'wind_u', 'wind_v']
    
    model, history = run_swin_test(
        ds_path=ds_path,
        variables=variables, 
        batch_size=4,  # Smaller batch size for transformer
        epochs=100,
        img_size=224
    )