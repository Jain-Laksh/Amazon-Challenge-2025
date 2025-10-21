import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

# ========== Load Data ==========
df = pd.read_csv("../train_final.csv")
print(f"Train data loaded with shape: {df.shape}")

gemma_df = pd.read_csv("../gemma_embeddings_train.csv")
print(f"Gemma embeddings loaded with shape: {gemma_df.shape}")

# Rename embedding columns
embedding_cols = [c for c in gemma_df.columns if c != "sample_id"]
gemma_df.rename(columns={c: f"gemma_{c}" for c in embedding_cols}, inplace=True)

# Merge
df = df.merge(gemma_df, on="sample_id", how="inner")
print(f"Merged dataset shape: {df.shape}")

# Target
y = np.log1p(df["price"].values)
assert np.all(y >= 0), "Price must be non-negative"

# Separate column groups
manual_cols = [c for c in df.columns if not (c.startswith("stella_") or c.startswith("siglip_")
                                             or c.startswith("tfidf_") or c.startswith("gemma_")
                                             or c in ["sample_id", "price"])]
tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
manual_tfidf_cols = manual_cols + tfidf_cols

gemma_cols = [c for c in df.columns if c.startswith("gemma_")]
siglip_cols = [c for c in df.columns if c.startswith("siglip_")]

print(f"Feature dimensions:")
print(f"Manual+TFIDF: {len(manual_tfidf_cols)}")
print(f"Gemma: {len(gemma_cols)}")
print(f"SigLIP: {len(siglip_cols)}")

# Extract features
X_manual = df[manual_tfidf_cols].values
X_gemma = df[gemma_cols].values
X_siglip = df[siglip_cols].values

# Clip extreme values
clip_val = 1e5
X_manual = np.clip(X_manual, -clip_val, clip_val)
X_gemma = np.clip(X_gemma, -clip_val, clip_val)
X_siglip = np.clip(X_siglip, -clip_val, clip_val)

# Scale separately
scaler_manual = StandardScaler()
scaler_gemma = StandardScaler()
scaler_siglip = StandardScaler()
X_manual = scaler_manual.fit_transform(X_manual)
X_gemma = scaler_gemma.fit_transform(X_gemma)
X_siglip = scaler_siglip.fit_transform(X_siglip)

# Split
X_m_train, X_m_val, X_g_train, X_g_val, X_i_train, X_i_val, y_train, y_val = train_test_split(
    X_manual, X_gemma, X_siglip, y, test_size=0.1, random_state=42
)

print(f"Training samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")

# ========== Dataset ==========
class EmbeddingDataset(Dataset):
    def __init__(self, X_m, X_g, X_i, y):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_g = torch.tensor(X_g, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_m)

    def __getitem__(self, idx):
        return self.X_m[idx], self.X_g[idx], self.X_i[idx], self.y[idx]

train_ds = EmbeddingDataset(X_m_train, X_g_train, X_i_train, y_train)
val_ds = EmbeddingDataset(X_m_val, X_g_val, X_i_val, y_val)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ========== Best Hyperparameters ==========
best_params = {
    'batch_size': 512,
    'lr': 0.004627057196763367,
    'weight_decay': 0.0003070210905159036,
    'dropout': 0.27083982063503664,
    'hidden_manual': 256,
    'hidden_gemma': 384,
    'hidden_siglip': 448,
    'final_hidden': 448,
    'huber_delta': 4.283841288638561
}

print("Using best hyperparameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# ========== Model ==========
class DeepWideEmbeddingMLP(nn.Module):
    def __init__(self, manual_dim, gemma_dim, siglip_dim,
                 hidden_manual, hidden_gemma, hidden_siglip,
                 final_hidden, dropout):
        super().__init__()
        self.manual_net = nn.Sequential(
            nn.Linear(manual_dim, hidden_manual),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_manual),
            nn.Dropout(dropout)
        )
        self.gemma_net = nn.Sequential(
            nn.Linear(gemma_dim, hidden_gemma),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_gemma),
            nn.Dropout(dropout)
        )
        self.siglip_net = nn.Sequential(
            nn.Linear(siglip_dim, hidden_siglip),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_siglip),
            nn.Dropout(dropout)
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_manual + hidden_gemma + hidden_siglip, final_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(final_hidden),
            nn.Dropout(dropout),
            nn.Linear(final_hidden, 1)
        )

    def forward(self, x_m, x_g, x_i):
        m = self.manual_net(x_m)
        g = self.gemma_net(x_g)
        i = self.siglip_net(x_i)
        combined = torch.cat([m, g, i], dim=1)
        return self.final(combined)

# ========== SMAPE ==========
def smape(y_true, y_pred):
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-6)
    return 200 * torch.mean(torch.abs(y_true - y_pred) / denom)

# ========== Create Model and Training Setup ==========
batch_size = best_params["batch_size"]
lr = best_params["lr"]
weight_decay = best_params["weight_decay"]
dropout = best_params["dropout"]
hidden_manual = best_params["hidden_manual"]
hidden_gemma = best_params["hidden_gemma"]
hidden_siglip = best_params["hidden_siglip"]
final_hidden = best_params["final_hidden"]
huber_delta = best_params["huber_delta"]

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

model = DeepWideEmbeddingMLP(
    len(manual_tfidf_cols), len(gemma_cols), len(siglip_cols),
    hidden_manual, hidden_gemma, hidden_siglip, final_hidden, dropout
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

criterion = nn.HuberLoss(delta=huber_delta)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scaler_amp = GradScaler()

# ========== Training Loop ==========
best_val_smape = float('inf')
patience_counter = 0
max_epochs = 100  # Increased for better training
patience = 5     # More patience

print(f"\nStarting training for up to {max_epochs} epochs with patience {patience}...")
print("="*80)

for epoch in range(max_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    for xb_m, xb_g, xb_i, yb in train_loader:
        xb_m, xb_g, xb_i, yb = xb_m.to(device), xb_g.to(device), xb_i.to(device), yb.to(device)
        optimizer.zero_grad()
        
        with autocast():
            pred = model(xb_m, xb_g, xb_i)
            loss = criterion(pred, yb)
        
        if torch.isnan(loss):
            continue
            
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        
        train_loss += loss.item()
        num_batches += 1

    avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0

    # Validation phase
    model.eval()
    preds, truths = [], []
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for xb_m, xb_g, xb_i, yb in val_loader:
            xb_m, xb_g, xb_i, yb = xb_m.to(device), xb_g.to(device), xb_i.to(device), yb.to(device)
            
            with autocast():
                pred = model(xb_m, xb_g, xb_i)
                loss = criterion(pred, yb)
            
            preds.append(pred.cpu())
            truths.append(yb.cpu())
            val_loss += loss.item()
            val_batches += 1
    
    preds = torch.cat(preds, dim=0)
    truths = torch.cat(truths, dim=0)
    val_smape = smape(torch.expm1(truths), torch.expm1(preds))
    avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0

    print(f"Epoch {epoch+1:3d}/{max_epochs}: "
          f"Train Loss: {avg_train_loss:.6f}, "
          f"Val Loss: {avg_val_loss:.6f}, "
          f"Val SMAPE: {val_smape:.4f}")

    # Early stopping and model saving
    if val_smape < best_val_smape:
        best_val_smape = val_smape
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'optuna_best_siglip_gemma.pt')
        print(f"    *** New best model saved with SMAPE: {best_val_smape:.4f} ***")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1} (patience: {patience})")
            break

print("="*80)
print("Training completed!")

# Load best model
model.load_state_dict(torch.load('optuna_best_siglip_gemma.pt'))
print(f"Best validation SMAPE achieved: {best_val_smape:.4f}")

# Save complete model information
model_info = {
    'model_state_dict': model.state_dict(),
    'best_params': best_params,
    'scaler_manual': scaler_manual,
    'scaler_gemma': scaler_gemma,
    'scaler_siglip': scaler_siglip,
    'manual_tfidf_cols': manual_tfidf_cols,
    'gemma_cols': gemma_cols,
    'siglip_cols': siglip_cols,
    'best_val_smape': best_val_smape.item(),
    'feature_dims': {
        'manual': len(manual_tfidf_cols),
        'gemma': len(gemma_cols),
        'siglip': len(siglip_cols)
    }
}

torch.save(model_info, 'optuna_best_siglip_gemma.pt')
print("Complete model information saved to 'optuna_best_siglip_gemma.pt'")

# Final summary
print(f"\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Best validation SMAPE: {best_val_smape:.4f}")
print(f"Model architecture:")
print(f"  Manual+TFIDF → {hidden_manual}")
print(f"  Gemma → {hidden_gemma}")  
print(f"  SigLIP → {hidden_siglip}")
print(f"  Final → {final_hidden} → 1")
print(f"  Dropout: {dropout:.3f}")
print(f"Total parameters: {trainable_params:,}")
print("="*50)