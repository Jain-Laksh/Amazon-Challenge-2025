import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ================= Load and preprocess data =================
df = pd.read_csv("../train_final.csv")
print(f"Train data loaded with shape: {df.shape}")

# Load Gemma embeddings
gemma_df = pd.read_csv("../gemma_embeddings_train.csv")
print(f"Gemma embeddings loaded with shape: {gemma_df.shape}")

# Rename embedding columns
embedding_cols = [c for c in gemma_df.columns if c != "sample_id"]
gemma_df.rename(columns={c: f"gemma_{c}" for c in embedding_cols}, inplace=True)

# Load Qwen embeddings
qwen_df = pd.read_csv("../qwen_embeddings_train.csv")
print(f"Qwen embeddings loaded with shape: {qwen_df.shape}")

# Rename embedding columns
embedding_cols_qwen = [c for c in qwen_df.columns if c != "sample_id"]
qwen_df.rename(columns={c: f"qwen_{c}" for c in embedding_cols_qwen}, inplace=True)

# Merge all embeddings
df = df.merge(gemma_df, on="sample_id", how="inner")
df = df.merge(qwen_df, on="sample_id", how="inner")
print(f"Merged dataset shape: {df.shape}")

y = df["price"].values
y = np.log1p(y)  # log scale

# Separate column groups
manual_cols = [c for c in df.columns if not (c.startswith("stella_") or c.startswith("siglip_") 
                                             or c.startswith("tfidf_") or c.startswith("gemma_")
                                             or c.startswith("qwen_") or c in ["sample_id","price"])]
tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
manual_tfidf_cols = manual_cols + tfidf_cols
stella_cols = [c for c in df.columns if c.startswith("stella_")]
siglip_cols = [c for c in df.columns if c.startswith("siglip_")]
gemma_cols = [c for c in df.columns if c.startswith("gemma_")]
qwen_cols = [c for c in df.columns if c.startswith("qwen_")]

# Extract features
X_manual = df[manual_tfidf_cols].values
X_stella = df[stella_cols].values
X_siglip = df[siglip_cols].values
X_gemma = df[gemma_cols].values
X_qwen = df[qwen_cols].values

# Clip extreme values
clip_val = 1e5
X_manual = np.clip(X_manual, -clip_val, clip_val)
X_stella = np.clip(X_stella, -clip_val, clip_val)
X_siglip = np.clip(X_siglip, -clip_val, clip_val)
X_gemma = np.clip(X_gemma, -clip_val, clip_val)
X_qwen = np.clip(X_qwen, -clip_val, clip_val)

# ================= Data Splitting and Scaling =================
# Split the raw, unscaled data first (test_size=0.25 to match ensemble1/2)
X_m_train_raw, X_m_val_raw, \
X_s_train_raw, X_s_val_raw, \
X_i_train_raw, X_i_val_raw, \
X_g_train_raw, X_g_val_raw, \
X_q_train_raw, X_q_val_raw, \
y_train, y_val = train_test_split(
    X_manual, X_stella, X_siglip, X_gemma, X_qwen, y, test_size=0.25, random_state=42
)

# Fit scalers ONLY on the training data to prevent data leakage
scaler_manual = StandardScaler().fit(X_m_train_raw)
scaler_stella = StandardScaler().fit(X_s_train_raw)
scaler_siglip = StandardScaler().fit(X_i_train_raw)
scaler_gemma = StandardScaler().fit(X_g_train_raw)
scaler_qwen = StandardScaler().fit(X_q_train_raw)

# Transform both train and validation sets for MLPs
X_m_train = scaler_manual.transform(X_m_train_raw)
X_m_val = scaler_manual.transform(X_m_val_raw)

X_s_train = scaler_stella.transform(X_s_train_raw)
X_s_val = scaler_stella.transform(X_s_val_raw)

X_i_train = scaler_siglip.transform(X_i_train_raw)
X_i_val = scaler_siglip.transform(X_i_val_raw)

X_g_train = scaler_gemma.transform(X_g_train_raw)
X_g_val = scaler_gemma.transform(X_g_val_raw)

X_q_train = scaler_qwen.transform(X_q_train_raw)
X_q_val = scaler_qwen.transform(X_q_val_raw)

# ========== Dataset Classes ==========
class EmbeddingDataset(Dataset):
    def __init__(self, X_m, X_s, X_i, y):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_s = torch.tensor(X_s, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X_m)
    def __getitem__(self, idx):
        return self.X_m[idx], self.X_s[idx], self.X_i[idx], self.y[idx]

class EmbeddingDatasetGemma(Dataset):
    def __init__(self, X_m, X_g, X_i, y):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_g = torch.tensor(X_g, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X_m)
    def __getitem__(self, idx):
        return self.X_m[idx], self.X_g[idx], self.X_i[idx], self.y[idx]

class EmbeddingDatasetQwen(Dataset):
    def __init__(self, X_m, X_q, X_i, y):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_q = torch.tensor(X_q, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X_m)
    def __getitem__(self, idx):
        return self.X_m[idx], self.X_q[idx], self.X_i[idx], self.y[idx]

# Create datasets
val_ds = EmbeddingDataset(X_m_val, X_s_val, X_i_val, y_val)
val_ds_gemma = EmbeddingDatasetGemma(X_m_val, X_g_val, X_i_val, y_val)
val_ds_qwen = EmbeddingDatasetQwen(X_m_val, X_q_val, X_i_val, y_val)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
val_loader_gemma = DataLoader(val_ds_gemma, batch_size=256, shuffle=False)
val_loader_qwen = DataLoader(val_ds_qwen, batch_size=256, shuffle=False)

# ========== Define MLP Models ==========
class DeepWideEmbeddingMLP(nn.Module):
    def __init__(self, manual_dim, stella_dim, siglip_dim, params):
        super().__init__()
        self.manual_net = nn.Sequential(
            nn.Linear(manual_dim, int(params["hidden_manual"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(params["hidden_manual"])),
            nn.Dropout(params["dropout"])
        )
        self.stella_net = nn.Sequential(
            nn.Linear(stella_dim, int(params["hidden_stella"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(params["hidden_stella"])),
            nn.Dropout(params["dropout"])
        )
        self.siglip_net = nn.Sequential(
            nn.Linear(siglip_dim, int(params["hidden_siglip"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(params["hidden_siglip"])),
            nn.Dropout(params["dropout"])
        )
        self.final = nn.Sequential(
            nn.Linear(int(params["hidden_manual"]) + int(params["hidden_stella"]) + int(params["hidden_siglip"]), int(params["final_hidden"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(params["final_hidden"])),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(params["final_hidden"]), 1)
        )
    def forward(self, x_m, x_s, x_i):
        m = self.manual_net(x_m)
        s = self.stella_net(x_s)
        i = self.siglip_net(x_i)
        combined = torch.cat([m,s,i], dim=1)
        return self.final(combined)

class DeepWideEmbeddingMLPGemma(nn.Module):
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

class DeepWideEmbeddingMLPQwen(nn.Module):
    def __init__(self, manual_dim, qwen_dim, siglip_dim,
                 hidden_manual, hidden_qwen, hidden_siglip,
                 final_hidden, dropout):
        super().__init__()
        self.manual_net = nn.Sequential(
            nn.Linear(manual_dim, hidden_manual),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_manual),
            nn.Dropout(dropout)
        )
        self.qwen_net = nn.Sequential(
            nn.Linear(qwen_dim, hidden_qwen),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_qwen),
            nn.Dropout(dropout)
        )
        self.siglip_net = nn.Sequential(
            nn.Linear(siglip_dim, hidden_siglip),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_siglip),
            nn.Dropout(dropout)
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_manual + hidden_qwen + hidden_siglip, final_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(final_hidden),
            nn.Dropout(dropout),
            nn.Linear(final_hidden, 1)
        )

    def forward(self, x_m, x_q, x_i):
        m = self.manual_net(x_m)
        q = self.qwen_net(x_q)
        i = self.siglip_net(x_i)
        combined = torch.cat([m, q, i], dim=1)
        return self.final(combined)

# ========== Load MLP models ==========
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load stella+siglip models
params1 = {"hidden_manual":1024,"hidden_stella":320,"hidden_siglip":320,"final_hidden":256,"dropout":0.296}
model1 = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols), params1).to(device)
model1.load_state_dict(torch.load("../MLP/optuna_best_48_46_siglip_stella.pt"))
model1.eval()

params2 = {"hidden_manual":896,"hidden_stella":384,"hidden_siglip":320,"final_hidden":448,"dropout":0.296}
model2 = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols), params2).to(device)
model2.load_state_dict(torch.load("../MLP/optuna_best_48_22_siglip_stella.pt"))
model2.eval()

# Load gemma+siglip model
model3_params = {
    "hidden_manual": 256,
    "hidden_gemma": 384, 
    "hidden_siglip": 448,
    "final_hidden": 448,
    "dropout": 0.27083982063503664
}

model3 = DeepWideEmbeddingMLPGemma(
    len(manual_tfidf_cols), len(gemma_cols), len(siglip_cols),
    model3_params["hidden_manual"], model3_params["hidden_gemma"], 
    model3_params["hidden_siglip"], model3_params["final_hidden"], 
    model3_params["dropout"]
).to(device)
model3.load_state_dict(torch.load("../MLP/optuna_best_siglip_gemma.pth"))
model3.eval()

# Load qwen+siglip model
model4_params = {
    "hidden_manual": 256,
    "hidden_qwen": 128,
    "hidden_siglip": 192,
    "final_hidden": 384,
    "dropout": 0.26353456356106886
}

model4 = DeepWideEmbeddingMLPQwen(
    len(manual_tfidf_cols), len(qwen_cols), len(siglip_cols),
    model4_params["hidden_manual"], model4_params["hidden_qwen"], 
    model4_params["hidden_siglip"], model4_params["final_hidden"], 
    model4_params["dropout"]
).to(device)
model4.load_state_dict(torch.load("../MLP/optuna_best_siglip_qwen.pt"))
model4.eval()

mlp_models_stella = [model1, model2]  # Models using stella embeddings
mlp_model_gemma = model3  # Model using gemma embeddings
mlp_model_qwen = model4   # Model using qwen embeddings

# ========== Load XGBoost model ==========
xgb_model = xgb.Booster()
xgb_model.load_model("best_xgb_model_regression.json")

# ========== SMAPE ==========
def smape(y_true, y_pred):
    y_true = y_true.to(y_pred.device)
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-8)
    return 200 * torch.mean(torch.abs(y_true - y_pred)/denom)

# ========== Get predictions from all models ==========
print("Getting predictions from all models...")

# XGBoost predictions (using unscaled validation data)
X_val_xgb = np.hstack([X_m_val_raw, X_s_val_raw, X_i_val_raw]).astype(np.float32)
xgb_val_preds = torch.tensor(
    xgb_model.predict(xgb.DMatrix(X_val_xgb)),
    dtype=torch.float32,
    device=device
).unsqueeze(1)

# Stella+SigLIP MLP predictions
val_preds_stella = []
val_truths = []
with torch.no_grad():
    for xb_m, xb_s, xb_i, yb in val_loader:
        xb_m, xb_s, xb_i = xb_m.to(device), xb_s.to(device), xb_i.to(device)
        preds_batch = [m(xb_m, xb_s, xb_i) for m in mlp_models_stella]
        val_preds_stella.append(torch.stack(preds_batch, dim=0))
        val_truths.append(yb)

val_preds_stella = torch.cat(val_preds_stella, dim=1)  # Shape: [n_stella_models, total_samples, 1]
val_truths = torch.cat(val_truths, dim=0)  # Shape: [total_samples, 1]

# Gemma+SigLIP MLP predictions
val_preds_gemma = []
with torch.no_grad():
    for xb_m, xb_g, xb_i, yb in val_loader_gemma:
        xb_m, xb_g, xb_i = xb_m.to(device), xb_g.to(device), xb_i.to(device)
        pred_batch = mlp_model_gemma(xb_m, xb_g, xb_i)
        val_preds_gemma.append(pred_batch)

val_preds_gemma = torch.cat(val_preds_gemma, dim=0).unsqueeze(0)  # Shape: [1, total_samples, 1]

# Qwen+SigLIP MLP predictions
val_preds_qwen = []
with torch.no_grad():
    for xb_m, xb_q, xb_i, yb in val_loader_qwen:
        xb_m, xb_q, xb_i = xb_m.to(device), xb_q.to(device), xb_i.to(device)
        pred_batch = mlp_model_qwen(xb_m, xb_q, xb_i)
        val_preds_qwen.append(pred_batch)

val_preds_qwen = torch.cat(val_preds_qwen, dim=0).unsqueeze(0)  # Shape: [1, total_samples, 1]

# Combine all predictions
# Shape: [n_stella_models + 1 + 1 + 1, total_samples, 1] = [5, total_samples, 1]
all_preds = torch.cat([val_preds_stella, val_preds_gemma, val_preds_qwen, xgb_val_preds.unsqueeze(0)], dim=0)

print(f"All predictions shape: {all_preds.shape}")
print(f"Number of models: {all_preds.shape[0]}")

# ========== Weighted Ensemble Optimization ==========
n_models = all_preds.shape[0]
weights = torch.tensor([1.0/n_models]*n_models, requires_grad=True, device=device)
optimizer = torch.optim.Adam([weights], lr=0.01)

print("Optimizing ensemble weights...")
print("="*60)

# Optimize ensemble weights
for step in range(200):
    optimizer.zero_grad()
    
    # Ensure weights sum to 1 and are non-negative at each step
    with torch.no_grad():
        weights.clamp_(min=0)
        weights.div_(weights.sum())

    # Calculate weighted prediction
    weighted_pred = torch.sum(weights.view(-1, 1, 1) * all_preds, dim=0)
    
    # Calculate loss and backpropagate
    loss = smape(torch.expm1(val_truths), torch.expm1(weighted_pred))
    loss.backward()
    
    optimizer.step()
    
    if step % 20 == 0:
        print(f"Step {step:3d}, SMAPE: {loss.item():.4f}, Weights: {weights.detach().cpu().numpy()}")

# Final projection of weights after training loop
with torch.no_grad():
    weights.clamp_(min=0)
    weights.div_(weights.sum())

weighted_pred = torch.sum(weights.view(-1,1,1) * all_preds, dim=0)
ensemble_smape = smape(torch.expm1(val_truths), torch.expm1(weighted_pred))

print("="*60)
print("ENSEMBLE WEIGHT OPTIMIZATION COMPLETE")
print("="*60)

print(f"\nFinal Ensemble SMAPE: {ensemble_smape.item():.4f}")
print(f"Optimized Ensemble Weights:")
final_weights = weights.detach().cpu().numpy()
model_names = ["Stella+SigLIP MLP 1", "Stella+SigLIP MLP 2", "Gemma+SigLIP MLP", "Qwen+SigLIP MLP", "XGBoost"]
for i, (name, weight) in enumerate(zip(model_names, final_weights)):
    print(f"  {name}: {weight:.4f}")

print("="*60)

# Calculate individual model SMAPEs for comparison
print("\nIndividual Model Performance:")
for i in range(n_models):
    individual_smape = smape(torch.expm1(val_truths), torch.expm1(all_preds[i]))
    print(f"{model_names[i]}: SMAPE = {individual_smape.item():.4f}")

print("="*60)
print(f"Ensemble Performance:")
print(f"  Final SMAPE: {ensemble_smape.item():.4f}")
print(f"  Number of models: {n_models}")
print("="*60)
