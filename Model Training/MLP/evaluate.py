import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

# ========== Parameters ==========
device = "cuda" if torch.cuda.is_available() else "cpu"

# Saved models
model_paths = ["optuna_best_48_46.pt", "optuna_best_48_22.pt"]
params_list = [
    {"hidden_manual": 1024, "hidden_stella": 320, "hidden_siglip": 320, "final_hidden": 256, "dropout": 0.296},
    {"hidden_manual": 896, "hidden_stella": 384, "hidden_siglip": 320, "final_hidden": 448, "dropout": 0.296}
]

# Learned ensemble weights
ensemble_weights = torch.tensor([0.5, 0.5], device=device)

# ========== Load test data ==========
df_test = pd.read_csv("../test_final.csv")
sample_ids = df_test["sample_id"].values

manual_cols = [c for c in df_test.columns if not (c.startswith("stella_") or c.startswith("siglip_") or c.startswith("tfidf_") or c=="sample_id")]
tfidf_cols = [c for c in df_test.columns if c.startswith("tfidf_")]
manual_tfidf_cols = manual_cols + tfidf_cols
stella_cols = [c for c in df_test.columns if c.startswith("stella_")]
siglip_cols = [c for c in df_test.columns if c.startswith("siglip_")]

X_manual = df_test[manual_tfidf_cols].values
X_stella = df_test[stella_cols].values
X_siglip = df_test[siglip_cols].values

# Clip extreme values
clip_val = 1e5
X_manual = np.clip(X_manual, -clip_val, clip_val)
X_stella = np.clip(X_stella, -clip_val, clip_val)
X_siglip = np.clip(X_siglip, -clip_val, clip_val)

# Load the same scalers used for training
scaler_manual = StandardScaler().fit(X_manual)  # replace with fitted scaler if saved
scaler_stella = StandardScaler().fit(X_stella)
scaler_siglip = StandardScaler().fit(X_siglip)

X_manual = scaler_manual.transform(X_manual)
X_stella = scaler_stella.transform(X_stella)
X_siglip = scaler_siglip.transform(X_siglip)

# ========== Dataset ==========
class TestDataset(Dataset):
    def __init__(self, X_m, X_s, X_i):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_s = torch.tensor(X_s, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
    def __len__(self):
        return len(self.X_m)
    def __getitem__(self, idx):
        return self.X_m[idx], self.X_s[idx], self.X_i[idx]

test_ds = TestDataset(X_manual, X_stella, X_siglip)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# ========== Model Definition ==========
class DeepWideEmbeddingMLP(torch.nn.Module):
    def __init__(self, manual_dim, stella_dim, siglip_dim, params):
        super().__init__()
        self.manual_net = torch.nn.Sequential(
            torch.nn.Linear(manual_dim, int(params["hidden_manual"])),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(int(params["hidden_manual"])),
            torch.nn.Dropout(params["dropout"])
        )
        self.stella_net = torch.nn.Sequential(
            torch.nn.Linear(stella_dim, int(params["hidden_stella"])),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(int(params["hidden_stella"])),
            torch.nn.Dropout(params["dropout"])
        )
        self.siglip_net = torch.nn.Sequential(
            torch.nn.Linear(siglip_dim, int(params["hidden_siglip"])),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(int(params["hidden_siglip"])),
            torch.nn.Dropout(params["dropout"])
        )
        self.final = torch.nn.Sequential(
            torch.nn.Linear(int(params["hidden_manual"])+int(params["hidden_stella"])+int(params["hidden_siglip"]), int(params["final_hidden"])),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(int(params["final_hidden"])),
            torch.nn.Dropout(params["dropout"]),
            torch.nn.Linear(int(params["final_hidden"]), 1)
        )
    def forward(self, x_m, x_s, x_i):
        m = self.manual_net(x_m)
        s = self.stella_net(x_s)
        i = self.siglip_net(x_i)
        combined = torch.cat([m,s,i], dim=1)
        return self.final(combined)

# ========== Load Models ==========
models = []
for path, params in zip(model_paths, params_list):
    model = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols), params).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    models.append(model)

# ========== Ensemble Inference ==========
predictions = []

with torch.no_grad():
    for xb_m, xb_s, xb_i in test_loader:
        xb_m, xb_s, xb_i = xb_m.to(device), xb_s.to(device), xb_i.to(device)
        preds_batch = [model(xb_m, xb_s, xb_i) for model in models]
        preds_batch = torch.stack(preds_batch, dim=0)  # [n_models, batch_size, 1]
        weighted_pred = torch.sum(ensemble_weights.view(-1,1,1) * preds_batch, dim=0)
        predictions.append(weighted_pred.cpu())

predictions = torch.cat(predictions, dim=0).numpy().flatten()
pred_prices = np.expm1(predictions)  # inverse log1p

# ========== Save to CSV ==========
submission = pd.DataFrame({"sample_id": sample_ids, "prediction": pred_prices})
submission.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")
