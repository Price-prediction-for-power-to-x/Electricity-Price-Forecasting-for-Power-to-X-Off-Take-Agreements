"""
1. **Channel attention gate** (squeeze-and-excitation): `Linear(20→5) → ReLU → Linear(5→20) → Sigmoid`, multiplied elementwise with the input — learns how much weight each of the 20 features deserves.
2. **Input projection:** `Linear(20→256) → LayerNorm → SiLU → Dropout(0.15)`.
3. **Self-attention:** `MultiheadAttention(256, 8 heads, dropout 0.15)` with residual connection + LayerNorm.
4. **4 residual blocks**, each: `Linear(256→256) → LayerNorm → SiLU → Dropout → Linear(256→256)`, then residual add + LayerNorm. The skip connections keep gradients healthy in a deep stack.
5. **Head:** `Linear(256→1)` → predicted price.

The design goal is *distributional* accuracy (low Wasserstein distance), not just low average error.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class WassNet(nn.Module):
    def __init__(self, input_dim, hidden=256, n_blocks=4, dropout=0.15, n_heads=4, use_channel_attn=False):
        super().__init__()
        self.use_channel_attn = use_channel_attn

        if use_channel_attn:
            self.channel_attn = nn.Sequential(
                nn.Linear(input_dim, max(1, input_dim // 4)),
                nn.ReLU(),
                nn.Linear(max(1, input_dim // 4), input_dim),
                nn.Sigmoid()
            )

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.self_attn = nn.MultiheadAttention(hidden, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden)
            )
            for _ in range(n_blocks)
        ])
        self.block_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_blocks)])

        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        if self.use_channel_attn:
            x = x * self.channel_attn(x)
        x = self.proj(x)
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.self_attn(x_seq, x_seq, x_seq)
        x = self.attn_norm(x + attn_out.squeeze(1))
        for block, norm in zip(self.blocks, self.block_norms):
            x = norm(x + block(x))
        return self.head(x)


# --- Quantile mapping ---
def learn_quantile_map(y_pred, y_actual):
    ps = np.sort(y_pred)
    ac = np.sort(y_actual)
    pct = np.linspace(0, 1, len(ps))
    pred_to_pct = interp1d(ps, pct, bounds_error=False, fill_value=(0, 1))
    pct_to_actual = interp1d(pct, ac, bounds_error=False, fill_value=(ac[0], ac[-1]))
    return pred_to_pct, pct_to_actual

def apply_quantile_map(y_pred, p2pct, pct2act):
    return pct2act(p2pct(y_pred))

print("WassNet: proj -> self-attn -> residual MLP blocks (Swish)")

numeric_features = ['Min Total Load', 'Max Total Load', 'Min Total Load_DE', 'Max Total Load_DE', 'hour_sin_1', 'hour_cos_1', 'hour_sin_2', 'hour_cos_2',
    'weekday_sin_1', 'weekday_cos_1', 'weekday_sin_2', 'weekday_cos_2','gas_price','total RE']
categorical_features = ['week', 'month', 'DE_holiday', 'Xmas_period',
                        'peak_status', 'businessday']
all_features = numeric_features + categorical_features

scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Use same train/val split as RF
X_train_raw = df.loc[train_mask, all_features].dropna()
y_train_raw = df.loc[X_train_raw.index, 'Day Ahead Auction (DK1)'].values.reshape(-1, 1)
X_val_raw = df.loc[val_mask, all_features].dropna()
y_val_raw = df.loc[X_val_raw.index, 'Day Ahead Auction (DK1)'].values.reshape(-1, 1)

X_train_s = scaler_X.fit_transform(X_train_raw)
X_val_s = scaler_X.transform(X_val_raw)
y_train_s = scaler_y.fit_transform(y_train_raw)
y_val_s = scaler_y.transform(y_val_raw)

X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_s, dtype=torch.float32).to(device)

# --- Pre-compute sample weights (always, used in training if toggle is on) ---
_w = 1.0 / (np.abs(y_train_raw.flatten() - 100.0) + 5)
w_train_np = _w / _w.mean()
w_train_t  = torch.tensor(w_train_np, dtype=torch.float32).to(device)

print(f"Train: {len(X_train_t)}, Val: {len(X_val_t)}, Features: {X_train_s.shape[1]}")

# derive input_dim from the already-scaled training array
input_dim = X_train_s.shape[1]

wn_hidden  = 256
wn_nblocks = 4
wn_nheads  = 8    # must divide wn_hidden evenly (e.g. 4, 8, 16, 32 for hidden=256)
wn_dropout = 0.15
wn_lr      = 1e-3
wn_wd      = 1e-4
wn_batch   = 512

assert wn_hidden % wn_nheads == 0, f"wn_hidden ({wn_hidden}) must be divisible by wn_nheads ({wn_nheads})"

USE_CHANNEL_ATTN  = True   # toggle SE channel attention on input features
USE_WEIGHTED_LOSS = True   # toggle threshold-weighted MAE (THRESHOLD=100)
THRESHOLD         = 100.0  # price level to focus errors around

# --- Stage 1: Train WassNet on pre-2024, early-stop on 2024, learn quantile mapping ---

# Rebuild loader here so it always matches the current USE_WEIGHTED_LOSS toggle
if USE_WEIGHTED_LOSS:
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t, w_train_t), batch_size=wn_batch, shuffle=True)
    print(f"Weighted loss ON  | threshold={THRESHOLD}")
else:
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=wn_batch, shuffle=True)
    print("Weighted loss OFF | plain MAE")

input_dim = X_train_s.shape[1]
wassnet = WassNet(input_dim, hidden=wn_hidden, n_blocks=wn_nblocks, dropout=wn_dropout, n_heads=wn_nheads, use_channel_attn=USE_CHANNEL_ATTN).to(device)
print(f"Channel attn: {USE_CHANNEL_ATTN}")
print(f"Parameters: {sum(p.numel() for p in wassnet.parameters()):,}")

optimizer = optim.AdamW(wassnet.parameters(), lr=wn_lr, weight_decay=wn_wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)
loss_fn = nn.L1Loss()  # used for val loss only
best_val, best_state, patience_ctr = float('inf'), None, 0
best_epoch = 0
train_losses, val_losses = [], []

for epoch in range(300):
    wassnet.train()
    eloss, n_b = 0, 0
    for batch in train_loader:
        xb, yb = batch[0], batch[1]
        optimizer.zero_grad()
        pred = wassnet(xb)
        if USE_WEIGHTED_LOSS:
            wb = batch[2]
            loss = (wb * torch.abs(pred - yb).squeeze()).mean()
        else:
            loss = torch.abs(pred - yb).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wassnet.parameters(), 1.0)
        optimizer.step()
        eloss += loss.item(); n_b += 1
    train_losses.append(eloss / n_b)
    
    wassnet.eval()
    with torch.no_grad():
        vl = loss_fn(wassnet(X_val_t), y_val_t).item()
    val_losses.append(vl)
    scheduler.step()
    
    if vl < best_val:
        best_val = vl
        best_state = {k: v.clone() for k, v in wassnet.state_dict().items()}
        patience_ctr = 0
        best_epoch = epoch + 1
    else:
        patience_ctr += 1
    
    if (epoch+1) % 25 == 0:
        print(f"Epoch {epoch+1:3d} | Train: {train_losses[-1]:.5f} | Val: {vl:.5f}")
    if patience_ctr >= 40:
        print(f"Early stop at epoch {epoch+1} (best was epoch {best_epoch})")
        break

wassnet.load_state_dict(best_state)
print(f"\nBest val MAE (scaled): {best_val:.5f} at epoch {best_epoch}")

# --- Learn quantile mapping on validation set ---
from scipy.stats import wasserstein_distance as w1_dist

wassnet.eval()
with torch.no_grad():
    val_pred_s = wassnet(X_val_t).cpu().numpy()
val_pred = scaler_y.inverse_transform(val_pred_s).flatten()
val_actual = y_val_raw.flatten()

pred_to_pct, pct_to_actual = learn_quantile_map(val_pred, val_actual)

val_recal = apply_quantile_map(val_pred, pred_to_pct, pct_to_actual)
print(f"\nVal W1 (raw NN):       {w1_dist(val_actual, val_pred):.3f}")
print(f"Val W1 (recalibrated): {w1_dist(val_actual, val_recal):.3f}")
print(f"Val MAE (raw NN):      {np.mean(np.abs(val_actual - val_pred)):.3f}")
print(f"\nQuantile mapping learned. Will retrain on full data next.")

# --- Stage 2: Retrain on FULL data (train+val), same as RF does ---
# This ensures a fair comparison: RF also retrains on full data before predicting 2025

X_full_raw = pd.concat([X_train_raw, X_val_raw])
y_full_raw = np.concatenate([y_train_raw, y_val_raw])

scaler_X_full = StandardScaler()
scaler_y_full = StandardScaler()
X_full_s = scaler_X_full.fit_transform(X_full_raw)
y_full_s = scaler_y_full.fit_transform(y_full_raw)

X_full_t = torch.tensor(X_full_s, dtype=torch.float32).to(device)
y_full_t = torch.tensor(y_full_s, dtype=torch.float32).to(device)

# Weights for full data (train+val) — always computed, used only if toggle is on
_wf = 1.0 / (np.abs(y_full_raw.flatten() - THRESHOLD) + 5)
w_full_t = torch.tensor(_wf / _wf.mean(), dtype=torch.float32).to(device)

if USE_WEIGHTED_LOSS:
    full_loader = DataLoader(TensorDataset(X_full_t, y_full_t, w_full_t), batch_size=wn_batch, shuffle=True)
else:
    full_loader = DataLoader(TensorDataset(X_full_t, y_full_t), batch_size=wn_batch, shuffle=True)

# New model, train for best_epoch epochs (from early stopping)
wassnet_final = WassNet(input_dim, hidden=wn_hidden, n_blocks=wn_nblocks, dropout=wn_dropout, n_heads=wn_nheads, use_channel_attn=USE_CHANNEL_ATTN).to(device)
opt_f = optim.AdamW(wassnet_final.parameters(), lr=wn_lr, weight_decay=wn_wd)
sch_f = optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=best_epoch, eta_min=1e-6)

n_retrain = best_epoch  # use same number of epochs that worked best
full_train_losses = []
print(f"Retraining on full data for {n_retrain} epochs (from early stopping)...")

for epoch in range(n_retrain):
    wassnet_final.train()
    eloss, n_b = 0, 0
    for batch in full_loader:
        xb, yb = batch[0], batch[1]
        opt_f.zero_grad()
        if USE_WEIGHTED_LOSS:
            wb = batch[2]
            loss = (wb * torch.abs(wassnet_final(xb) - yb).squeeze()).mean()
        else:
            loss = torch.abs(wassnet_final(xb) - yb).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wassnet_final.parameters(), 1.0)
        opt_f.step()
        eloss += loss.item(); n_b += 1
    full_train_losses.append(eloss / n_b)
    sch_f.step()
    if (epoch+1) % 25 == 0:
        print(f"  Epoch {epoch+1:3d}/{n_retrain} | Loss: {full_train_losses[-1]:.5f}")

print(f"Full retrain complete. Final loss: {full_train_losses[-1]:.5f}")

# --- Predict 2025 ---
X_test_2025 = df2025[all_features].ffill().bfill()
X_test_s = scaler_X_full.transform(X_test_2025)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

wassnet_final.eval()
with torch.no_grad():
    pred_2025_s = wassnet_final(X_test_t).cpu().numpy()
pred_2025_raw = scaler_y_full.inverse_transform(pred_2025_s).flatten()

# Apply quantile mapping (learned from validation stage)
pred_2025_recal = apply_quantile_map(pred_2025_raw, pred_to_pct, pct_to_actual)

df2025['wassnet_raw'] = pred_2025_raw
df2025['wassnet_prediction'] = pred_2025_recal

actual_2025 = df2025['Day Ahead Auction (DK1)'].values
print(f"\n--- 2025 Results ---")
print(f"WassNet raw   | mean: {pred_2025_raw.mean():.2f}, std: {pred_2025_raw.std():.2f}")
print(f"WassNet recal | mean: {pred_2025_recal.mean():.2f}, std: {pred_2025_recal.std():.2f}")
print(f"Actual 2025   | mean: {actual_2025.mean():.2f}, std: {actual_2025.std():.2f}")
