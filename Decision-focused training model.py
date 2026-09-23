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
