import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr

# ---- Config ---- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WEEKS = 10
STEPS_PER_WEEK = 168
PATCH_SIZE = 16
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
IMAGE_SIZE = 64  # assume 64x64 crops for simplicity

# ---- Helper: Patch Embedding ---- #
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# ---- Temporal Positional Encoding ---- #
class TemporalPositionEncoding(nn.Module):
    def __init__(self, num_steps, embed_dim):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, num_steps, 1, embed_dim))

    def forward(self, x):
        return x + self.pos_emb  # x: [B, T, N, D]

# ---- Vision Transformer for Week Prediction ---- #
class WeekViT(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, nhead, num_layers):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=1, embed_dim=embed_dim)
        self.temporal_pe = TemporalPositionEncoding(num_steps=7, embed_dim=embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(embed_dim, patch_size * patch_size)  # predict 1 patch
        self.num_patches = self.patch_embed.num_patches

    def forward(self, x):  # x: [B, 7, 1, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        patches = self.patch_embed(x)  # [B*T, N, D]
        patches = patches.reshape(B, T, self.num_patches, -1)  # [B, T, N, D]
        patches = self.temporal_pe(patches)
        tokens = patches.reshape(B, T * self.num_patches, -1).permute(1, 0, 2)  # [seq_len, B, D]
        encoded = self.encoder(tokens)  # [seq_len, B, D]
        last_frame = encoded[-self.num_patches:]  # [N, B, D]
        decoded = self.decoder(last_frame)  # [N, B, patch_area]
        decoded = decoded.permute(1, 0, 2)  # [B, N, patch_area]
        H_p = W_p = int(np.sqrt(self.num_patches))
        out = decoded.reshape(B, 1, H_p, W_p, PATCH_SIZE, PATCH_SIZE)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(B, 1, IMAGE_SIZE, IMAGE_SIZE)
        return out

# ---- Data Loader ---- #
def week_data():
    dataset = xr.open_zarr('/notebook_dir/public/mickellals-public/goes-16-2003-10-weeks.tmp.zarr')
    ds = dataset.assign_coords(lon=((dataset.lon + 180) % 360 - 180))
    cmic13 = ds["CMI_C13"].sel(
        lat=slice(25, 45),   
        lon=slice(-86, -63)  
    )
    weekly_means = []
    for i in range(10):
        week = cmic13.isel(t=slice(i * 168, (i + 1) * 168))
        week_mean = week.mean(dim='t')
        weekly_means.append(week_mean)
    return weekly_means
    
def prepare_input(weeks):
    input_weeks = weeks[:7]   # use Weeks 0–6 to predict Week 7
    target = weeks[7]         # Week 7 = Week 8 in 1-based count

    input_tensor = torch.stack([
        torch.tensor(w.values).float().unsqueeze(0) for w in input_weeks
    ])  # shape: [7, 1, H, W]

    return (
        input_tensor.unsqueeze(0).to(DEVICE),    # [1, 7, 1, H, W]
        torch.tensor(target.values).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, H, W]
    )

# ---- Main ---- #
def main():
    model = WeekViT(IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    weeks = week_data()
    x, y = prepare_input(weeks)

    model.train()
    for epoch in range(10):
        pred = model(x)  # [B, 1, H, W]
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    main()

def get_week8_prediction():
    weeks = week_data()
    x, _ = prepare_input(weeks)

    model = WeekViT(IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)
    model.eval()

    with torch.no_grad():
        pred = model(x)  # shape: [1, 1, H, W]

    return pred.squeeze().cpu().numpy()  # shape: [H, W]
