import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ========== 1. Dataset skeleton ==========
class SatelliteSequenceDataset(Dataset):
    """
    Skeleton Dataset for temporal satellite prediction.
    Replace the data loading logic in __getitem__ with your own:
    - Load history_length images (as numpy arrays or torch.Tensors) of shape (C, H, W)
    - Load the next-step target image of shape (C, H, W)
    - Normalize/scale as needed
    """
    def __init__(self, history_length, in_channels, image_size, list_of_timestamps, patch_coords, band_names_or_paths=None):
        self.history_length = history_length
        self.in_channels = in_channels
        self.image_size = image_size
        self.timestamps = list_of_timestamps
        self.patch_coords = patch_coords
        self.band_info = band_names_or_paths

        self.samples = []
        num_times = len(self.timestamps)
        for idx in range(num_times - history_length):
            for (y0, x0) in self.patch_coords:
                self.samples.append((idx, y0, x0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx, y0, x0 = self.samples[idx]
        T = self.history_length
        C, H, W = self.in_channels, self.image_size[0], self.image_size[1]
        history = torch.zeros((T, C, H, W), dtype=torch.float32)
        target = torch.zeros((C, H, W), dtype=torch.float32)

        # TODO: Replace placeholder with actual data loading logic
        for t in range(T):
            arr = torch.randn(C, H, W)  # placeholder for history image
            history[t] = arr
        arr_tgt = torch.randn(C, H, W)  # placeholder for target image
        target = arr_tgt

        return history, target


# ========== 2. Model components ==========
class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, embed_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ph = pw = self.patch_size
        assert H % ph == 0 and W % pw == 0, "H,W must be divisible by patch_size"
        nh = H // ph
        nw = W // pw
        x = x.view(B, C, nh, ph, nw, pw)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous().view(B, nh * nw, C * ph * pw)
        x = self.proj(x)  # (B, num_patches, embed_dim)
        return x


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, grid_size=(16, 16)):
        super().__init__()
        nh, nw = grid_size
        self.pos_embed = nn.Parameter(torch.randn(1, nh * nw, embed_dim))

    def forward(self, x):
        return x + self.pos_embed


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=30):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(max_len, embed_dim))

    def forward(self, x):
        B, T, P, D = x.shape
        emb = self.pos_embed[:T]  # (T, D)
        emb = emb.view(1, T, 1, D)
        return x + emb


class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, embed_dim=128, image_size=(256, 256), history_length=3):
        super().__init__()
        self.patch_embed = PatchEmbed2D(in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dim)
        H, W = image_size
        nh = H // patch_size
        nw = W // patch_size
        self.spatial_pe = SpatialPositionalEncoding(embed_dim, grid_size=(nh, nw))
        self.temporal_pe = TemporalPositionalEncoding(embed_dim, max_len=history_length)
        self.history_length = history_length
        self.num_patches = nh * nw

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        patches = []
        for t in range(T):
            x_t = x_seq[:, t]  # (B, C, H, W)
            p = self.patch_embed(x_t)  # (B, num_patches, embed_dim)
            p = self.spatial_pe(p)
            patches.append(p)
        patches = torch.stack(patches, dim=1)  # (B, T, P, D)
        patches = self.temporal_pe(patches)    # (B, T, P, D)
        B, T, P, D = patches.shape
        out = patches.view(B, T * P, D)        # (B, T*P, D)
        return out


class TransformerEncoderLayerSimple(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x


class TransformerEncoderSimple(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerEncoderLayerSimple(d_model, num_heads, dim_ff, dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x


class NextStepDecoder(nn.Module):
    def __init__(self, embed_dim=128, patch_size=16, out_channels=1, image_size=(256, 256),
                 history_length=3, use_cnn_decoder=False):
        super().__init__()
        H, W = image_size
        nh = H // patch_size
        nw = W // patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.nh = nh
        self.nw = nw
        self.history_length = history_length
        self.use_cnn = use_cnn_decoder

        self.mlp_combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.to_patch = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

        if use_cnn_decoder:
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.conv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.ConvTranspose2d(embed_dim // 8, out_channels, kernel_size=4, stride=2, padding=1)
            self.act = nn.ReLU(inplace=True)

    def forward(self, encodings):
        B, S, D = encodings.shape
        P = self.nh * self.nw
        start = (self.history_length - 1) * P
        last_tokens = encodings[:, start:start + P, :]  # (B, P, D)
        global_ctx = encodings.mean(dim=1, keepdim=True)  # (B, 1, D)
        global_exp = global_ctx.expand(-1, P, -1)         # (B, P, D)
        combo = torch.cat([last_tokens, global_exp], dim=-1)  # (B, P, 2D)
        tokens_next = self.mlp_combine(combo)  # (B, P, D)

        if not self.use_cnn:
            patches = self.to_patch(tokens_next.view(B * P, D))
            patches = patches.view(B, P, -1, self.patch_size, self.patch_size)
            out = patches.new_zeros(B, patches.shape[2], self.nh * self.patch_size, self.nw * self.patch_size)
            for idx in range(P):
                i = idx // self.nw
                j = idx % self.nw
                out[:, :, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size] = patches[:, idx]
            return out
        else:
            feat = tokens_next.permute(0, 2, 1).contiguous().view(B, D, self.nh, self.nw)
            x = self.act(self.proj(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.act(self.conv3(x))
            x = self.conv4(x)
            return x


class TemporalPredictionTransformer(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, embed_dim=128,
                 image_size=(256,256), history_length=3,
                 encoder_layers=4, num_heads=8, dim_ff=256,
                 dropout=0.1, use_cnn_decoder=False):
        super().__init__()
        self.history_length = history_length
        self.st_embed = SpatioTemporalEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            image_size=image_size,
            history_length=history_length
        )
        self.encoder = TransformerEncoderSimple(
            num_layers=encoder_layers,
            d_model=embed_dim,
            num_heads=num_heads,
            dim_ff=dim_ff,
            dropout=dropout
        )
        self.decoder = NextStepDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            out_channels=in_channels,
            image_size=image_size,
            history_length=history_length,
            use_cnn_decoder=use_cnn_decoder
        )

    def forward(self, x_seq):
        tokens = self.st_embed(x_seq)  # (B, T*P, D)
        enc = self.encoder(tokens)     # (B, T*P, D)
        out = self.decoder(enc)        # (B, C, H, W)
        return out


def train_model():
    # Hyperparameters (example values)
    in_channels = 1
    patch_size = 16
    embed_dim = 128
    image_size = (128, 128)  # use appropriate size or patch size for your data
    history_length = 3
    encoder_layers = 2
    num_heads = 4
    dim_ff = 256
    dropout = 0.1
    use_cnn_decoder = False

    # TODO: Supply actual data identifiers and patch coordinates
    list_of_timestamps = []  # fill with your time identifiers
    patch_coords = []        # list of (y0, x0) for patches; or [(0,0)] for full-image
    train_dataset = SatelliteSequenceDataset(
        history_length=history_length,
        in_channels=in_channels,
        image_size=image_size,
        list_of_timestamps=list_of_timestamps,
        patch_coords=patch_coords,
        band_names_or_paths=None
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalPredictionTransformer(
        in_channels=in_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        image_size=image_size,
        history_length=history_length,
        encoder_layers=encoder_layers,
        num_heads=num_heads,
        dim_ff=dim_ff,
        dropout=dropout,
        use_cnn_decoder=use_cnn_decoder
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for history, target in train_loader:
            history = history.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(history)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * history.size(0)

        if len(train_loader.dataset) > 0:
            epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "temporal_transformer_model.pth")
    print("Training complete and model saved.")


if __name__ == "__main__":
    # For testing: this will run but dataset returns random data until you implement loading logic
    train_model()
