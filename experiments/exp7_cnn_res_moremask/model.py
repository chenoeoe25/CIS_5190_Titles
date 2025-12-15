import torch
import torch.nn as nn
from typing import Any, Iterable, List

MAX_LEN = 96
VOCAB_SIZE = 256


class ResBlock1D(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(c, c, 3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 256
        nhead = 8
        num_layers = 4
        dim_ff = 512
        dropout = 0.1

        # Token embedding (no padding_idx; padding token remains learnable)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)

        # Absolute positional embedding
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(0.1)

        c = 128

        def make_branch(k: int):
            pad = k // 2
            return nn.Sequential(
                nn.Conv1d(d_model, c, kernel_size=k, padding=pad),
                nn.GELU(),
                ResBlock1D(c),
                ResBlock1D(c),
            )

        # Multiple CNN branches with different kernel sizes
        self.cnns = nn.ModuleList([
            make_branch(3),
            make_branch(5),
            make_branch(7),
        ])

        feat_dim = c * len(self.cnns)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ids: Tensor of shape [B, L], byte-level token ids (0 = padding)
        """
        B, L = ids.shape

        # ----- Token + position embedding -----
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(ids) + self.pos_emb(pos)

        # Padding mask: True indicates padding positions
        pad_mask = (ids == 0)  # [B, L]

        # Mask enhancement 1:
        # Zero out all padded positions after adding positional embeddings
        # This prevents positional signals from leaking into padding tokens
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # ----- Transformer encoder -----
        # Padding tokens are excluded from attention
        h = self.encoder(x, src_key_padding_mask=pad_mask)  # [B, L, D]

        # ----- CNN + masked global max pooling -----
        h_t = h.transpose(1, 2)  # [B, D, L]
        feats = []

        for branch in self.cnns:
            z = branch(h_t)  # [B, C, L]

            # Mask enhancement 2:
            # Ensure padding positions are never selected by global max pooling
            z = z.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
            z = torch.amax(z, dim=2)  # [B, C]

            feats.append(z)

        feat = torch.cat(feats, dim=1)  # [B, C * num_branches]
        feat = self.dropout(feat)
        return self.classifier(feat)

    def predict(self, batch: Iterable[Any]) -> List[int]:
        items = list(batch)

        if len(items) > 0 and isinstance(items[0], str):
            def to_ids(s: str) -> torch.Tensor:
                b = s.encode("utf-8", errors="ignore")[:MAX_LEN]
                arr = list(b) + [0] * (MAX_LEN - len(b))
                return torch.tensor(arr, dtype=torch.long)

            ids = torch.stack([to_ids(s) for s in items], dim=0)
        else:
            ids = torch.stack([it["ids"] for it in items], dim=0)

        self.eval()
        with torch.no_grad():
            logits = self.forward(ids.to(next(self.parameters()).device))
            return logits.argmax(dim=1).cpu().tolist()


def get_model() -> Model:
    return Model()
