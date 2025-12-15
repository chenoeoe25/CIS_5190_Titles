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

        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)
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

        self.cnns = nn.ModuleList([
            make_branch(3),
            make_branch(5),
            make_branch(7),
        ])

        feat_dim = 2 * c * len(self.cnns)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        
        # padding mask
        pad_mask = (ids == 0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)

        h_t = h.transpose(1, 2)
        feats = []
        for branch in self.cnns:
            z = branch(h_t)          # [B, C, L]
            z_for_max = z.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
            z_max = torch.amax(z_for_max, dim=2) # [B, C]
            z_for_mean = z.masked_fill(pad_mask.unsqueeze(1), 0.0)  # padding place set to 0
            denom = (~pad_mask).sum(dim=1).clamp_min(1).unsqueeze(1)  # [B,1]
            z_mean = z_for_mean.sum(dim=2) / denom  # [B, C]
            feats.append(torch.cat([z_max, z_mean], dim=1))  # [B, 2C]

        feat = torch.cat(feats, dim=1)
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
            logits = self.forward(ids)
            return logits.argmax(dim=1).cpu().tolist()

def get_model() -> Model:
    return Model()