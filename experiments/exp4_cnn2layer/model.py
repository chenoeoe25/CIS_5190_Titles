import torch
import torch.nn as nn
from typing import Any, Iterable, List

MAX_LEN = 96
VOCAB_SIZE = 256  # byte 0..255

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 256
        nhead = 8
        num_layers = 4
        dim_ff = 512
        dropout = 0.1

        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
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

        c1 = 128
        c2 = 128

        def make_branch(k: int):
            pad = k // 2
            return nn.Sequential(
                nn.Conv1d(d_model, c1, kernel_size=k, padding=pad),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(c1, c2, kernel_size=3, padding=1),
                nn.GELU(),
            )

        self.cnns = nn.ModuleList([
            make_branch(3),
            make_branch(5),
            make_branch(7),
        ])

        feat_dim = c2 * len(self.cnns)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, L]
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(ids) + self.pos_emb(pos)     # [B, L, D]
        h = self.encoder(x)                           # [B, L, D]

        h_t = h.transpose(1, 2)                       # [B, D, L]
        feats = []
        for branch in self.cnns:
            z = branch(h_t)                           # [B, C2, L]
            z = torch.amax(z, dim=2)                  # [B, C2]
            feats.append(z)

        feat = torch.cat(feats, dim=1)                # [B, C2*3]
        feat = self.dropout(feat)
        return self.classifier(feat)                  # [B, 2]

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