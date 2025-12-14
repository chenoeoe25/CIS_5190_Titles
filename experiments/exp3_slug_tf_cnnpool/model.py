# experiments/exp3_slug_tf_cnnpool/model.py
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

        # CNN pooling head: multi-kernel Conv1d + global maxpool
        c_out = 128
        self.cnns = nn.ModuleList([
            nn.Conv1d(d_model, c_out, kernel_size=3, padding=1),
            nn.Conv1d(d_model, c_out, kernel_size=5, padding=2),
            nn.Conv1d(d_model, c_out, kernel_size=7, padding=3),
        ])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)

        feat_dim = c_out * len(self.cnns)
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
        x = self.tok_emb(ids) + self.pos_emb(pos)           # [B, L, D]
        h = self.encoder(x)                                 # [B, L, D]

        # Conv1d expects [B, D, L]
        h_t = h.transpose(1, 2)                             # [B, D, L]
        feats = []
        for conv in self.cnns:
            z = self.act(conv(h_t))                         # [B, C, L]
            z = torch.amax(z, dim=2)                        # global max pool -> [B, C]
            feats.append(z)

        feat = torch.cat(feats, dim=1)                      # [B, C*k]
        feat = self.dropout(feat)
        return self.classifier(feat)                        # [B, 2]

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