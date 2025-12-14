# experiments/exp_byte_tf/model.py
import torch
import torch.nn as nn
from typing import Any, Iterable, List

MAX_LEN = 96
VOCAB_SIZE = 256  # byte 0..255

class Model(nn.Module):
    """
    纯本地 Transformer，不用 from_pretrained，不需要任何外部文件。
    evaluator:
      - 先 Model()
      - 再 load_state_dict(model.pt)
      - 再 predict(batch)
    """

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

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, L]
        if ids.dim() != 2:
            raise ValueError(f"ids must be [B,L], got {ids.shape}")

        B, L = ids.shape

        # --- hard safety ---
        # truncate or pad to MAX_LEN
        if L > MAX_LEN:
            ids = ids[:, :MAX_LEN]
            L = MAX_LEN
        elif L < MAX_LEN:
            pad = ids.new_zeros((B, MAX_LEN - L))
            ids = torch.cat([ids, pad], dim=1)
            L = MAX_LEN

        # clamp token ids to [0, 255]
        ids = ids.clamp(0, VOCAB_SIZE - 1)

        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(ids) + self.pos_emb(pos)

        h = self.encoder(x)
        feat = h.mean(dim=1)
        return self.classifier(feat)

    def predict(self, batch: Iterable[Any]) -> List[int]:
        items = list(batch)

        # 强制检查输入格式，防止 preprocess / eval 的误用被悄悄吞掉
        if len(items) == 0:
            return []
        if not isinstance(items[0], dict) or "ids" not in items[0]:
            raise TypeError(f"Expected each item to be dict with key 'ids', got: {type(items[0])} / {items[0]}")

        ids = torch.stack([it["ids"] for it in items], dim=0)  # [B, L]

        self.eval()
        with torch.no_grad():
            logits = self.forward(ids)
            return logits.argmax(dim=1).cpu().tolist()

def get_model() -> Model:
    return Model()