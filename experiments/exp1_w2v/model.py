import torch
import torch.nn as nn
from typing import Any, Iterable, List

class Model(nn.Module):
    """
    Word2Vec-style trainable embedding + mean pooling + MLP
    """
    def __init__(self, vocab_size: int = 50000, emb_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch_dict):
        # batch_dict: {"ids": LongTensor[B, L], "mask": FloatTensor[B, L]}
        ids = batch_dict["ids"]
        mask = batch_dict["mask"]
        emb = self.tok_emb(ids)  # [B,L,D]
        emb = emb * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = emb.sum(dim=1) / denom  # mean pooling [B,D]
        return self.classifier(pooled)

    def predict(self, batch: Iterable[Any]) -> List[int]:
        items = list(batch)
        ids = torch.stack([it["ids"] for it in items], dim=0)
        mask = torch.stack([it["mask"] for it in items], dim=0)
        self.eval()
        with torch.no_grad():
            logits = self.forward({"ids": ids, "mask": mask})
            return logits.argmax(dim=1).cpu().tolist()

def get_model() -> Model:
    return Model()