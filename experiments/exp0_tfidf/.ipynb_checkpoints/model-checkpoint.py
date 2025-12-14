import math
import torch
import torch.nn as nn
from typing import Any, Dict, Iterable, List, Tuple

# -------- TF-IDF (pure torch, no sklearn) --------
class TfidfVectorizerTorch:
    """
    char n-gram TF-IDF, implemented with Python dicts + torch tensors.
    - fit() builds vocab + idf
    - transform_one(text) returns sparse indices + values
    For speed, we do sparse -> dense with torch.index_add_ during predict/train.
    """

    def __init__(self, ngram_min=2, ngram_max=2, max_features=50000):
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.max_features = max_features

        self.vocab: Dict[str, int] = {}
        self.idf: torch.Tensor | None = None

    def _char_ngrams(self, s: str) -> List[str]:
        s = s.strip()
        if not s:
            return []
        s = f" {s} "  # padding helps
        grams = []
        L = len(s)
        for n in range(self.ngram_min, self.ngram_max + 1):
            if L < n:
                continue
            grams.extend([s[i : i + n] for i in range(0, L - n + 1)])
        return grams

    def fit(self, texts: List[str]) -> None:
        # document frequency
        df: Dict[str, int] = {}
        for t in texts:
            uniq = set(self._char_ngrams(t))
            for g in uniq:
                df[g] = df.get(g, 0) + 1

        # top max_features by df
        items = sorted(df.items(), key=lambda kv: kv[1], reverse=True)[: self.max_features]
        self.vocab = {g: i for i, (g, _) in enumerate(items)}

        N = max(1, len(texts))
        idf = torch.zeros(len(self.vocab), dtype=torch.float32)
        for g, i in self.vocab.items():
            dfg = df.get(g, 1)
            # smooth idf
            idf[i] = math.log((1.0 + N) / (1.0 + dfg)) + 1.0
        self.idf = idf

    def transform_one(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.idf is not None, "Vectorizer not fitted"
        grams = self._char_ngrams(text)
        if not grams:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)

        tf: Dict[int, int] = {}
        for g in grams:
            idx = self.vocab.get(g, None)
            if idx is None:
                continue
            tf[idx] = tf.get(idx, 0) + 1

        if not tf:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)

        idxs = torch.tensor(list(tf.keys()), dtype=torch.long)
        vals = torch.tensor([tf[i] for i in idxs.tolist()], dtype=torch.float32)

        # tf-idf
        vals = vals * self.idf[idxs]

        # L2 normalize
        norm = torch.norm(vals, p=2)
        if norm > 0:
            vals = vals / norm

        return idxs, vals

    @property
    def dim(self) -> int:
        return len(self.vocab)


class Model(nn.Module):
    """
    Leaderboard interface:
      - instantiate Model() with no args
      - load_state_dict(model.pt)
      - predict(batch) where batch is list of {"text": str}
    """
    def __init__(self):
        super().__init__()
        # these will be filled after loading weights
        self.dim = 0
        self.linear = nn.Linear(1, 2)  # placeholder
        self._idf = None
        self._vocab_keys = None
        self._vocab_vals = None
        self._ngram_min = 2
        self._ngram_max = 2

    def _build_vectorizer_from_state(self) -> TfidfVectorizerTorch:
        v = TfidfVectorizerTorch(self._ngram_min, self._ngram_max, max_features=10**9)
        # reconstruct vocab dict from saved tensors
        keys: List[str] = self._vocab_keys  # type: ignore[assignment]
        vals: List[int] = self._vocab_vals  # type: ignore[assignment]
        v.vocab = {k: int(i) for k, i in zip(keys, vals)}
        v.idf = self._idf  # type: ignore[assignment]
        return v

    def forward_dense(self, X_dense: torch.Tensor) -> torch.Tensor:
        return self.linear(X_dense)

    def predict(self, batch: Iterable[Any]) -> List[int]:
        items = list(batch)
        # lazy init after load_state_dict
        vec = self._build_vectorizer_from_state()

        B = len(items)
        X = torch.zeros(B, vec.dim, dtype=torch.float32)
        for i, it in enumerate(items):
            text = it["text"] if isinstance(it, dict) else str(it)
            idxs, vals = vec.transform_one(text)
            if idxs.numel() > 0:
                X[i].index_add_(0, idxs, vals)

        self.eval()
        with torch.no_grad():
            logits = self.forward_dense(X)
            return logits.argmax(dim=1).cpu().tolist()

def get_model() -> Model:
    return Model()