import pandas as pd
import re
import torch
from typing import Any, List, Tuple
from collections import Counter

MAX_VOCAB = 50000
MAX_LEN = 32

def _label_from_url(url: str) -> int:
    u = url.lower()
    if "foxnews.com" in u:
        return 1
    if "nbcnews.com" in u:
        return 0
    return -1

def _clean_slug(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"-rcna\d+.*$", "", slug)
    slug = re.sub(r"[^a-zA-Z\-]", " ", slug)
    return slug.replace("-", " ").lower().strip()

def _build_vocab(texts: List[str], max_vocab: int):
    c = Counter()
    for t in texts:
        for w in t.split():
            c[w] += 1
    itos = ["<pad>", "<unk>"] + [w for w, _ in c.most_common(max_vocab - 2)]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi

def _encode(text: str, stoi: dict, max_len: int):
    toks = text.split()
    ids = [stoi.get(w, stoi["<unk>"]) for w in toks[:max_len]]
    if len(ids) < max_len:
        ids += [stoi["<pad>"]] * (max_len - len(ids))
    mask = [0.0 if i == stoi["<pad>"] else 1.0 for i in ids]
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.float32),
    )

# ---- IMPORTANT ----
# evaluator calls prepare_data() on BOTH train/val csv and leaderboard csv
# so we must NOT depend on training split files here.
# simplest: build vocab from the same csv we are given.
def prepare_data(path: str) -> Tuple[List[Any], List[int]]:
    df = pd.read_csv(path)
    urls = df["url"].astype(str).tolist()

    texts = []
    labels = []
    for url in urls:
        y = _label_from_url(url)
        if y < 0:
            continue
        texts.append(_clean_slug(url))
        labels.append(y)

    stoi = _build_vocab(texts, MAX_VOCAB)

    X_list = []
    for t in texts:
        ids, mask = _encode(t, stoi, MAX_LEN)
        X_list.append({"ids": ids, "mask": mask})

    return X_list, labels