# experiments/exp_byte_tf/preprocess2.py
import pandas as pd
import re
import torch
from typing import List, Tuple, Any
from urllib.parse import urlparse

MAX_LEN = 160

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\.\-_/=]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def url_to_text(url: str) -> str:
    u = urlparse(url)

    host = u.netloc.lower()
    path = u.path.strip("/")

    parts = [p for p in path.split("/") if p]
    p1 = parts[0] if len(parts) > 0 else ""
    p2 = parts[1] if len(parts) > 1 else ""
    slug = parts[-1] if parts else ""

    # 去掉 NBC 常见噪声
    slug = re.sub(r"(?:-)?r?cna\d+.*$", "", slug)
    slug = slug.replace("-", " ").replace("_", " ")

    # 形状特征（直接变成文本）
    digits = sum(c.isdigit() for c in slug)
    letters = sum(c.isalpha() for c in slug)
    total = max(len(slug), 1)

    text = (
        f"host={host} "
        f"p1={p1} "
        f"p2={p2} "
        f"slug={slug} "
        f"dr={digits/total:.2f} "
        f"lr={letters/total:.2f}"
    )
    return normalize(text)

def text_to_ids(t: str, max_len: int) -> torch.Tensor:
    b = t.encode("utf-8", errors="ignore")[:max_len]
    arr = list(b)
    if len(arr) < max_len:
        arr += [0] * (max_len - len(arr))
    return torch.tensor(arr, dtype=torch.long)

def prepare_data(path: str) -> Tuple[List[Any], torch.Tensor]:
    df = pd.read_csv(path)

    X_list, y_list = [], []

    for url in df["url"].astype(str).tolist():
        if "foxnews.com" in url:
            y = 1
        elif "nbcnews.com" in url:
            y = 0
        else:
            continue

        t = url_to_text(url)
        ids = text_to_ids(t, MAX_LEN)

        X_list.append({"ids": ids})
        y_list.append(y)

    return X_list, torch.tensor(y_list, dtype=torch.long)