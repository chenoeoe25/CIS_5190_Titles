# experiments/exp_byte_tf/preprocess.py
import pandas as pd
import re
import torch
from typing import List, Tuple, Any

MAX_LEN = 96  # 64/96/128

def clean_slug(url: str) -> str:
    # only rightmost
    slug = url.rstrip("/").split("/")[-1].lower()

    # remove rcna, cna
    slug = re.sub(r"(?:-)?rcna\d+$", "", slug)
    slug = re.sub(r"(?:-)?r?cna\d+$", "", slug)
    slug = re.sub(r"-n\d+$", "", slug)
    slug = re.sub(r"(?:-)?ncna\d+$", "", slug)
    slug = re.sub(r"(?:-)?ncpn\d+$", "", slug)
    slug = re.sub(r"(?:-)?\d{5,}$", "", slug)

    # all to space
    slug = re.sub(r"[^a-z0-9\-_]+", " ", slug)
    slug = slug.replace("_", "-")
    slug = re.sub(r"\s+", " ", slug).strip()

    # - to space
    slug = slug.replace("-", " ")
    slug = re.sub(r"\s+", " ", slug).strip()
    return slug

def text_to_ids(t: str, max_len: int = MAX_LEN) -> torch.Tensor:
    # byte-level
    b = t.encode("utf-8", errors="ignore")[:max_len]
    arr = list(b)
    if len(arr) < max_len:
        arr += [0] * (max_len - len(arr))
    return torch.tensor(arr, dtype=torch.long)

def prepare_data(path: str) -> Tuple[List[Any], torch.Tensor]:
    df = pd.read_csv(path)

    X_list = []
    y_list = []

    for url in df["url"].astype(str).tolist():
        if "foxnews.com" in url:
            y = 1
        elif "nbcnews.com" in url:
            y = 0
        else:
            continue

        s = clean_slug(url)
        ids = text_to_ids(s, MAX_LEN)

        # eval_project_b.py interface
        X_list.append({"ids": ids})
        y_list.append(y)

    return X_list, torch.tensor(y_list, dtype=torch.long)