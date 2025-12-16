import pandas as pd
import re
import torch
from typing import Any, List, Tuple

def clean_slug(url: str) -> str:
    # rightmost
    slug = url.rstrip("/").split("/")[-1].lower()

    # endings
    slug = re.sub(r"(?:-)?rcna\d+$", "", slug)
    slug = re.sub(r"(?:-)?r?cna\d+$", "", slug)
    slug = re.sub(r"-n\d+$", "", slug)
    slug = re.sub(r"(?:-)?ncna\d+$", "", slug)
    slug = re.sub(r"(?:-)?ncpn\d+$", "", slug)
    slug = re.sub(r"(?:-)?\d{5,}$", "", slug)

    # meaningful words
    slug = re.sub(r"[^a-z0-9\-_]+", " ", slug)
    slug = slug.replace("_", "-")
    slug = re.sub(r"\s+", " ", slug).strip()

    # spaces
    slug = slug.replace("-", " ")
    slug = re.sub(r"\s+", " ", slug).strip()
    return slug

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

        text = clean_slug(url)
        X_list.append({"text": text})
        y_list.append(y)

    return X_list, torch.tensor(y_list, dtype=torch.long)