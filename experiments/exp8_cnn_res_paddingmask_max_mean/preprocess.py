# experiments/exp_byte_tf/preprocess.py
import pandas as pd
import re
import torch
from typing import List, Tuple, Any

MAX_LEN = 96  # 你可以试 64/96/128

def clean_slug(url: str) -> str:
    # 只取最后一个 / 右侧
    slug = url.rstrip("/").split("/")[-1].lower()

    # 去掉像 rcna223374 这种尾巴（NBC 常见）
    slug = re.sub(r"(?:-)?rcna\d+$", "", slug)
    slug = re.sub(r"(?:-)?r?cna\d+$", "", slug)

    # 保留字母数字和分隔符 - _ （其余变空格）
    slug = re.sub(r"[^a-z0-9\-_]+", " ", slug)
    slug = slug.replace("_", "-")
    slug = re.sub(r"\s+", " ", slug).strip()

    # 把 - 当成空格（让 token 更像词）
    slug = slug.replace("-", " ")
    slug = re.sub(r"\s+", " ", slug).strip()
    return slug

def text_to_ids(t: str, max_len: int = MAX_LEN) -> torch.Tensor:
    # byte-level：把字符串转成 UTF-8 bytes，再截断/补齐
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

        # eval_project_b.py 会把 X 当 list 逐个 item 传进 predict(batch)
        X_list.append({"ids": ids})
        y_list.append(y)

    return X_list, torch.tensor(y_list, dtype=torch.long)