import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from preprocess import prepare_data
from model import Model

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

SEED = 0
MAX_VOCAB = 50000
MAX_LEN = 32
BATCH = 256
EPOCHS = 40
LR = 2e-4

def split_indices(n: int, seed: int = 0, train=0.8, val=0.1):
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_train = int(n * train)
    n_val = int(n * val)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return train_idx, val_idx, test_idx

def build_vocab(texts: List[str], max_vocab: int):
    from collections import Counter
    c = Counter()
    for t in texts:
        for w in t.split():
            c[w] += 1
    itos = ["<pad>", "<unk>"] + [w for w,_ in c.most_common(max_vocab-2)]
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi

def encode(text: str, stoi: dict, max_len: int):
    toks = text.split()
    ids = []
    for w in toks[:max_len]:
        ids.append(stoi.get(w, stoi["<unk>"]))
    if len(ids) < max_len:
        ids += [stoi["<pad>"]] * (max_len - len(ids))
    mask = [0.0 if i == stoi["<pad>"] else 1.0 for i in ids]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

class TxtDS(Dataset):
    def __init__(self, texts, labels, indices, stoi):
        self.texts = texts
        self.labels = labels
        self.indices = indices
        self.stoi = stoi

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        ids, mask = encode(self.texts[j], self.stoi, MAX_LEN)
        y = torch.tensor(self.labels[j], dtype=torch.long)
        return {"ids": ids, "mask": mask}, y

def acc_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def evaluate(model, loader, device):
    model.eval()
    tot = 0
    corr = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = {k:v.to(device) for k,v in xb.items()}
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            corr += (pred == yb).sum().item()
            tot += yb.numel()
    return corr / max(1, tot)

def main():
    texts, labels = prepare_data(str((ROOT / "../../data/train_urls_url_only.csv").resolve()))
    n = len(labels)
    tr, va, te = split_indices(n, seed=SEED)
    stoi = build_vocab([texts[i] for i in tr], MAX_VOCAB)

    json.dump({"seed": SEED, "train": tr, "val": va, "test": te}, open(ART/"split.json","w"))

    ds_tr = TxtDS(texts, labels, tr, stoi)
    ds_va = TxtDS(texts, labels, va, stoi)
    ds_te = TxtDS(texts, labels, te, stoi)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(vocab_size=len(stoi), emb_dim=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_va = -1
    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in dl_tr:
            xb = {k:v.to(device) for k,v in xb.items()}
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        va_acc = evaluate(model, dl_va, device)
        print(f"Epoch {ep:03d}: val_acc={va_acc:.4f}")
        if va_acc > best_va:
            best_va = va_acc
            torch.save(model.state_dict(), ART/"model.pt")

    # load best
    model.load_state_dict(torch.load(ART/"model.pt", map_location="cpu"))
    model.to(device)

    train_acc = evaluate(model, dl_tr, device)
    val_acc = evaluate(model, dl_va, device)
    test_acc = evaluate(model, dl_te, device)

    # external teacher csv
    X_t, y_t = prepare_data(str((ROOT / "../../data/url_only_data.csv").resolve()))
    idx_t = list(range(len(y_t)))
    ds_t = TxtDS(X_t, y_t, idx_t, stoi)
    dl_t = DataLoader(ds_t, batch_size=BATCH, shuffle=False, num_workers=0)
    teacher_acc = evaluate(model, dl_t, device)

    print("\nACC summary:")
    print("  train_acc  =", train_acc)
    print("  val_acc    =", val_acc)
    print("  test_acc   =", test_acc)
    print("  teacher_acc=", teacher_acc)
    print("\nSaved:", ART/"model.pt")

if __name__ == "__main__":
    main()