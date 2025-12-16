import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .preprocess import prepare_data
from .model import Model, TfidfVectorizerTorch

def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ListDataset(Dataset):
    def __init__(self, X_list, y_tensor):
        self.X = X_list
        self.y = y_tensor
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], int(self.y[i].item())

def split_indices(n, val_ratio=0.1, test_ratio=0.1, seed=0):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = idx[:n_test]
    val = idx[n_test:n_test+n_val]
    train = idx[n_test+n_val:]
    return train, val, test

def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_all, y_all = prepare_data("data/train_urls_url_only.csv")
    n = len(y_all)

    tr_idx, va_idx, te_idx = split_indices(n, val_ratio=0.1, test_ratio=0.1, seed=0)

    def subset(idxs):
        Xs = [X_all[i] for i in idxs]
        ys = y_all[idxs]
        return Xs, ys

    Xtr, ytr = subset(tr_idx)
    Xva, yva = subset(va_idx)
    Xte, yte = subset(te_idx)

    # fit TF-IDF on training texts only
    texts_tr = [x["text"] for x in Xtr]
    vec = TfidfVectorizerTorch(ngram_min=2, ngram_max=2, max_features=50000)
    vec.fit(texts_tr)

    def to_dense_batch(X_items):
        B = len(X_items)
        X = torch.zeros(B, vec.dim, dtype=torch.float32)
        for i, it in enumerate(X_items):
            idxs, vals = vec.transform_one(it["text"])
            if idxs.numel() > 0:
                X[i].index_add_(0, idxs, vals)
        return X

    class DenseDataset(Dataset):
        def __init__(self, X_items, y_tensor):
            self.X_items = X_items
            self.y = y_tensor
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            return self.X_items[i], int(self.y[i].item())

    def collate_fn(batch):
        X_items, ys = zip(*batch)
        X = to_dense_batch(list(X_items))
        y = torch.tensor(ys, dtype=torch.long)
        return X, y

    tr_loader = DataLoader(DenseDataset(Xtr, ytr), batch_size=256, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(DenseDataset(Xva, yva), batch_size=512, shuffle=False, collate_fn=collate_fn)
    te_loader = DataLoader(DenseDataset(Xte, yte), batch_size=512, shuffle=False, collate_fn=collate_fn)

    model = Model().to(device)
    # replace placeholder linear with correct dim
    model.linear = nn.Linear(vec.dim, 2).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    def eval_acc(loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in loader:
                X = X.to(device)
                y = y.to(device)
                logits = model.forward_dense(X)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return correct / max(total, 1)

    best_val = -1.0
    os.makedirs("experiments/exp0_tfidf/artifacts", exist_ok=True)
    best_path = "experiments/exp0_tfidf/artifacts/model_best.pt"

    EPOCHS = 30
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in tr_loader:
            X = X.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model.forward_dense(X)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)

        val_acc = eval_acc(va_loader)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)

        if ep % 5 == 0 or ep == 1:
            avg_loss = total_loss / max(len(tr_loader.dataset), 1)
            print(f"Epoch {ep:03d}: loss={avg_loss:.4f} val_acc={val_acc:.4f} best={best_val:.4f}")

    # load best and pack vectorizer params into state_dict
    model.load_state_dict(torch.load(best_path, map_location=device))

    # pack tfidf vocab + idf into model for saving
    model._ngram_min = vec.ngram_min
    model._ngram_max = vec.ngram_max
    model._idf = vec.idf.cpu()
    # store vocab as 2 parallel lists so torch can save it
    # (python dict itself can't be in state_dict)
    keys = list(vec.vocab.keys())
    vals = [vec.vocab[k] for k in keys]
    model._vocab_keys = keys
    model._vocab_vals = vals

    # torch only saves tensors in state_dict; so we add them as buffers
    # recreate as buffers so they appear in state_dict
    model.register_buffer("idf_buf", model._idf)
    model.register_buffer("vocab_vals_buf", torch.tensor(model._vocab_vals, dtype=torch.long))
    # keys are strings -> save separately in a small sidecar file
    # (still allowed locally; for leaderboard you will include that file if needed,
    #  but ideally keep it tiny)
    sidecar = "experiments/exp0_tfidf/artifacts/vocab_keys.txt"
    with open(sidecar, "w", encoding="utf-8") as f:
        for k in model._vocab_keys:
            f.write(k.replace("\n", " ") + "\n")

    out_path = "experiments/exp0_tfidf/artifacts/model.pt"
    torch.save(model.state_dict(), out_path)

    test_acc = eval_acc(te_loader)

    # teacher acc
    X_teacher, y_teacher = prepare_data("data/url_only_data.csv")
    teacher_loader = DataLoader(DenseDataset(X_teacher, y_teacher), batch_size=512, shuffle=False, collate_fn=collate_fn)
    teacher_acc = eval_acc(teacher_loader)

    print("Saved:", out_path)
    print("ACC summary:")
    print(f"  best_val     = {best_val:.4f}")
    print(f"  test_acc     = {test_acc:.4f}")
    print(f"  teacher_acc  = {teacher_acc:.4f}")
    print("NOTE: vocab keys sidecar:", sidecar)

if __name__ == "__main__":
    main()