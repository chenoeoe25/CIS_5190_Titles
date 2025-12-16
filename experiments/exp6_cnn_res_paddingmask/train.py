# experiments/exp6_cnn/train.py
import os
import random
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .preprocess import prepare_data
from .model import Model


def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ListDataset(Dataset):
    def __init__(self, X_list, y_list):
        self.X = X_list
        self.y = y_list

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def collate_fn(batch):
    Xs, ys = zip(*batch)
    ids = torch.stack([x["ids"] for x in Xs], dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return ids, y


def split_indices(n, val_ratio=0.1, test_ratio=0.1, seed=0):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train, val, test


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for ids, yy in loader:
        ids, yy = ids.to(device), yy.to(device)
        pred = model(ids).argmax(dim=1)
        correct += (pred == yy).sum().item()
        total += yy.numel()
    return correct / max(total, 1)


@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    for ids, yy in loader:
        ids, yy = ids.to(device), yy.to(device)
        loss = criterion(model(ids), yy)
        total_loss += loss.item() * yy.size(0)
        total += yy.size(0)
    return total_loss / max(total, 1)


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====== data ======
    # X, y = prepare_data("data/train_urls_url_only_60k.csv")
    X, y = prepare_data("data/train_urls_url_only.csv")
    tr_idx, va_idx, te_idx = split_indices(len(y))

    def subset(idxs):
        return [X[i] for i in idxs], y[idxs].tolist()

    Xtr, ytr = subset(tr_idx)
    Xva, yva = subset(va_idx)
    Xte, yte = subset(te_idx)

    tr_loader = DataLoader(ListDataset(Xtr, ytr), batch_size=128, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(ListDataset(Xva, yva), batch_size=256, shuffle=False, collate_fn=collate_fn)
    te_loader = DataLoader(ListDataset(Xte, yte), batch_size=256, shuffle=False, collate_fn=collate_fn)

    # ====== model ======
    model = Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    # class weight
    ytr_t = torch.tensor(ytr)
    n0, n1 = (ytr_t == 0).sum(), (ytr_t == 1).sum()
    w = torch.tensor([(n0+n1)/(2*n0), (n0+n1)/(2*n1)], device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    # ====== saving ======
    out_dir = "experiments/exp6_cnn_res_paddingmask/artifacts"
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "model_best.pt")
    metrics_path = os.path.join(out_dir, "metrics.csv")

    # ====== training ======
    EPOCHS = 50
    PATIENCE = 30
    best_val, no_improve = -1, 0

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

        for ep in range(1, EPOCHS + 1):
            model.train()
            total_loss, correct, total = 0.0, 0, 0

            for ids, yy in tr_loader:
                ids, yy = ids.to(device), yy.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(ids)
                loss = criterion(logits, yy)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                total_loss += loss.item() * yy.size(0)
                correct += (logits.argmax(1) == yy).sum().item()
                total += yy.size(0)

            train_loss = total_loss / total
            train_acc = correct / total
            val_loss = eval_loss(model, va_loader, criterion, device)
            val_acc = eval_acc(model, va_loader, device)

            writer.writerow([ep, train_loss, val_loss, train_acc, val_acc])
            f.flush()

            if val_acc > best_val:
                best_val = val_acc
                no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                no_improve += 1

            print(f"Epoch {ep:03d} | "
                  f"train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} "
                  f"train_acc={train_acc:.4f} "
                  f"val_acc={val_acc:.4f}")

            if no_improve >= PATIENCE:
                print(f"Early stop at epoch {ep}, best val acc = {best_val:.4f}")
                break

    # ====== test ======
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    test_acc = eval_acc(model, te_loader, device)
    print(f"Final test acc = {test_acc:.4f}")


if __name__ == "__main__":
    main()