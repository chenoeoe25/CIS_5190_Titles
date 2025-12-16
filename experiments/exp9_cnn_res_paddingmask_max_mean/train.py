# experiments/exp_byte_tf/train.py
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
        ids = ids.to(device)
        yy = yy.to(device)
        logits = model.forward(ids)
        pred = logits.argmax(dim=1)
        correct += (pred == yy).sum().item()
        total += yy.numel()
    return correct / max(total, 1)


@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for ids, yy in loader:
        ids = ids.to(device)
        yy = yy.to(device)
        logits = model.forward(ids)
        loss = criterion(logits, yy)
        total_loss += loss.item() * yy.size(0)
        total += yy.size(0)
    return total_loss / max(total, 1)


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====== load data ======
    X, y = prepare_data("data/train_urls_url_only_60k.csv")
    n = len(y)

    tr_idx, va_idx, te_idx = split_indices(n, seed=0)

    def subset(idxs):
        return [X[i] for i in idxs], y[idxs].tolist()

    Xtr, ytr = subset(tr_idx)
    Xva, yva = subset(va_idx)
    Xte, yte = subset(te_idx)

    tr_loader = DataLoader(ListDataset(Xtr, ytr), batch_size=128, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(ListDataset(Xva, yva), batch_size=256, shuffle=False, collate_fn=collate_fn)
    te_loader = DataLoader(ListDataset(Xte, yte), batch_size=256, shuffle=False, collate_fn=collate_fn)

    # ====== model / optim ======
    model = Model().to(device)

    LR = 2e-4
    WD = 1e-2
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # ====== class weights ======
    ytr_tensor = torch.tensor(ytr, dtype=torch.long)
    n0 = (ytr_tensor == 0).sum().item()
    n1 = (ytr_tensor == 1).sum().item()
    w0 = (n0 + n1) / max(2 * n0, 1)
    w1 = (n0 + n1) / max(2 * n1, 1)
    class_w = torch.tensor([w0, w1], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # ====== saving ======
    out_dir = "experiments/exp8_cnn_res_paddingmask_max_mean/artifacts"
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "model_best.pt")
    final_path = os.path.join(out_dir, "model.pt")
    metrics_path = os.path.join(out_dir, "metrics.csv")

    # ====== training control ======
    EPOCHS = 200
    PATIENCE = 30
    SAVE_EVERY = 10
    GRAD_CLIP = 1.0

    best_val = -1.0
    no_improve = 0

    # ====== CSV logging (loss + acc) ======
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

        for ep in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0

            for ids, yy in tr_loader:
                ids = ids.to(device)
                yy = yy.to(device)

                opt.zero_grad(set_to_none=True)
                logits = model.forward(ids)
                loss = criterion(logits, yy)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

                total_loss += loss.item() * yy.size(0)

            train_loss = total_loss / max(len(tr_loader.dataset), 1)
            val_loss = eval_loss(model, va_loader, criterion, device)
            val_acc = eval_acc(model, va_loader, device)

            writer.writerow([ep, train_loss, val_loss, val_acc])
            f.flush()

            if val_acc > best_val + 1e-6:
                best_val = val_acc
                no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                no_improve += 1

            if ep % SAVE_EVERY == 0 or ep == 1:
                print(
                    f"Epoch {ep:03d}: "
                    f"train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.4f} "
                    f"best_val={best_val:.4f}"
                )

            if no_improve >= PATIENCE:
                print(f"Early stop at epoch {ep}, best val acc = {best_val:.4f}")
                break

    # ====== final eval ======
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    torch.save(model.state_dict(), final_path)

    test_acc = eval_acc(model, te_loader, device)

    X_teacher, y_teacher = prepare_data("data/url_only_data.csv")
    teacher_loader = DataLoader(
        ListDataset(X_teacher, y_teacher.tolist()),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
    )
    teacher_acc = eval_acc(model, teacher_loader, device)

    print("Saved:", final_path)
    print(f"best_val={best_val:.4f} test_acc={test_acc:.4f} teacher_acc={teacher_acc:.4f}")


if __name__ == "__main__":
    main()