# experiments/exp_byte_tf/train.py
import os
import random
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
    ids = torch.stack([x["ids"] for x in Xs], dim=0)  # [B, L]
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


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====== load train data ======
    X, y = prepare_data("data/train_urls_url_only_60k.csv")
    n = len(y)

    tr_idx, va_idx, te_idx = split_indices(n, val_ratio=0.1, test_ratio=0.1, seed=0)

    def subset(idxs):
        Xs = [X[i] for i in idxs]
        ys = y[idxs].tolist()  # y is tensor
        return Xs, ys

    Xtr, ytr = subset(tr_idx)
    Xva, yva = subset(va_idx)
    Xte, yte = subset(te_idx)

    tr_loader = DataLoader(ListDataset(Xtr, ytr), batch_size=128, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(ListDataset(Xva, yva), batch_size=256, shuffle=False, collate_fn=collate_fn)
    te_loader = DataLoader(ListDataset(Xte, yte), batch_size=256, shuffle=False, collate_fn=collate_fn)

    # ====== model / optim ======
    model = Model().to(device)

    # ✅ 更合理的默认：更快收敛
    LR = 2e-4
    WD = 1e-2
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # ====== handle class imbalance (simple) ======
    # fox=1, nbc=0
    ytr_tensor = torch.tensor(ytr, dtype=torch.long)
    n0 = (ytr_tensor == 0).sum().item()
    n1 = (ytr_tensor == 1).sum().item()
    w0 = (n0 + n1) / max(2 * n0, 1)
    w1 = (n0 + n1) / max(2 * n1, 1)
    class_w = torch.tensor([w0, w1], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # ====== saving ======
    out_dir = "experiments/exp4_cnn2layer/artifacts"
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "model_best.pt")
    final_path = os.path.join(out_dir, "model.pt")

    # ====== training control ======
    EPOCHS = 200
    SAVE_EVERY = 10
    PATIENCE = 30  # ✅ early stop
    GRAD_CLIP = 1.0

    best_val = -1.0
    no_improve = 0

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

            # ✅ stabilize
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            opt.step()
            total_loss += loss.item() * yy.size(0)

        val_acc = eval_acc(model, va_loader, device)

        improved = val_acc > best_val + 1e-6
        if improved:
            best_val = val_acc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        if ep % SAVE_EVERY == 0 or ep == 1:
            avg_loss = total_loss / max(len(tr_loader.dataset), 1)
            ckpt_path = os.path.join(out_dir, f"ckpt_epoch{ep:03d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Epoch {ep:03d}: loss={avg_loss:.4f} val_acc={val_acc:.4f} best={best_val:.4f} no_improve={no_improve}")

        if no_improve >= PATIENCE:
            print(f"Early stop: no val improvement for {PATIENCE} epochs. Best={best_val:.4f}")
            break

    # ====== load best -> final save ======
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    torch.save(model.state_dict(), final_path)

    # ====== evaluate ======
    test_acc = eval_acc(model, te_loader, device)

    # teacher
    X_teacher, y_teacher = prepare_data("data/url_only_data.csv")
    teacher_loader = DataLoader(
        ListDataset(X_teacher, y_teacher.tolist()),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
    )
    teacher_acc = eval_acc(model, teacher_loader, device)

    print("Saved:", final_path)
    print("ACC summary:")
    print(f"  best_val     = {best_val:.4f}")
    print(f"  test_acc     = {test_acc:.4f}")
    print(f"  teacher_acc  = {teacher_acc:.4f}")
    print(f"  LR={LR} WD={WD} class_w={[float(w0), float(w1)]}")


if __name__ == "__main__":
    main()