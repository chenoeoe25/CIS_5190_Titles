import os
import pandas as pd
import matplotlib.pyplot as plt

# ====== paths ======
ARTIFACT_DIR = "experiments/exp8_cnn_res_paddingmask_max_mean/artifacts"
CSV_PATH = os.path.join(ARTIFACT_DIR, "metrics.csv")

LOSS_OUT_PATH = os.path.join(ARTIFACT_DIR, "loss_curve.png")
ACC_OUT_PATH = os.path.join(ARTIFACT_DIR, "acc_curve.png")

# ====== load data ======
df = pd.read_csv(CSV_PATH)

epochs = df["epoch"]
train_loss = df["train_loss"]
val_loss = df["val_loss"]
val_acc = df["val_acc"]

# ====== plot loss ======
plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(LOSS_OUT_PATH, dpi=150)
plt.close()

print(f"Saved loss curve to: {LOSS_OUT_PATH}")

# ====== plot accuracy ======
plt.figure(figsize=(6, 4))
plt.plot(epochs, val_acc, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(ACC_OUT_PATH, dpi=150)
plt.close()

print(f"Saved accuracy curve to: {ACC_OUT_PATH}")