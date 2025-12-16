# experiments/exp6_cnn/plot.py
import os
import pandas as pd
import matplotlib.pyplot as plt

ARTIFACT_DIR = "experiments/exp6_cnn_res_paddingmask/artifacts"
CSV_PATH = os.path.join(ARTIFACT_DIR, "metrics.csv")

df = pd.read_csv(CSV_PATH)

# ===== loss curve =====
plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(os.path.join(ARTIFACT_DIR, "loss_curve.png"))
plt.close()

# ===== acc curve =====
plt.figure()
plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(os.path.join(ARTIFACT_DIR, "acc_curve.png"))
plt.close()

print("Saved loss_curve.png and acc_curve.png")