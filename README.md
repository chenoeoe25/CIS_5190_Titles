````markdown
# CIS 5190 Project: URL Classification

This repository contains the code and experiments for the CIS 5190 course project on URL classification.

---

## Environment Setup

Make sure you are in the root directory of the project (`CIS_5190_Titles_Project`).

### Option 1: Using Conda

```bash
conda env create -f environment.yml
````

Activate the environment after creation:

```bash
conda activate cis5190
```

### Option 2: Using Pip

```bash
pip install -r requirements.txt
```

---

## Training

To train the model using the `exp6_cnn_res_paddingmask` experiment configuration, run:

```bash
python -m experiments.exp6_cnn_res_paddingmask.train
```

---

## Evaluation

### Best Model Checkpoint

The trained weights for the final model are stored at:

```
experiments/exp6_cnn_res_paddingmask/artifacts/model_best.pt
```

### Running Evaluation

All evaluation commands should be executed from the project root directory.

#### 1. Evaluation on `train_urls_url_only.csv`

```bash
python eval/eval_project_b.py \
  --model experiments/exp6_cnn_res_paddingmask/model.py \
  --preprocess experiments/exp6_cnn_res_paddingmask/preprocess.py \
  --csv data/train_urls_url_only.csv \
  --weights experiments/exp6_cnn_res_paddingmask/artifacts/model_best.pt \
  --batch-size 32
```

#### 2. Evaluation on `train_urls_url_only_60k.csv`

```bash
python eval/eval_project_b.py \
  --model experiments/exp6_cnn_res_paddingmask/model.py \
  --preprocess experiments/exp6_cnn_res_paddingmask/preprocess.py \
  --csv data/train_urls_url_only_60k.csv \
  --weights experiments/exp6_cnn_res_paddingmask/artifacts/model_best.pt \
  --batch-size 32
```

