Our final model is in /experiments/exp6_cnn_res_paddingmask/artifacts/model_best.pt

Environment:

If you are using conda: conda env export > environment.yml

If you are using pip  : pip install -r requirements.txt

Evaluations:

when you are inside CIS_5190_Titles Project

1. evaluate on train_urls_url_only.csv
   Run


python eval/eval_project_b.py \
  --model experiments/exp6_cnn_res_paddingmask/model.py \
  --preprocess experiments/exp6_cnn_res_paddingmask/preprocess.py \
  --csv data/train_urls_url_only.csv \
  --weights experiments/exp6_cnn_res_paddingmask/artifacts/model_best.pt \
  --batch-size 32

3. evaluate on train_urls_url_only_60k.csv
  Run


python eval/eval_project_b.py \
  --model experiments/exp6_cnn_res_paddingmask/model.py \
  --preprocess experiments/exp6_cnn_res_paddingmask/preprocess.py \
  --csv data/train_urls_url_only_60k.csv \
  --weights experiments/exp6_cnn_res_paddingmask/artifacts/model_best.pt \
  --batch-size 32

Training:
python -m experiments.exp6_cnn_res_paddingmask.train
