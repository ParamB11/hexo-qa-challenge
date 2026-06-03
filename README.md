# Challenge 1: QA Agent
## Step 1: Run the servers
```
bash scripts/run_text.sh
bash scripts/run_table.sh
```

## Step 2: Run Inference
```
python src/hybrid_infer.py
```

## Step 3: Evaluate
```
python src/eval.py
```

# Challenge 2: Proxy Ranking
For training:
```
python src/train_lambdamart.py
```

For evaluation of the validation dataset:
```
python src/eval_kendall.py
```
