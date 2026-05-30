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
python src/temp_eval_v2.py
```