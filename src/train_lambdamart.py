import os
os.environ["HF_HOME"] = "../hf_cache"
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import lightgbm as lgb
from tqdm import tqdm
from sklearn.decomposition import PCA

class CodeRankerPipeline:
    def __init__(self, model_name='microsoft/codebert-base', batch_size=16, pca_components=32):
        self.model_name = model_name
        print(f"Initializing CodeRankerPipeline with model: {model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize Code LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size

        # Initialize PCA variables
        self.pca_components = pca_components
        self.pca = None
        
        # Initialize LightGBM Classifier (replacing LGBMRanker)
        self.classifier = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss", # Optimizes logloss for uniform pair accuracy
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=20,
            max_depth=-1,
            min_child_samples=40,
            reg_alpha=0.00,
            reg_lambda=0.00,
            random_state=42
        )

        # Print classifier parameters for verification
        print("Initialized LightGBM Classifier with parameters:")
        print(self.classifier.get_params())

    def _generate_pairwise_data(self, df, X):
        """
        Transforms the dataset into explicit pairs (X_i - X_j) for binary classification.
        Label is 1 if score_i > score_j, and 0 for the symmetric negative pair.
        """
        X_pairs = []
        y_pairs = []
        
        # Reset index to ensure integer indexing aligns with the X matrix
        df = df.reset_index(drop=True)
        
        for _, group in df.groupby('branch_id', sort=False):
            indices = group.index.values
            scores = group['score'].values
            n = len(indices)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if scores[i] == scores[j]:
                        continue # Skip exact ties
                        
                    # Difference vector representing the pair
                    diff = X[indices[i]] - X[indices[j]]
                    
                    if scores[i] > scores[j]:
                        X_pairs.append(diff)
                        y_pairs.append(1)
                        # Add the symmetric inverse to balance the dataset
                        X_pairs.append(-diff)
                        y_pairs.append(0)
                    else:
                        X_pairs.append(-diff)
                        y_pairs.append(1)
                        # Add the symmetric inverse
                        X_pairs.append(diff)
                        y_pairs.append(0)
                        
        if not X_pairs:
            return np.array([]), np.array([])
            
        return np.vstack(X_pairs), np.array(y_pairs)

    def train(self, train_path, val_ratio=0.2):
        """
        Splits data temporally, generates pairwise differences, and trains a binary classifier.
        """
        print("Loading training data...")
        df = pd.read_csv(train_path)
        
        # Sort to ensure temporal split
        df = df.sort_values(by=['branch_id', 'journal_step']).reset_index(drop=True)
        
        print(f"Splitting data temporally within each group (val_ratio={val_ratio})...")
        train_frames = []
        val_frames = []
        
        for branch_id, group in df.groupby('branch_id', sort=False):
            n = len(group)
            split_idx = int(n * (1 - val_ratio))
            
            if n >= 3:
                if n - split_idx < 2:
                    split_idx = n - 2
                if split_idx < 1:
                    split_idx = 1
            elif n == 2:
                split_idx = 1
            else:
                split_idx = 1
                
            train_frames.append(group.iloc[:split_idx])
            if n - split_idx > 0:
                val_frames.append(group.iloc[split_idx:])
                
        df_train = pd.concat(train_frames).reset_index(drop=True)
        df_val = pd.concat(val_frames).reset_index(drop=True) if val_frames else None
        
        print("Preparing base features...")
        X_train_base = self.prepare_features(df_train)
        
        print(f"Generating explicit pairwise training pairs...")
        X_train_pairs, y_train_pairs = self._generate_pairwise_data(df_train, X_train_base)
        print(f"Generated {len(y_train_pairs)} training pairs.")
        
        fit_kwargs = {}
        
        # Process validation data if it exists
        if df_val is not None and not df_val.empty:
            X_val_base = self.prepare_features(df_val)
            X_val_pairs, y_val_pairs = self._generate_pairwise_data(df_val, X_val_base)
            
            if len(y_val_pairs) > 0:
                print(f"Generated {len(y_val_pairs)} validation pairs.")
                fit_kwargs["eval_set"] = [(X_train_pairs, y_train_pairs), (X_val_pairs, y_val_pairs)]
                fit_kwargs["eval_names"] = ['training', 'validation']
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(stopping_rounds=20, verbose=True),
                    lgb.log_evaluation(period=10)
                ]
        
        print("Training LightGBM Pairwise Classifier...")
        self.classifier.fit(X_train_pairs, y_train_pairs, **fit_kwargs)
        print("Training complete.")

    def predict_and_format(self, holdout_path, output_path='submission.csv'):
        """
        Runs inference using a Borda count approach (sum of pairwise winning probabilities)
        to assign absolute scores and formats the submission.
        """
        print("Loading holdout data...")
        df_holdout = pd.read_csv(holdout_path).reset_index(drop=True)
        
        print("Preparing holdout features...")
        X_holdout = self.prepare_features(df_holdout)
        
        print("Generating Borda count ranking predictions...")
        global_pred_scores = np.zeros(len(df_holdout))
        
        # Iterate over each branch to predict pairwise outcomes
        for branch_id, group in tqdm(df_holdout.groupby('branch_id', sort=False), desc="Scoring Branches"):
            indices = group.index.values
            n = len(indices)
            
            if n <= 1:
                # If only one item, it gets a default score of 0 (will become Rank 1)
                continue
                
            # Build all unique (i, j) pairs for this branch
            pairs_i = []
            pairs_j = []
            diffs = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    pairs_i.append(indices[i])
                    pairs_j.append(indices[j])
                    diffs.append(X_holdout[indices[i]] - X_holdout[indices[j]])
                    
            # Vectorized prediction: Probability that program `i` is better than program `j`
            diffs_matrix = np.vstack(diffs)
            p_i_beats_j = self.classifier.predict_proba(diffs_matrix)[:, 1]
            
            # Accumulate scores for each program
            for idx, (i, j) in enumerate(zip(pairs_i, pairs_j)):
                p = p_i_beats_j[idx]
                global_pred_scores[i] += p
                global_pred_scores[j] += (1 - p)
                
        df_holdout['pred_score'] = global_pred_scores
        
        # Convert cumulative win-probabilities to 1-based ranks
        df_holdout['rank'] = df_holdout.groupby('branch_id')['pred_score'].rank(
            method='first', 
            ascending=False
        ).astype(int)
        
        submission = df_holdout[['code_path', 'rank']]
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        
    def extract_code_embeddings(self, code_series):
        """Extracts dense vector representations from the source code."""
        embeddings = []
        code_list = code_series.tolist()
        
        for i in tqdm(range(0, len(code_list), self.batch_size), desc="Extracting embeddings"):
            batch = code_list[i:i + self.batch_size]
            
            # Tokenize with truncation to handle long code snippets
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                if self.model_name.startswith("Qwen"):
                    last_hidden_state = outputs.last_hidden_state
                
                    # Get the attention mask to find the last non-padded token
                    attention_mask = inputs['attention_mask']
                    
                    # Calculate the actual lengths of sequences (subtract 1 for 0-based indexing)
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    
                    # Extract the embedding of the last valid token for each item in the batch
                    current_batch_size = last_hidden_state.shape[0]
                    batch_embeddings = last_hidden_state[
                        torch.arange(current_batch_size, device=last_hidden_state.device), 
                        sequence_lengths
                    ].to(torch.float32).cpu().numpy() # Cast to float32 before calling .numpy() #.cpu().numpy()
                    
                    # Cast from bfloat16 to float32 before sending to LightGBM
                    embeddings.append(batch_embeddings) #.astype(np.float32))
                else:
                    # Use the [CLS] token representation as the aggregate embedding
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(cls_embeddings)
                
        return np.vstack(embeddings)

    def prepare_features(self, df, is_train=False):
        """Combines LLM embeddings with scalar metadata."""
        # 1. Extract LLM features
        code_embeds = self.extract_code_embeddings(df['code'])

        # Apply PCA to embeddings
        if self.pca_components is not None:
            if is_train:
                print(f"Fitting PCA with {self.pca_components} components...")
                n_samples = code_embeds.shape[0]
                n_comps = self.pca_components
                if isinstance(n_comps, int) and n_comps > n_samples:
                    n_comps = n_samples
                    print(f"Warning: Capping PCA components to {n_samples} due to small train size.")
                
                self.pca = PCA(n_components=n_comps, random_state=42)
                
                # --- NEW: Keep a copy of original embeddings for reconstruction error ---
                original_embeds = code_embeds.copy()
                
                # Fit and transform
                code_embeds = self.pca.fit_transform(code_embeds)
                
                # --- NEW: Calculate reconstruction error and explained variance ---
                reconstructed_embeds = self.pca.inverse_transform(code_embeds)
                mse = np.mean((original_embeds - reconstructed_embeds) ** 2)
                explained_var = np.sum(self.pca.explained_variance_ratio_)
                
                print(f"PCA reduced embeddings to shape: {code_embeds.shape}")
                print(f"➔ PCA Reconstruction MSE (Closeness to original): {mse:.6f}")
                print(f"➔ PCA Total Explained Variance: {explained_var * 100:.2f}%\n")
                
            else:
                if self.pca is not None:
                    code_embeds = self.pca.transform(code_embeds)
        
        # 2. Extract scalar features (journal_step captures temporal evolution)
        scalar_features = df[['journal_step']].values
        
        # Combine into a single feature matrix
        X = np.hstack((code_embeds, scalar_features))
        return X

    def _generate_pairwise_data(self, df, X):
        """
        Transforms the dataset into explicit pairs (X_i - X_j) for binary classification.
        Label is 1 if score_i > score_j, and 0 for the symmetric negative pair.
        """
        X_pairs = []
        y_pairs = []
        
        # Reset index to ensure integer indexing aligns with the X matrix
        df = df.reset_index(drop=True)
        
        for _, group in df.groupby('branch_id', sort=False):
            indices = group.index.values
            scores = group['score'].values
            n = len(indices)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if scores[i] == scores[j]:
                        continue # Skip exact ties
                        
                    # Difference vector representing the pair
                    diff = X[indices[i]] - X[indices[j]]
                    
                    if scores[i] > scores[j]:
                        X_pairs.append(diff)
                        y_pairs.append(1)
                        # Add the symmetric inverse to balance the dataset
                        X_pairs.append(-diff)
                        y_pairs.append(0)
                    else:
                        X_pairs.append(-diff)
                        y_pairs.append(1)
                        # Add the symmetric inverse
                        X_pairs.append(diff)
                        y_pairs.append(0)
                        
        if not X_pairs:
            return np.array([]), np.array([])
            
        return np.vstack(X_pairs), np.array(y_pairs)

    def train(self, train_path, val_ratio=0.2, val_pred_path='val_predictions.csv'):
        """
        Splits data temporally, generates pairwise differences, and trains a binary classifier.
        """
        print("Loading training data...")
        df = pd.read_csv(train_path)
        
        # Sort to ensure temporal split
        df = df.sort_values(by=['branch_id', 'journal_step']).reset_index(drop=True)
        
        print(f"Splitting data temporally within each group (val_ratio={val_ratio})...")
        train_frames = []
        val_frames = []
        
        for branch_id, group in df.groupby('branch_id', sort=False):
            n = len(group)
            split_idx = int(n * (1 - val_ratio))
            
            if n >= 3:
                if n - split_idx < 2:
                    split_idx = n - 2
                if split_idx < 1:
                    split_idx = 1
            elif n == 2:
                split_idx = 1
            else:
                split_idx = 1
                
            train_frames.append(group.iloc[:split_idx])
            if n - split_idx > 0:
                val_frames.append(group.iloc[split_idx:])
                
        df_train = pd.concat(train_frames).reset_index(drop=True)
        df_val = pd.concat(val_frames).reset_index(drop=True) if val_frames else None
        
        print("Preparing base features...")
        X_train_base = self.prepare_features(df_train, is_train=True)
        
        print(f"Generating explicit pairwise training pairs...")
        X_train_pairs, y_train_pairs = self._generate_pairwise_data(df_train, X_train_base)
        print(f"Generated {len(y_train_pairs)} training pairs.")
        
        fit_kwargs = {}
        
        # Process validation data if it exists
        if df_val is not None and not df_val.empty:
            X_val_base = self.prepare_features(df_val, is_train=False)
            X_val_pairs, y_val_pairs = self._generate_pairwise_data(df_val, X_val_base)
            
            if len(y_val_pairs) > 0:
                print(f"Generated {len(y_val_pairs)} validation pairs.")
                fit_kwargs["eval_set"] = [(X_train_pairs, y_train_pairs), (X_val_pairs, y_val_pairs)]
                fit_kwargs["eval_names"] = ['training', 'validation']
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(stopping_rounds=20, verbose=True),
                    lgb.log_evaluation(period=10)
                ]
        
        print("Training LightGBM Pairwise Classifier...")
        self.classifier.fit(X_train_pairs, y_train_pairs, **fit_kwargs)
        print("Training complete.")

         # --- NEW CODE: Predict and save best predictions on the validation set ---
        if df_val is not None and not df_val.empty:
            print("Generating Borda count predictions for the validation set...")
            val_pred_scores = np.zeros(len(df_val))
            
            for branch_id, group in tqdm(df_val.groupby('branch_id', sort=False), desc="Scoring Val Branches"):
                indices = group.index.values
                n = len(indices)
                
                if n <= 1:
                    continue
                    
                pairs_i = []
                pairs_j = []
                diffs = []
                
                for i in range(n):
                    for j in range(i + 1, n):
                        pairs_i.append(indices[i])
                        pairs_j.append(indices[j])
                        diffs.append(X_val_base[indices[i]] - X_val_base[indices[j]])
                        
                diffs_matrix = np.vstack(diffs)
                # predict_proba automatically uses best_iteration_ from early stopping
                p_i_beats_j = self.classifier.predict_proba(diffs_matrix)[:, 1]
                
                for idx, (i, j) in enumerate(zip(pairs_i, pairs_j)):
                    p = p_i_beats_j[idx]
                    val_pred_scores[i] += p
                    val_pred_scores[j] += (1 - p)
                    
            df_val['pred_score'] = val_pred_scores
            df_val['pred_rank'] = df_val.groupby('branch_id')['pred_score'].rank(
                method='first', 
                ascending=False
            ).astype(int)
            
            # Save relevant columns to evaluate validation metrics later
            cols_to_save = [c for c in ['branch_id', 'journal_step', 'score', 'pred_score', 'pred_rank', 'code_path'] if c in df_val.columns]
            df_val[cols_to_save].to_csv(val_pred_path, index=False)
            print(f"Validation predictions saved to {val_pred_path}")

    def predict_and_format(self, holdout_path, output_path='submission.csv'):
        """
        Runs inference using a Borda count approach (sum of pairwise winning probabilities)
        to assign absolute scores and formats the submission.
        """
        print("Loading holdout data...")
        df_holdout = pd.read_csv(holdout_path).reset_index(drop=True)
        
        print("Preparing holdout features...")
        X_holdout = self.prepare_features(df_holdout, is_train=False)
        
        print("Generating Borda count ranking predictions...")
        global_pred_scores = np.zeros(len(df_holdout))
        
        # Iterate over each branch to predict pairwise outcomes
        for branch_id, group in tqdm(df_holdout.groupby('branch_id', sort=False), desc="Scoring Branches"):
            indices = group.index.values
            n = len(indices)
            
            if n <= 1:
                # If only one item, it gets a default score of 0 (will become Rank 1)
                continue
                
            # Build all unique (i, j) pairs for this branch
            pairs_i = []
            pairs_j = []
            diffs = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    pairs_i.append(indices[i])
                    pairs_j.append(indices[j])
                    diffs.append(X_holdout[indices[i]] - X_holdout[indices[j]])
                    
            # Vectorized prediction: Probability that program `i` is better than program `j`
            diffs_matrix = np.vstack(diffs)
            p_i_beats_j = self.classifier.predict_proba(diffs_matrix)[:, 1]
            
            # Accumulate scores for each program
            for idx, (i, j) in enumerate(zip(pairs_i, pairs_j)):
                p = p_i_beats_j[idx]
                global_pred_scores[i] += p
                global_pred_scores[j] += (1 - p)
                
        df_holdout['pred_score'] = global_pred_scores
        
        # Convert cumulative win-probabilities to 1-based ranks
        df_holdout['rank'] = df_holdout.groupby('branch_id')['pred_score'].rank(
            method='first', 
            ascending=False
        ).astype(int)
        
        submission = df_holdout[['code_path', 'rank']]
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    # Define file paths
    folder_dir = "ventilator_pressure_proxy_ranking_assignment"
    TRAIN_FILE = f'{folder_dir}/data/train_with_labels.csv'
    HOLDOUT_FILE = f'{folder_dir}/data/holdout_no_labels.csv'
    SUBMISSION_FILE = f'data/tree_v2.csv'
    VAL_PREDICTIONS_FILE = f'data/val_v2.csv' # Set validation output
    
    # Initialize and run pipeline
    pipeline = CodeRankerPipeline(model_name="microsoft/codebert-base", pca_components=128) # Set PCA components
    pipeline.train(TRAIN_FILE, val_pred_path=VAL_PREDICTIONS_FILE)
    pipeline.predict_and_format(HOLDOUT_FILE, SUBMISSION_FILE)