import json
import re
import collections

# --- Metric 1: Answer F1 Helpers ---

def normalize_text(text: str) -> str:
    """
    Normalizes text for F1 evaluation:
    - Lowercases
    - Removes articles (a, an, the)
    - Removes punctuation EXCEPT decimal points between digits
    - Collapses whitespace
    """
    text = text.lower()
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Keep decimal points between digits, but remove other punctuation
    # 1. Replace valid decimals with a temporary placeholder
    text = re.sub(r'(?<=\d)\.(?=\d)', '<DECIMAL>', text)
    # 2. Remove all other punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # 3. Restore decimals
    text = text.replace('<DECIMAL>', '.')
    
    # Collapse whitespace
    text = ' '.join(text.split())
    return text

def compute_f1(predicted: str, ground_truth: str) -> float:
    """Computes SQuAD-style token-level F1 score."""
    pred_tokens = normalize_text(predicted).split()
    gt_tokens = normalize_text(ground_truth).split()
    
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    overlap = sum(common.values())
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 1.0 if pred_tokens == gt_tokens else 0.0
        
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
        
    return (2 * precision * recall) / (precision + recall)


# --- Metric 2: Exact Value Recall Helpers ---

def extract_measurable_values(text: str) -> list:
    """Extracts numeric values with units (ignoring spaces between them)."""
    # Collapse spaces between numbers and letters (e.g., "3.3 V" -> "3.3V")
    compact_text = re.sub(r'(\d)\s+([a-zA-Z°Ω%]+)', r'\1\2', text, flags=re.IGNORECASE)
    
    # Pattern: digits + optional decimal + specific units
    units = r'(?:V|mV|A|mA|uA|W|Hz|kHz|MHz|ns|us|ms|°C|pF|nF|Ω|%|bits|bytes|dB)'
    pattern = r'\b\d+(?:\.\d+)?' + units + r'\b'
    
    return re.findall(pattern, compact_text, flags=re.IGNORECASE)

def compute_exact_value_recall(predicted: str, gt_exact: str) -> float:
    """Computes exact value recall following Path A (Numeric) or Path B (Non-Numeric)."""
    gt_values = extract_measurable_values(gt_exact)
    
    # Path A: Numeric Questions
    if gt_values:
        # Compact prediction to ignore spacing, just like we did for GT
        pred_compact = re.sub(r'(\d)\s+([a-zA-Z°Ω%]+)', r'\1\2', predicted, flags=re.IGNORECASE).lower()
        matched = sum(1 for val in gt_values if val.lower() in pred_compact)
        return matched / len(gt_values)
        
    # Path B: Non-Numeric Questions
    else:
        stopwords = {'this', 'that', 'with', 'from', 'your', 'have', 'does', 'when'}
        # Extract significant words: length > 3 and not a stopword
        gt_words = [w.lower() for w in re.findall(r'\b\w+\b', gt_exact) 
                    if len(w) > 3 and w.lower() not in stopwords]
        
        if not gt_words:
            return 0.0 # Fallback for empty/unmatchable ground truth
            
        pred_lower = predicted.lower()
        matched = sum(1 for word in gt_words if word in pred_lower)
        return matched / len(gt_words)


# --- Metric 3 & 4: Tool Metrics Helpers ---

def compute_tool_validity(generated_calls: list) -> float:
    """Calculates the fraction of tool calls that are syntactically valid."""
    if not generated_calls:
        return 0.0
        
    valid_count = 0
    allowed_tools = {'text_search', 'table_search', 'figure_search'}
    
    for call in generated_calls:
        if isinstance(call, dict):
            has_valid_tool = call.get("tool") in allowed_tools
            has_valid_query = isinstance(call.get("query"), str) and call.get("query", "").strip() != ""
            
            if has_valid_tool and has_valid_query:
                valid_count += 1
                
    return valid_count / len(generated_calls)

def compute_tool_efficiency(generated_calls: list, expected_calls: list) -> float:
    """Calculates efficiency based on expected tool calls vs actual tool calls."""
    actual = len(generated_calls)
    expected = len(expected_calls)
    
    if expected == 0:
        return 1.0 if actual == 0 else 0.0
        
    recall = min(actual, expected) / expected
    precision = 0.5 ** max(0, actual - expected)  # Exponential decay for excess calls
    
    return recall * precision


# --- Main Evaluation Pipeline ---

def evaluate_predictions(predictions_file: str, ground_truth_file: str):
    """
    Evaluates a JSONL predictions file against a JSONL ground truth file line-by-line.
    Assumes both files have the exact same number of lines and are aligned.
    Returns a dictionary of aggregated metrics.
    """
    metrics = {
        "f1_scores": [],
        "exact_value_recalls": [],
        "tool_validities": [],
        "tool_efficiencies": [],
        "overall_scores": []
    }

    # Open both files and process them line-by-line simultaneously
    with open(predictions_file, 'r', encoding='utf-8') as f_pred, \
         open(ground_truth_file, 'r', encoding='utf-8') as f_gt:
        
        for line_num, (pred_line, gt_line) in enumerate(zip(f_pred, f_gt), 1):
            pred_obj = json.loads(pred_line)
            gt_obj = json.loads(gt_line)
            
            # Optional alignment check
            if pred_obj.get('question') != gt_obj.get('question'):
                print(f"Warning: Question mismatch at line {line_num}!\n"
                      f"  Pred: {pred_obj.get('question')}\n"
                      f"  GT:   {gt_obj.get('question')}")
            
            # Extract fields
            pred_ans = pred_obj.get('predicted_answer', '')
            # Extract the first sentence if multiple sentences are present
            # pred_ans = pred_ans.split('.')[0] if '.' in pred_ans else pred_ans
            gen_tools = pred_obj.get('tool_calls_generated', [])
            
            gt_full_ans = gt_obj.get('answer', '')
            gt_short_ans = gt_obj.get('ground_truth', '')
            gt_tools = gt_obj.get('tool_calls', [])

            # 1. Answer F1 (Max of full answer vs concise answer)
            f1_full = compute_f1(pred_ans, gt_full_ans)
            f1_short = compute_f1(pred_ans, gt_short_ans)
            f1 = max(f1_full, f1_short)
            
            # 2. Exact Value Recall
            exact_val = compute_exact_value_recall(pred_ans, gt_short_ans)
            
            # 3. Tool Validity
            tool_val = compute_tool_validity(gen_tools)
            
            # 4. Tool Efficiency
            tool_eff = compute_tool_efficiency(gen_tools, gt_tools)
            
            # Overall Score per example
            overall = (0.60 * f1) + (0.20 * exact_val) + (0.15 * tool_val) + (0.05 * tool_eff)
            
            metrics["f1_scores"].append(f1)
            metrics["exact_value_recalls"].append(exact_val)
            metrics["tool_validities"].append(tool_val)
            metrics["tool_efficiencies"].append(tool_eff)
            metrics["overall_scores"].append(overall)

    # Calculate Aggregated Averages
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    results = {
        "Answer F1": avg(metrics["f1_scores"]),
        "Exact Value Recall": avg(metrics["exact_value_recalls"]),
        "Tool Validity": avg(metrics["tool_validities"]),
        "Tool Efficiency": avg(metrics["tool_efficiencies"]),
        "Overall Score": avg(metrics["overall_scores"])
    }
    
    return results

# Example Usage Block
if __name__ == "__main__":
    # To run this script, supply the paths to your JSONL files:

    predictions_path = "data/submission.jsonl"
    ground_truth_path = "data/test_ground_truth.jsonl"
    print(f"Evaluating predictions from {predictions_path} against ground truth {ground_truth_path}...")
    
    final_scores = evaluate_predictions(predictions_path, ground_truth_path)
    
    print("\n--- Evaluation Results ---")
    for metric, score in final_scores.items():
        print(f"{metric:<20}: {score:.4f}")