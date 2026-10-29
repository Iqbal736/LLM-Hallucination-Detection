"""
Module: src/transformer_model.py
Description: Evaluates the Transformer-based Cross-Encoder ('cross-encoder/nli-distilroberta-base')
             on the 500 held-out test split for factual consistency and hallucination detection.
             Measures test accuracy, F1, latency, and generates predictions.
"""

import os
import time
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_NAME = "cross-encoder/nli-distilroberta-base"


def evaluate_transformer():
    print(f"[*] Initializing Transformer model: {MODEL_NAME}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Load test split (500 held-out test pairs)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    contexts = test_df["context"].tolist()
    generations = test_df["generation"].tolist()
    y_true = test_df["label"].values

    print(f"[*] Tokenizing and running batch inference on {len(test_df)} held-out test pairs...")
    batch_size = 16
    all_preds = []
    factual_probs = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i + batch_size]
            batch_gen = generations[i:i + batch_size]

            # Cross-Encoder joint sentence-pair encoding: [CLS] Context [SEP] Generation [SEP]
            inputs = tokenizer(
                batch_ctx,
                batch_gen,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # Mapping for NLI: 0=contradiction, 1=entailment, 2=neutral
            # Factual (1) if entailment score > contradiction score
            for prob in probs:
                p_contra = prob[0]
                p_entail = prob[1]
                p_neutral = prob[2]

                pred_label = 1 if p_entail >= p_contra else 0
                all_preds.append(pred_label)
                factual_probs.append(float(p_entail))

    total_infer_time = time.time() - start_time
    per_sample_latency_ms = (total_infer_time / len(test_df)) * 1000.0

    acc = accuracy_score(y_true, all_preds)
    macro_p = precision_score(y_true, all_preds, average="macro")
    macro_r = recall_score(y_true, all_preds, average="macro")
    macro_f1 = f1_score(y_true, all_preds, average="macro")
    weighted_f1 = f1_score(y_true, all_preds, average="weighted")
    cm = confusion_matrix(y_true, all_preds).tolist()
    class_rep = classification_report(y_true, all_preds, output_dict=True)

    metrics = {
        "model": "DistilRoBERTa Cross-Encoder (NLI)",
        "checkpoint": MODEL_NAME,
        "parameters": "≈ 82 million",
        "device": str(device),
        "total_inference_time_seconds": round(total_infer_time, 2),
        "inference_latency_ms_per_sample": round(per_sample_latency_ms, 2),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm,
        "class_report": class_rep
    }

    with open(os.path.join(RESULTS_DIR, "transformer_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    test_df_copy = test_df.copy()
    test_df_copy["pred_transformer"] = all_preds
    test_df_copy["prob_transformer_factual"] = factual_probs
    test_df_copy.to_csv(os.path.join(RESULTS_DIR, "transformer_test_predictions.csv"), index=False)

    print("\n[+] Transformer Cross-Encoder Test Results:")
    print(f"    - Accuracy:          {acc * 100:.2f}%")
    print(f"    - Macro F1:          {macro_f1 * 100:.2f}%")
    print(f"    - Class 0 (Hallu) F1:{class_rep['0']['f1-score'] * 100:.2f}%")
    print(f"    - Class 1 (Fact) F1: {class_rep['1']['f1-score'] * 100:.2f}%")
    print(f"    - Latency per Pair:  {per_sample_latency_ms:.2f} ms")

    return metrics


if __name__ == "__main__":
    evaluate_transformer()
