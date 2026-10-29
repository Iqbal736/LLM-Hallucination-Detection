"""
Module: src/baseline_model.py
Description: Trains and evaluates the classical Lexical Overlap + Logistic Regression baseline
             for factual consistency / hallucination detection.
             Includes hyperparameter tuning of regularisation parameter C on validation set.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from preprocessing import extract_pair_features

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_matrix(contexts, generations):
    rows = [extract_pair_features(c, g) for c, g in zip(contexts, generations)]
    return np.vstack(rows)


def train_and_evaluate_baseline():
    print("[*] Loading train, val, and test splits...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print("[*] Extracting domain-agnostic lexical overlap and negation features...")
    X_train = extract_matrix(train_df["context"], train_df["generation"])
    y_train = train_df["label"].values

    X_val = extract_matrix(val_df["context"], val_df["generation"])
    y_val = val_df["label"].values

    X_test = extract_matrix(test_df["context"], test_df["generation"])
    y_test = test_df["label"].values

    # Step 1: Hyperparameter Tuning of C on Validation Set
    print("\n[*] Tuning Logistic Regression Regularization Parameter C on Validation Split...")
    c_candidates = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_c = 1.0
    best_val_f1 = -1.0
    tuning_log = []

    for c in c_candidates:
        temp_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=c,
                solver="lbfgs",
                max_iter=1000,
                random_state=42
            ))
        ])
        temp_pipeline.fit(X_train, y_train)
        val_preds = temp_pipeline.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average="macro")
        tuning_log.append({"C": c, "val_macro_f1": round(val_f1, 4)})
        print(f"    - C={c:5.2f} -> Validation Macro F1 = {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c

    print(f"\n[+] Selected Optimal C = {best_c} (Best Val Macro F1: {best_val_f1:.4f})")

    # Step 2: Train Final Pipeline with Optimal C
    print(f"[*] Training final Lexical Baseline Pipeline (C={best_c}) on 4,000 training pairs...")
    start_train_time = time.time()
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=best_c,
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        ))
    ])
    final_pipeline.fit(X_train, y_train)
    training_duration = time.time() - start_train_time
    print(f"[+] Baseline trained successfully in {training_duration:.2f} seconds.")

    # Save trained pipeline
    joblib.dump(final_pipeline, os.path.join(MODELS_DIR, "baseline_pipeline.joblib"))

    # Step 3: Evaluate on Test Split
    print("[*] Evaluating baseline on 500 held-out test pairs...")
    start_infer_time = time.time()
    test_preds = final_pipeline.predict(X_test)
    test_probs = final_pipeline.predict_proba(X_test)
    total_infer_time = time.time() - start_infer_time
    per_sample_latency_ms = (total_infer_time / len(X_test)) * 1000.0

    acc = accuracy_score(y_test, test_preds)
    macro_p = precision_score(y_test, test_preds, average="macro")
    macro_r = recall_score(y_test, test_preds, average="macro")
    macro_f1 = f1_score(y_test, test_preds, average="macro")
    weighted_f1 = f1_score(y_test, test_preds, average="weighted")
    cm = confusion_matrix(y_test, test_preds).tolist()
    class_rep = classification_report(y_test, test_preds, output_dict=True)

    metrics = {
        "model": "Lexical Overlap + Logistic Regression",
        "optimal_C": best_c,
        "training_time_seconds": round(training_duration, 4),
        "inference_latency_ms_per_sample": round(per_sample_latency_ms, 4),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm,
        "class_report": class_rep,
        "tuning_log": tuning_log
    }

    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    test_df_copy = test_df.copy()
    test_df_copy["pred_baseline"] = test_preds
    test_df_copy["prob_baseline_factual"] = [p[1] for p in test_probs]
    test_df_copy.to_csv(os.path.join(RESULTS_DIR, "baseline_test_predictions.csv"), index=False)

    print("\n[+] Baseline Test Results:")
    print(f"    - Accuracy:          {acc * 100:.2f}%")
    print(f"    - Macro F1:          {macro_f1 * 100:.2f}%")
    print(f"    - Class 0 (Hallu) F1:{class_rep['0']['f1-score'] * 100:.2f}%")
    print(f"    - Class 1 (Fact) F1: {class_rep['1']['f1-score'] * 100:.2f}%")
    print(f"    - Latency per Pair:  {per_sample_latency_ms:.4f} ms")

    return metrics


if __name__ == "__main__":
    train_and_evaluate_baseline()
