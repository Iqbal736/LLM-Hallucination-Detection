"""
Module: src/evaluation.py
Description: Merges predictions from Baseline and Transformer models, produces
             the comparative analysis tables (Table 8, 9, 10, 11), builds error_analysis.csv,
             and generates Figures 7, 8, 9, 10, 11.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def run_comprehensive_evaluation():
    print("[*] Running comprehensive evaluation and building visual comparison figures...")

    # Load metrics
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "r") as f:
        base_metrics = json.load(f)

    with open(os.path.join(RESULTS_DIR, "transformer_metrics.json"), "r") as f:
        trans_metrics = json.load(f)

    # Load predictions
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    base_preds_df = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_test_predictions.csv"))
    trans_preds_df = pd.read_csv(os.path.join(RESULTS_DIR, "transformer_test_predictions.csv"))

    merged_df = test_df.copy()
    merged_df["pred_baseline"] = base_preds_df["pred_baseline"]
    merged_df["prob_baseline_factual"] = base_preds_df["prob_baseline_factual"]
    merged_df["pred_transformer"] = trans_preds_df["pred_transformer"]
    merged_df["prob_transformer_factual"] = trans_preds_df["prob_transformer_factual"]

    y_true = merged_df["label"].values
    y_base = merged_df["pred_baseline"].values
    y_trans = merged_df["pred_transformer"].values

    # Determine agreement categories
    both_correct = int(((y_base == y_true) & (y_trans == y_true)).sum())
    only_trans_correct = int(((y_base != y_true) & (y_trans == y_true)).sum())
    only_base_correct = int(((y_base == y_true) & (y_trans != y_true)).sum())
    both_wrong = int(((y_base != y_true) & (y_trans != y_true)).sum())
    total_test = len(y_true)

    merged_df["analysis_category"] = "Both Wrong"
    merged_df.loc[(y_base == y_true) & (y_trans == y_true), "analysis_category"] = "Both Correct"
    merged_df.loc[(y_base != y_true) & (y_trans == y_true), "analysis_category"] = "Only Transformer Correct"
    merged_df.loc[(y_base == y_true) & (y_trans != y_true), "analysis_category"] = "Only Baseline Correct"

    merged_df.to_csv(os.path.join(RESULTS_DIR, "error_analysis.csv"), index=False)
    print(f"[+] Saved error analysis to {os.path.join(RESULTS_DIR, 'error_analysis.csv')}")

    # Build Figures
    sns.set_theme(style="white")

    # Figure 7: Confusion Matrices
    cm_base = np.array(base_metrics["confusion_matrix"])
    cm_trans = np.array(trans_metrics["confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Hallucinated (0)", "Factual (1)"],
                yticklabels=["Hallucinated (0)", "Factual (1)"])
    axes[0].set_title(f"Confusion Matrix: TF-IDF + Logistic Regression\nAccuracy: {base_metrics['accuracy']*100:.2f}%", fontsize=11, weight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=10)
    axes[0].set_ylabel("True Label", fontsize=10)

    sns.heatmap(cm_trans, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1],
                xticklabels=["Hallucinated (0)", "Factual (1)"],
                yticklabels=["Hallucinated (0)", "Factual (1)"])
    axes[1].set_title(f"Confusion Matrix: DistilRoBERTa Cross-Encoder\nAccuracy: {trans_metrics['accuracy']*100:.2f}%", fontsize=11, weight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=10)
    axes[1].set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig7_confusion_matrices.png"), dpi=300)
    plt.close()

    # Figure 8: Overall Metrics Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    metrics_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    base_vals = [base_metrics["accuracy"] * 100, base_metrics["macro_precision"] * 100,
                 base_metrics["macro_recall"] * 100, base_metrics["macro_f1"] * 100]
    trans_vals = [trans_metrics["accuracy"] * 100, trans_metrics["macro_precision"] * 100,
                  trans_metrics["macro_recall"] * 100, trans_metrics["macro_f1"] * 100]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.bar(x - width/2, base_vals, width, label="TF-IDF + Logistic Regression", color="#4a90e2")
    plt.bar(x + width/2, trans_vals, width, label="DistilRoBERTa Cross-Encoder", color="#2ecc71")

    plt.ylabel("Score (%)", fontsize=11)
    plt.title("Model Comparison: TF-IDF + LR vs DistilRoBERTa (Held-Out Test Set)", fontsize=13, weight="bold")
    plt.xticks(x, metrics_names, fontsize=10)
    plt.ylim(50, 100)
    plt.legend(loc="lower right")

    for i in range(len(metrics_names)):
        plt.text(x[i] - width/2, base_vals[i] + 0.8, f"{base_vals[i]:.2f}%", ha="center", fontsize=9)
        plt.text(x[i] + width/2, trans_vals[i] + 0.8, f"{trans_vals[i]:.2f}%", ha="center", fontsize=9, weight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig8_overall_metrics.png"), dpi=300)
    plt.close()

    # Figure 9: Class-wise Metrics Bar Chart
    plt.figure(figsize=(10, 5))
    labels_class = ["Hallu Prec", "Hallu Rec", "Hallu F1", "Fact Prec", "Fact Rec", "Fact F1"]
    rep_b = base_metrics["class_report"]
    rep_t = trans_metrics["class_report"]

    vals_b = [
        rep_b["0"]["precision"] * 100, rep_b["0"]["recall"] * 100, rep_b["0"]["f1-score"] * 100,
        rep_b["1"]["precision"] * 100, rep_b["1"]["recall"] * 100, rep_b["1"]["f1-score"] * 100
    ]
    vals_t = [
        rep_t["0"]["precision"] * 100, rep_t["0"]["recall"] * 100, rep_t["0"]["f1-score"] * 100,
        rep_t["1"]["precision"] * 100, rep_t["1"]["recall"] * 100, rep_t["1"]["f1-score"] * 100
    ]

    xc = np.arange(len(labels_class))
    plt.bar(xc - width/2, vals_b, width, label="TF-IDF + LR", color="#4a90e2")
    plt.bar(xc + width/2, vals_t, width, label="DistilRoBERTa", color="#2ecc71")

    plt.ylabel("Score (%)", fontsize=11)
    plt.title("Class-wise Precision, Recall and F1-score (Held-Out Test Set)", fontsize=13, weight="bold")
    plt.xticks(xc, labels_class, fontsize=10)
    plt.ylim(50, 100)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig9_classwise_metrics.png"), dpi=300)
    plt.close()

    # Figure 10: Model Agreement Bar Chart
    plt.figure(figsize=(9, 4.5))
    cats = ["Both correct", "Only DistilRoBERTa correct", "Only TF-IDF + LR correct", "Both wrong"]
    counts = [both_correct, only_trans_correct, only_base_correct, both_wrong]
    colors = ["#7f8c8d", "#2ecc71", "#4a90e2", "#e74c3c"]

    y_pos = np.arange(len(cats))
    bars = plt.barh(y_pos, counts, color=colors, height=0.6)
    plt.yticks(y_pos, cats, fontsize=10)
    plt.xlabel("Number of Test Pairs", fontsize=11)
    plt.title("Prediction Agreement Between the Two Models on Test Set", fontsize=12, weight="bold")
    plt.gca().invert_yaxis()

    for bar, count in zip(bars, counts):
        pct = (count / total_test) * 100
        plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                 f"{count} ({pct:.1f}%)", va="center", fontsize=10, weight="bold")

    plt.xlim(0, max(counts) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig10_model_agreement.png"), dpi=300)
    plt.close()

    # Figure 11: Inference Latency Comparison (Log Scale)
    plt.figure(figsize=(8, 4))
    lat_models = ["TF-IDF + LR", "DistilRoBERTa"]
    lat_values = [base_metrics["inference_latency_ms_per_sample"], trans_metrics["inference_latency_ms_per_sample"]]
    lat_colors = ["#4a90e2", "#2ecc71"]

    y_l = np.arange(len(lat_models))
    plt.barh(y_l, lat_values, color=lat_colors, height=0.5)
    plt.yticks(y_l, lat_models, fontsize=11, weight="bold")
    plt.xscale("log")
    plt.xlabel("Inference Latency per Pair (ms, log scale)", fontsize=11)
    plt.title("Inference Latency Comparison (Log Scale)", fontsize=13, weight="bold")
    plt.gca().invert_yaxis()

    for i, v in enumerate(lat_values):
        plt.text(v * 1.3, i, f"{v:.4f} ms" if v < 1 else f"{v:.2f} ms", va="center", fontsize=10, weight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig11_latency_comparison.png"), dpi=300)
    plt.close()

    print("[+] All comparison figures generated successfully in plots/ directory.")

    # Save summary JSON for tables in Report
    comparison_summary = {
        "Table8_Performance": {
            "Accuracy": {"Baseline": f"{base_metrics['accuracy']*100:.2f}%", "Transformer": f"{trans_metrics['accuracy']*100:.2f}%", "Diff": f"+{(trans_metrics['accuracy']-base_metrics['accuracy'])*100:.2f}%"},
            "Macro Precision": {"Baseline": f"{base_metrics['macro_precision']*100:.2f}%", "Transformer": f"{trans_metrics['macro_precision']*100:.2f}%", "Diff": f"+{(trans_metrics['macro_precision']-base_metrics['macro_precision'])*100:.2f}%"},
            "Macro Recall": {"Baseline": f"{base_metrics['macro_recall']*100:.2f}%", "Transformer": f"{trans_metrics['macro_recall']*100:.2f}%", "Diff": f"+{(trans_metrics['macro_recall']-base_metrics['macro_recall'])*100:.2f}%"},
            "Macro F1-score": {"Baseline": f"{base_metrics['macro_f1']*100:.2f}%", "Transformer": f"{trans_metrics['macro_f1']*100:.2f}%", "Diff": f"+{(trans_metrics['macro_f1']-base_metrics['macro_f1'])*100:.2f}%"},
            "Weighted F1-score": {"Baseline": f"{base_metrics['weighted_f1']*100:.2f}%", "Transformer": f"{trans_metrics['weighted_f1']*100:.2f}%", "Diff": f"+{(trans_metrics['weighted_f1']-base_metrics['weighted_f1'])*100:.2f}%"},
            "Hallucinated F1": {"Baseline": f"{rep_b['0']['f1-score']*100:.2f}%", "Transformer": f"{rep_t['0']['f1-score']*100:.2f}%", "Diff": f"+{(rep_t['0']['f1-score']-rep_b['0']['f1-score'])*100:.2f}%"},
            "Factual F1": {"Baseline": f"{rep_b['1']['f1-score']*100:.2f}%", "Transformer": f"{rep_t['1']['f1-score']*100:.2f}%", "Diff": f"+{(rep_t['1']['f1-score']-rep_b['1']['f1-score'])*100:.2f}%"},
            "Training Time": {"Baseline": f"{base_metrics['training_time_seconds']:.2f} s", "Transformer": "Pre-trained checkpoint", "Diff": "-"},
            "Inference Latency": {"Baseline": f"{base_metrics['inference_latency_ms_per_sample']:.4f} ms/pair", "Transformer": f"{trans_metrics['inference_latency_ms_per_sample']:.2f} ms/pair", "Diff": f"≈ {int(trans_metrics['inference_latency_ms_per_sample']/max(0.0001, base_metrics['inference_latency_ms_per_sample']))}x slower"}
        },
        "Table9_ConfusionMatrix": {
            "TF-IDF + LR": {"TN": int(cm_base[0][0]), "FP": int(cm_base[0][1]), "FN": int(cm_base[1][0]), "TP": int(cm_base[1][1])},
            "DistilRoBERTa": {"TN": int(cm_trans[0][0]), "FP": int(cm_trans[0][1]), "FN": int(cm_trans[1][0]), "TP": int(cm_trans[1][1])}
        },
        "Agreement_Stats": {
            "Both Correct": int(both_correct),
            "Only Transformer": int(only_trans_correct),
            "Only Baseline": int(only_base_correct),
            "Both Wrong": int(both_wrong),
            "Total": int(total_test)
        }
    }

    with open(os.path.join(RESULTS_DIR, "comparison_summary.json"), "w") as f:
        json.dump(comparison_summary, f, indent=4)

    return comparison_summary


if __name__ == "__main__":
    run_comprehensive_evaluation()
