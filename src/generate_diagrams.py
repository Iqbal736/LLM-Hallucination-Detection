"""
Module: src/generate_diagrams.py
Description: Generates Figure 1 (Overall System Workflow), Figure 2 (Baseline Lexical Pipeline),
             and Figure 3 (Transformer Cross-Encoder Architecture) matching the sample report style.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def draw_fig1_workflow():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')

    def draw_box(x, y, w, h, text, bg="#fef3c7", border="#d97706", fontsize=9):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                      facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, weight="bold", color="#1f2937")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#4b5563', edgecolor='#4b5563', width=1.5, headwidth=8))

    # Top: Dataset
    draw_box(3.5, 8.5, 3.0, 0.8, "Hallucination Benchmark Dataset\n(5,000 Verified Pairs)", bg="#fef08a", border="#ca8a04")
    draw_box(3.2, 7.0, 3.6, 0.8, "Stratified OOD Partitioning\nTrain: 4,000 | Val: 500 | Test: 500", bg="#e0e7ff", border="#4f46e5")
    draw_arrow(5.0, 8.5, 5.0, 7.8)

    # Branch left: Baseline
    draw_arrow(4.2, 7.0, 2.5, 5.8)
    ax.text(2.6, 6.4, "Model 1: Lexical ML", fontsize=10, style="italic", weight="bold")
    draw_box(1.0, 5.0, 3.0, 0.8, "Text Normalization &\nContraction Expansion", bg="#e0f2fe", border="#0284c7")
    draw_arrow(2.5, 5.0, 2.5, 4.3)
    draw_box(1.0, 3.5, 3.0, 0.8, "Dense Overlap Extraction\n(Jaccard, N-grams, Negation)", bg="#e0f2fe", border="#0284c7")
    draw_arrow(2.5, 3.5, 2.5, 2.8)
    draw_box(1.0, 2.0, 3.0, 0.8, "Logistic Regression Classifier\n(C tuned on validation = 0.01)", bg="#dcfce7", border="#16a34a")

    # Branch right: Transformer
    draw_arrow(5.8, 7.0, 7.5, 5.8)
    ax.text(6.8, 6.4, "Model 2: Transformer", fontsize=10, style="italic", weight="bold")
    draw_box(6.0, 5.0, 3.0, 0.8, "Sentence-Pair Tokenization\n[CLS] Context [SEP] Claim [SEP]", bg="#f3e8ff", border="#9333ea")
    draw_arrow(7.5, 5.0, 7.5, 4.3)
    draw_box(6.0, 3.5, 3.0, 0.8, "DistilRoBERTa Cross-Encoder\n(6 layers, 12 heads, 82M params)", bg="#f3e8ff", border="#9333ea")
    draw_arrow(7.5, 3.5, 7.5, 2.8)
    draw_box(6.0, 2.0, 3.0, 0.8, "NLI Classification Head\n(Softmax: Contradiction / Entailment)", bg="#dcfce7", border="#16a34a")

    # Bottom merge
    draw_arrow(2.5, 2.0, 4.2, 0.8)
    draw_arrow(7.5, 2.0, 5.8, 0.8)
    draw_box(3.0, 0.1, 4.0, 0.7, "Empirical Evaluation, Error Analysis\n& Streamlit Web Deployment", bg="#ffe4e6", border="#e11d48")

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.2, 9.8)
    plt.title("Figure 1: Overall workflow of the factual consistency detection system", fontsize=12, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_workflow.png"), dpi=300)
    plt.close()


def draw_fig2_baseline_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.axis('off')

    steps = [
        ("Raw Text Pair\n(Context, Claim)", "#fef3c7", "#d97706"),
        ("clean_text()\nNormalisation", "#e0e7ff", "#4f46e5"),
        ("Dense Lexical\nFeature Vector (8-d)", "#e0f2fe", "#0284c7"),
        ("StandardScaler()\nZ-Score Normalise", "#f3e8ff", "#9333ea"),
        ("Logistic Regression\nσ(w·x + b)", "#dcfce7", "#16a34a"),
        ("Factual / Hallu\nVerdict + Conf", "#ffe4e6", "#e11d48")
    ]

    w, h = 1.4, 1.2
    for i, (text, bg, border) in enumerate(steps):
        x = i * 1.8 + 0.2
        y = 0.6
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.5, weight="bold", color="#1f2937")

        if i < len(steps) - 1:
            ax.annotate('', xy=(x + w + 0.35, y + h/2), xytext=(x + w, y + h/2),
                        arrowprops=dict(facecolor='#4b5563', edgecolor='#4b5563', width=1.2, headwidth=6))

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.5)
    plt.title("Figure 2: Lexical Overlap + Logistic Regression baseline pipeline", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_baseline_pipeline.png"), dpi=300)
    plt.close()


def draw_fig3_transformer_architecture():
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.axis('off')

    def draw_layer(y, text, bg="#f3e8ff", border="#9333ea", h=0.6, w=6.0):
        x = 1.0
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9, weight="bold", color="#1f2937")

    def draw_up_arrow(y_start, y_end):
        ax.annotate('', xy=(4.0, y_end), xytext=(4.0, y_start),
                    arrowprops=dict(facecolor='#6b7280', edgecolor='#6b7280', width=1.5, headwidth=7))

    # Layers bottom to top
    draw_layer(0.4, "Input Pair: Context (Premise) and Generation (Hypothesis)", bg="#fef08a", border="#ca8a04")
    draw_up_arrow(1.1, 1.4)

    draw_layer(1.4, "RoBERTa BPE Tokenizer → [CLS] Context [SEP] Claim [SEP]", bg="#e0e7ff", border="#4f46e5")
    draw_up_arrow(2.1, 2.4)

    draw_layer(2.4, "Token Embeddings + Positional Embeddings (768-d)", bg="#e0f2fe", border="#0284c7")
    draw_up_arrow(3.1, 3.4)

    # Encoder block
    rect_box = patches.Rectangle((0.6, 3.4), 6.8, 2.6, fill=False, edgecolor="#9333ea", linestyle="--", linewidth=1.5)
    ax.add_patch(rect_box)
    ax.text(7.2, 5.8, "× 6 Transformer Blocks", fontsize=9, weight="bold", color="#9333ea", rotation=270)

    draw_layer(3.6, "Multi-Head Cross-Self-Attention (12 heads, 768-d)", bg="#ffffff", border="#9333ea", h=0.5, w=5.6)
    draw_layer(4.4, "Add & Layer Normalization + Dropout (0.1)", bg="#ffffff", border="#9333ea", h=0.4, w=5.6)
    draw_layer(5.1, "Feed-Forward Network (768 → 3072 → 768, GELU)", bg="#ffffff", border="#9333ea", h=0.5, w=5.6)
    draw_up_arrow(6.1, 6.4)

    draw_layer(6.4, "[CLS] Contextualized Representation (768-d) → Pooler Dense Layer", bg="#e0f2fe", border="#0284c7")
    draw_up_arrow(7.1, 7.4)

    draw_layer(7.4, "Classification Head + Softmax: P(Contradiction), P(Entailment), P(Neutral)", bg="#dcfce7", border="#16a34a")

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8.5)
    plt.title("Figure 3: Layer-wise architecture of DistilRoBERTa Cross-Encoder", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_transformer_architecture.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    print("[*] Generating Figure 1, Figure 2, and Figure 3 diagrams...")
    draw_fig1_workflow()
    draw_fig2_baseline_pipeline()
    draw_fig3_transformer_architecture()
    print("[+] All architectural diagrams saved to plots/ directory.")

