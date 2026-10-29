"""
Streamlit Web Application: Real-Time LLM Hallucination & Factual Consistency Detection
Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
from PIL import Image

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

sys.path.append(SRC_DIR)
from predict import HallucinationPredictor

st.set_page_config(
    page_title="LLM Hallucination & Factuality Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        padding: 1.2rem;
        border-radius: 0.6rem;
        border: 1px solid #E5E7EB;
        background-color: #F9FAFB;
        margin-bottom: 1rem;
    }
    .badge-agree {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: 700;
    }
    .badge-disagree {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading NLP and Transformer Models into Memory...")
def get_predictor():
    return HallucinationPredictor(use_gpu=True)


# Load summary
comparison_data = {}
summary_path = os.path.join(RESULTS_DIR, "comparison_summary.json")
if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        comparison_data = json.load(f)

# Sidebar Navigation
st.sidebar.title("Navigation & Details")
view_mode = st.sidebar.radio("Select View:", ["🔍 Live Hallucination Detector", "📊 Model Comparison & Benchmarks"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Course:** Natural Language Processing for LLM (24AM7PCLLM)")
st.sidebar.markdown("**Institution:** B.M.S. College of Engineering")
st.sidebar.markdown("**Department:** Machine Learning")
st.sidebar.markdown("**Faculty In-charge:** Dr. Monika Puttaramaiah")

# ---------------- VIEW 1: LIVE PREDICTOR ----------------
if view_mode == "🔍 Live Hallucination Detector":
    st.markdown('<div class="main-title">Real-Time LLM Hallucination & Factuality Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluate whether an LLM generation is factually consistent with the ground-truth context using Lexical ML vs. DistilRoBERTa Cross-Encoder.</div>', unsafe_allow_html=True)

    predictor = get_predictor()

    # Preset examples
    preset = st.selectbox(
        "Choose a preset test scenario or input custom texts below:",
        [
            "Custom Input",
            "Example 1: Factual Grounding (LIGO Gravitational Waves)",
            "Example 2: Contradictory Polarity Inversion (Metformin)",
            "Example 3: High-Overlap Entity Swap (Apollo 11 Crew)",
            "Example 4: Temporal / Date Fabrication (Base Editing)"
        ]
    )

    preset_texts = {
        "Example 1: Factual Grounding (LIGO Gravitational Waves)": (
            "On September 14, 2015, the LIGO observatory made the first direct physical detection of gravitational waves produced by merging black holes.",
            "LIGO detected physical ripples in spacetime from colliding black holes in September 2015."
        ),
        "Example 2: Contradictory Polarity Inversion (Metformin)": (
            "Metformin decreases hepatic glucose production and increases insulin sensitivity in peripheral tissues.",
            "Metformin significantly increases hepatic glucose production and induces severe insulin resistance."
        ),
        "Example 3: High-Overlap Entity Swap (Apollo 11 Crew)": (
            "The Apollo 11 lunar module Eagle touched down on the Moon on July 20, 1969, carrying Neil Armstrong and Buzz Aldrin.",
            "The Apollo 11 lunar module Eagle touched down on the Moon on July 20, 1969, carrying Neil Armstrong and Michael Collins."
        ),
        "Example 4: Temporal / Date Fabrication (Base Editing)": (
            "Base editing was developed by David Liu's laboratory in 2016 without requiring double-stranded DNA breaks.",
            "Base editing was developed by David Liu's laboratory in 1985 without requiring double-stranded DNA breaks."
        )
    }

    if preset != "Custom Input":
        default_ctx, default_gen = preset_texts[preset]
    else:
        default_ctx = "Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming at St. Mary's Hospital London."
        default_gen = "Alexander Fleming discovered penicillin in 1928, marking a monumental breakthrough in antibiotic therapy."

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        context_input = st.text_area("Ground-Truth Source Context (Premise):", value=default_ctx, height=140)
    with col_in2:
        generation_input = st.text_area("LLM-Generated Claim (Hypothesis):", value=default_gen, height=140)

    if st.button("🚀 Verify Factual Consistency", type="primary", use_container_width=True):
        if not context_input.strip() or not generation_input.strip():
            st.error("Please provide both context and generation texts.")
        else:
            with st.spinner("Analyzing cross-sentence semantic alignment..."):
                res = predictor.predict_single(context_input, generation_input)

            # Consensus & Agreement Header
            st.markdown("### Evaluation Summary")
            agree_col1, agree_col2 = st.columns([3, 1])
            with agree_col1:
                st.subheader(res["verdict"])
            with agree_col2:
                if res["agreement"]:
                    st.markdown('<div class="badge-agree">✓ MODELS AGREE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-disagree">⚠ MODEL DISAGREEMENT</div>', unsafe_allow_html=True)

            # Lexical overlap pills
            lex = res["lexical_overlap"]
            st.markdown(
                f"**Lexical Diagnostic Signals:** "
                f"Jaccard Token Similarity: `{lex['jaccard_similarity']}` | "
                f"Claim Word Coverage: `{lex['claim_token_coverage']}%` | "
                f"Bigram Precision: `{lex['bigram_precision']}%` | "
                f"Negation Conflict: `{lex['negation_conflict']}`"
            )

            st.markdown("---")

            # Model 1 vs Model 2 Side by Side
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### 🔹 Model 1: Lexical Overlap + Logistic Regression")
                base = res["baseline"]
                st.metric("Predicted Status", base["label"])
                st.metric("Model Confidence", f"{base['confidence']}%")
                st.progress(base["prob_factual"])
                st.caption(f"P(Factual): {base['prob_factual']*100:.2f}% | P(Hallucinated): {base['prob_hallucinated']*100:.2f}%")
                st.info("Uses unigram/bigram Jaccard overlap, lexical coverage, and negation indicators.")

            with c2:
                st.markdown("#### 🔸 Model 2: DistilRoBERTa Cross-Encoder (NLI)")
                trans = res["transformer"]
                st.metric("Predicted Status", trans["label"])
                st.metric("Model Confidence", f"{trans['confidence']}%")
                st.progress(trans["prob_factual"])
                st.caption(f"P(Entailment): {trans['nli_breakdown']['entailment']}% | P(Contradiction): {trans['nli_breakdown']['contradiction']}% | P(Neutral): {trans['nli_breakdown']['neutral']}%")
                st.success("Deep bidirectional cross-attention captures subtle semantic shifts and entity swaps.")

# ---------------- VIEW 2: BENCHMARKS & EVALUATION ----------------
else:
    st.markdown('<div class="main-title">Evaluation Benchmarks & Empirical Findings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quantitative performance metrics, confusion matrices, and latency comparisons on 500 held-out test pairs.</div>', unsafe_allow_html=True)

    if comparison_data:
        st.subheader("Table 8: Performance of Both Models on Held-Out Test Set")
        perf_df = pd.DataFrame(comparison_data["Table8_Performance"]).T
        st.table(perf_df)

        st.markdown("---")
        st.subheader("Figure 7: Confusion Matrices Comparison")
        fig7_path = os.path.join(PLOTS_DIR, "fig7_confusion_matrices.png")
        if os.path.exists(fig7_path):
            st.image(Image.open(fig7_path), use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.subheader("Figure 8: Overall Performance Metrics")
            fig8_path = os.path.join(PLOTS_DIR, "fig8_overall_metrics.png")
            if os.path.exists(fig8_path):
                st.image(Image.open(fig8_path), use_container_width=True)

        with col_b2:
            st.subheader("Figure 9: Class-wise Precision, Recall & F1")
            fig9_path = os.path.join(PLOTS_DIR, "fig9_classwise_metrics.png")
            if os.path.exists(fig9_path):
                st.image(Image.open(fig9_path), use_container_width=True)

        st.markdown("---")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader("Figure 10: Prediction Agreement")
            fig10_path = os.path.join(PLOTS_DIR, "fig10_model_agreement.png")
            if os.path.exists(fig10_path):
                st.image(Image.open(fig10_path), use_container_width=True)

        with col_c2:
            st.subheader("Figure 11: Inference Latency Comparison")
            fig11_path = os.path.join(PLOTS_DIR, "fig11_latency_comparison.png")
            if os.path.exists(fig11_path):
                st.image(Image.open(fig11_path), use_container_width=True)

    else:
        st.warning("Benchmark results not found. Run src/evaluation.py to generate figures and tables.")

