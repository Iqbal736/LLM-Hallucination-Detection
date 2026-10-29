"""
Module: src/predict.py
Description: Unified inference pipeline for both Baseline (Lexical Overlap + Logistic Regression)
             and Transformer Cross-Encoder. Exposes predict_single(context, generation)
             for real-time Streamlit front-end integration.
"""

import os
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing import clean_text, extract_pair_features, compute_lexical_overlap

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_NAME = "cross-encoder/nli-distilroberta-base"


class HallucinationPredictor:
    """
    Unified predictor encapsulating both Classical ML (Lexical + LR)
    and Deep Contextual Transformer (DistilRoBERTa Cross-Encoder).
    """

    def __init__(self, use_gpu: bool = True):
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        print(f"[*] Loading HallucinationPredictor on {self.device}...")

        # 1. Load Baseline Pipeline
        pipeline_path = os.path.join(MODELS_DIR, "baseline_pipeline.joblib")
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Baseline pipeline not found at {pipeline_path}. Run baseline_model.py first.")
        self.baseline_pipeline = joblib.load(pipeline_path)

        # 2. Load Transformer Checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_NAME)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_NAME).to(self.device)
        self.transformer.eval()
        print("[+] Both models loaded into memory successfully.")

    def predict_single(self, context: str, generation: str) -> dict:
        """
        Runs inference on a single (context, generation) pair across both models.
        Returns labels, confidences, class probabilities, and an agreement flag.
        """
        # Lexical feature extraction
        lex_stats = compute_lexical_overlap(context, generation)
        feats = extract_pair_features(context, generation).reshape(1, -1)

        # --- Model 1: Lexical Overlap + Logistic Regression ---
        base_probs = self.baseline_pipeline.predict_proba(feats)[0]
        # base_probs: [P(label=0/Hallu), P(label=1/Factual)]
        p_base_hallu = float(base_probs[0])
        p_base_fact = float(base_probs[1])
        base_label = 1 if p_base_fact >= 0.5 else 0
        base_confidence = max(p_base_fact, p_base_hallu)

        # --- Model 2: Transformer Cross-Encoder ---
        inputs = self.tokenizer(
            context,
            generation,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.transformer(**inputs)
            trans_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # id2label: 0=contradiction, 1=entailment, 2=neutral
        p_contra = float(trans_probs[0])
        p_entail = float(trans_probs[1])
        p_neutral = float(trans_probs[2])

        # Factual if entailment dominates contradiction
        trans_label = 1 if p_entail >= p_contra else 0
        p_trans_fact = p_entail
        p_trans_hallu = p_contra + p_neutral
        trans_confidence = max(p_trans_fact, p_trans_hallu)

        agreement = (base_label == trans_label)

        if agreement:
            verdict = "Consensus: Factual & Grounded" if base_label == 1 else "Consensus: Hallucination Detected"
        else:
            verdict = "Model Disagreement (Linguistic Edge Case)"

        return {
            "agreement": agreement,
            "verdict": verdict,
            "lexical_overlap": lex_stats,
            "baseline": {
                "label": "Factual / Grounded" if base_label == 1 else "Hallucinated",
                "label_code": base_label,
                "confidence": round(base_confidence * 100, 2),
                "prob_factual": round(p_base_fact, 4),
                "prob_hallucinated": round(p_base_hallu, 4)
            },
            "transformer": {
                "label": "Factual / Grounded" if trans_label == 1 else "Hallucinated",
                "label_code": trans_label,
                "confidence": round(trans_confidence * 100, 2),
                "prob_factual": round(p_trans_fact, 4),
                "prob_hallucinated": round(p_trans_hallu, 4),
                "nli_breakdown": {
                    "entailment": round(p_entail * 100, 2),
                    "contradiction": round(p_contra * 100, 2),
                    "neutral": round(p_neutral * 100, 2)
                }
            }
        }
