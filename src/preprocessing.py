"""
Module: src/preprocessing.py
Description: Feature engineering for factual consistency and hallucination detection.
             Extracts dense lexical similarity, n-gram overlap, and negation mismatch features
             alongside text normalization.
"""

import re
import numpy as np

NEGATION_TOKENS = {"not", "never", "no", "neither", "nor", "barely", "hardly", "failed", "zero", "without", "disproved"}


def clean_text(text: str) -> str:
    """Standardizes text, expands contractions, cleans whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Contractions
    text = re.sub(r"\bcan\'t\b", "can not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwon\'t\b", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bn\'t\b", " not", text, flags=re.IGNORECASE)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def get_ngrams(tokens: list, n: int) -> set:
    """Returns a set of n-gram tuples."""
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def extract_pair_features(context: str, generation: str) -> np.ndarray:
    """
    Extracts an 8-dimensional domain-agnostic factual consistency feature vector:
    1. Unigram Jaccard Similarity
    2. Unigram Precision (Claim coverage in Context)
    3. Unigram Recall (Context coverage in Claim)
    4. Bigram Jaccard Similarity
    5. Bigram Precision
    6. Negation Mismatch Flag
    7. Word Count Length Ratio
    8. Token Overlap Absolute Count
    """
    c_clean = clean_text(context)
    g_clean = clean_text(generation)

    c_tokens = re.findall(r"\b\w+\b", c_clean)
    g_tokens = re.findall(r"\b\w+\b", g_clean)

    c_set = set(c_tokens)
    g_set = set(g_tokens)

    # 1. Unigram Overlap
    inter_uni = c_set.intersection(g_set)
    union_uni = c_set.union(g_set)
    jaccard_uni = len(inter_uni) / max(1, len(union_uni))
    prec_uni = len(inter_uni) / max(1, len(g_set))
    rec_uni = len(inter_uni) / max(1, len(c_set))

    # 2. Bigram Overlap
    c_bi = get_ngrams(c_tokens, 2)
    g_bi = get_ngrams(g_tokens, 2)
    inter_bi = c_bi.intersection(g_bi)
    union_bi = c_bi.union(g_bi)
    jaccard_bi = len(inter_bi) / max(1, len(union_bi))
    prec_bi = len(inter_bi) / max(1, len(g_bi))

    # 3. Negation Discrepancy
    c_neg = bool(c_set.intersection(NEGATION_TOKENS))
    g_neg = bool(g_set.intersection(NEGATION_TOKENS))
    neg_mismatch = 1.0 if (c_neg != g_neg) else 0.0

    # 4. Length Ratios
    len_ratio = len(g_tokens) / max(1, len(c_tokens))
    abs_overlap = float(len(inter_uni))

    return np.array([
        jaccard_uni,
        prec_uni,
        rec_uni,
        jaccard_bi,
        prec_bi,
        neg_mismatch,
        len_ratio,
        abs_overlap
    ], dtype=np.float32)


def compute_lexical_overlap(context: str, generation: str) -> dict:
    """Helper for user-facing Streamlit app display."""
    feats = extract_pair_features(context, generation)
    return {
        "jaccard_similarity": round(float(feats[0]), 4),
        "claim_token_coverage": round(float(feats[1] * 100), 2),
        "bigram_precision": round(float(feats[4] * 100), 2),
        "negation_conflict": bool(feats[5] == 1.0)
    }
