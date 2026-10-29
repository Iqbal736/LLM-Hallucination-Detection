"""
Module: tests/test_cases.py
Description: Probes both models on hand-crafted rubric test cases targeting subtle
             linguistic phenomena:
             - TC-1: Clearly factual paraphrase
             - TC-2: Direct contradiction / Polarity reversal
             - TC-3: High-overlap entity swap
             - TC-4: Numerical / Date fabrication
             Generates Table 10 for the Valuation Report.
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import HallucinationPredictor

RESULTS_DIR = os.path.join(BASE_DIR, "results")


def run_rubric_tests():
    print("[*] Loading HallucinationPredictor to evaluate Rubric Test Cases...")
    predictor = HallucinationPredictor()

    test_cases = [
        {
            "id": "TC-1",
            "category": "Clearly factual paraphrase",
            "context": "On September 14, 2015, the LIGO observatory made the first direct physical detection of gravitational waves produced by merging black holes.",
            "generation": "LIGO detected physical ripples in spacetime from colliding black holes in September 2015.",
            "expected": "Factual / Grounded"
        },
        {
            "id": "TC-2",
            "category": "Direct contradiction / Polarity flip",
            "context": "Metformin decreases hepatic glucose production and increases insulin sensitivity in peripheral tissues.",
            "generation": "Metformin significantly increases hepatic glucose production and induces severe insulin resistance.",
            "expected": "Hallucinated"
        },
        {
            "id": "TC-3",
            "category": "High-overlap entity swap",
            "context": "The Apollo 11 lunar module Eagle touched down on the Moon on July 20, 1969, carrying Neil Armstrong and Buzz Aldrin.",
            "generation": "The Apollo 11 lunar module Eagle touched down on the Moon on July 20, 1969, carrying Neil Armstrong and Michael Collins.",
            "expected": "Hallucinated"
        },
        {
            "id": "TC-4",
            "category": "Numerical / Temporal fabrication",
            "context": "Base editing was developed by David Liu's laboratory in 2016 without requiring double-stranded DNA breaks.",
            "generation": "Base editing was developed by David Liu's laboratory in 1985 without requiring double-stranded DNA breaks.",
            "expected": "Hallucinated"
        }
    ]

    results = []
    print("\n" + "="*80)
    print("TABLE 10: RUBRIC TEST CASES AND MODEL PREDICTIONS")
    print("="*80)

    for tc in test_cases:
        pred = predictor.predict_single(tc["context"], tc["generation"])
        b_res = pred["baseline"]
        t_res = pred["transformer"]

        row = {
            "Case": tc["id"],
            "Category": tc["category"],
            "Context": tc["context"],
            "Generation": tc["generation"],
            "Expected": tc["expected"],
            "Baseline_Label": b_res["label"],
            "Baseline_Conf": f"{b_res['confidence']:.1f}%",
            "Transformer_Label": t_res["label"],
            "Transformer_Conf": f"{t_res['confidence']:.1f}%",
            "Agreement": pred["agreement"],
            "Lexical_Coverage": f"{pred['lexical_overlap']['claim_token_coverage']}%"
        }
        results.append(row)

        print(f"[{tc['id']}] {tc['category']}")
        print(f"    - Expected:     {tc['expected']}")
        print(f"    - Baseline:     {b_res['label']} ({b_res['confidence']:.1f}%)")
        print(f"    - Transformer:  {t_res['label']} ({t_res['confidence']:.1f}%)")
        print(f"    - Agreement:    {pred['agreement']} | Word Coverage: {pred['lexical_overlap']['claim_token_coverage']}%")
        print("-" * 80)

    with open(os.path.join(RESULTS_DIR, "rubric_test_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    run_rubric_tests()

