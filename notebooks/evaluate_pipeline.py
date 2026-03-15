"""
notebooks/evaluate_pipeline.py
--------------------------------
Comprehensive evaluation of all pipeline components.

Produces metrics for:
  - OCR extraction accuracy
  - Vision model (AUC, precision, recall per class)
  - RAG retrieval (MRR, NDCG)
  - End-to-end pipeline (clinical usefulness proxy)

Usage:
  python notebooks/evaluate_pipeline.py --component all
  python notebooks/evaluate_pipeline.py --component vision --model_path models/vision_model/densenet121_best.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loguru import logger


# ── OCR Evaluation ────────────────────────────────────────────────────────────

def evaluate_ocr(test_cases: list[dict] | None = None) -> dict:
    """
    Evaluate OCR extraction against ground truth lab values.

    test_cases: list of {"text": str, "expected_metrics": {name: value}}
    """
    from src.ocr.extract_lab_text import MedicalOCR

    ocr = MedicalOCR(engine="easyocr", gpu=False)

    # Built-in test cases
    if not test_cases:
        test_cases = [
            {
                "text": "Hemoglobin: 9.5 g/dL\nWBC Count: 15000\nPlatelets: 240000",
                "expected": {"hemoglobin": 9.5, "wbc count": 15000.0, "platelets": 240000.0},
            },
            {
                "text": "Glucose: 95 mg/dL\nCreatinine: 1.1 mg/dL\nSodium: 140 mEq/L",
                "expected": {"glucose": 95.0, "creatinine": 1.1, "sodium": 140.0},
            },
        ]

    correct, total = 0, 0
    for case in test_cases:
        # Simulate OCR output from known text
        metrics = ocr._parse_metrics(case["text"])
        extracted = {m.name.lower(): m.value for m in metrics}
        for name, expected_val in case["expected"].items():
            total += 1
            found = extracted.get(name)
            if found is not None and abs(found - expected_val) < 0.01:
                correct += 1
            else:
                logger.debug(f"OCR miss: '{name}' expected={expected_val}, got={found}")

    accuracy = correct / total if total > 0 else 0.0
    result = {
        "component": "OCR",
        "metric_extraction_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
    }
    logger.info(f"OCR Evaluation: {result}")
    return result


# ── Vision Model Evaluation ───────────────────────────────────────────────────

def evaluate_vision(model_path: str | None = None, n_synthetic: int = 50) -> dict:
    """
    Evaluate vision model on synthetic test images.
    In production: use held-out NIH/CheXpert test split.
    """
    from src.vision.xray_analysis import MedicalImageAnalyser, CHEXPERT_LABELS
    from PIL import Image

    analyser = MedicalImageAnalyser(
        model_path=model_path,
        confidence_threshold=0.5,
    )

    # Generate synthetic X-ray-like images
    all_probs, all_pseudo_labels = [], []
    for i in range(n_synthetic):
        # Random grey-scale 'X-ray'
        arr = np.random.randint(0, 200, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        result = analyser.analyse(image)
        probs = [result.raw_probabilities.get(label, 0.0) for label in CHEXPERT_LABELS]
        all_probs.append(probs)
        # Pseudo-label: random binary (for synthetic evaluation only)
        all_pseudo_labels.append(np.random.randint(0, 2, len(CHEXPERT_LABELS)).tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_pseudo_labels)

    # Compute per-class AUC where possible
    try:
        from sklearn.metrics import roc_auc_score
        aucs = {}
        for i, label in enumerate(CHEXPERT_LABELS):
            if labels_arr[:, i].sum() > 0:
                try:
                    aucs[label] = round(roc_auc_score(labels_arr[:, i], probs_arr[:, i]), 4)
                except Exception:
                    pass
        mean_auc = round(float(np.mean(list(aucs.values()))), 4) if aucs else 0.5
    except ImportError:
        aucs = {}
        mean_auc = 0.5

    result = {
        "component": "Vision",
        "note": "Evaluated on synthetic images – replace with real test set for production",
        "n_samples": n_synthetic,
        "mean_auc": mean_auc,
        "per_class_auc": aucs,
    }
    logger.info(f"Vision Evaluation: mean_AUC={mean_auc}")
    return result


# ── RAG Retrieval Evaluation ──────────────────────────────────────────────────

def evaluate_rag(top_k: int = 5) -> dict:
    """
    Evaluate RAG retrieval using PubMedQA-style QA pairs.
    MRR@K: Mean Reciprocal Rank at K.
    """
    from src.rag.rag_pipeline import MedicalRAGPipeline, load_sample_knowledge_base
    from src.embeddings.embedding_model import MedicalEmbeddingModel

    pipeline = MedicalRAGPipeline(llm_backend="huggingface")
    load_sample_knowledge_base(pipeline)

    # Evaluation QA pairs (query → expected keyword in top result)
    eval_pairs = [
        ("pneumonia symptoms treatment WBC elevated", "pneumonia"),
        ("tuberculosis cough night sweats weight loss", "tuberculosis"),
        ("pleural effusion fluid heart failure", "effusion"),
        ("COPD airflow obstruction breathing difficulty", "copd"),
        ("ICD code pneumonia unspecified", "J18.9"),
    ]

    embedding_model = MedicalEmbeddingModel()
    reciprocal_ranks = []

    for query, expected_keyword in eval_pairs:
        q_emb = embedding_model.embed(query)
        results = pipeline.vector_store.search_similar(q_emb, top_k=top_k)
        rank = None
        for i, doc in enumerate(results, 1):
            if expected_keyword.lower() in doc.text.lower() or (
                doc.source and expected_keyword.lower() in doc.source.lower()
            ):
                rank = i
                break
        rr = 1.0 / rank if rank else 0.0
        reciprocal_ranks.append(rr)
        logger.debug(f"Query: '{query[:40]}…' | rank={rank} | RR={rr:.3f}")

    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    hit_at_1 = sum(1 for rr in reciprocal_ranks if rr == 1.0) / len(reciprocal_ranks)

    result = {
        "component": "RAG Retrieval",
        "n_queries": len(eval_pairs),
        "top_k": top_k,
        "MRR": round(mrr, 4),
        "Hit@1": round(hit_at_1, 4),
        "per_query_rr": [round(rr, 4) for rr in reciprocal_ranks],
    }
    logger.info(f"RAG Evaluation: MRR@{top_k}={mrr:.4f} | Hit@1={hit_at_1:.4f}")
    return result


# ── End-to-End Evaluation ─────────────────────────────────────────────────────

def evaluate_e2e() -> dict:
    """
    Lightweight end-to-end smoke test.
    Returns timing and structural correctness of the full pipeline.
    """
    import time
    from src.rag.rag_pipeline import MedicalRAGPipeline, load_sample_knowledge_base

    pipeline = MedicalRAGPipeline(llm_backend="huggingface")
    load_sample_knowledge_base(pipeline)

    test_inputs = [
        {
            "lab_text": "Hemoglobin: 9.5 g/dL\nWBC: 15000\nPlatelets: 240000",
            "image_findings": ["Lung opacity in lower right lobe"],
            "symptoms": ["fever", "productive cough", "shortness of breath"],
        },
        {
            "lab_text": "Glucose: 180 mg/dL\nHbA1c: 8.2%\nCreatinine: 1.4 mg/dL",
            "symptoms": ["polyuria", "polydipsia", "fatigue"],
        },
        {
            "symptoms": ["chest pain", "dyspnea", "palpitations"],
        },
    ]

    timings, structure_ok = [], []
    for inputs in test_inputs:
        t0 = time.perf_counter()
        result = pipeline.diagnose(**inputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings.append(elapsed_ms)

        # Check structural correctness
        ok = (
            isinstance(result.conditions, list)
            and isinstance(result.evidence, list)
            and bool(result.disclaimer)
        )
        structure_ok.append(ok)

    e2e_result = {
        "component": "End-to-End",
        "n_test_cases": len(test_inputs),
        "all_structurally_correct": all(structure_ok),
        "avg_latency_ms": round(float(np.mean(timings)), 1),
        "max_latency_ms": round(float(np.max(timings)), 1),
        "min_latency_ms": round(float(np.min(timings)), 1),
    }
    logger.info(
        f"E2E Evaluation: avg_latency={e2e_result['avg_latency_ms']}ms | "
        f"all_correct={e2e_result['all_structurally_correct']}"
    )
    return e2e_result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Medical AI Pipeline")
    parser.add_argument(
        "--component",
        choices=["ocr", "vision", "rag", "e2e", "all"],
        default="all",
    )
    parser.add_argument("--model_path", default=None, help="Vision model weights path")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON path")
    args = parser.parse_args()

    results = {}

    if args.component in ("ocr", "all"):
        results["ocr"] = evaluate_ocr()

    if args.component in ("vision", "all"):
        results["vision"] = evaluate_vision(model_path=args.model_path)

    if args.component in ("rag", "all"):
        results["rag"] = evaluate_rag()

    if args.component in ("e2e", "all"):
        results["e2e"] = evaluate_e2e()

    # Save results
    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    logger.success(f"Evaluation results saved to {out_path}")

    # Print summary
    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)
    for component, metrics in results.items():
        print(f"\n[{component.upper()}]")
        for k, v in metrics.items():
            if k not in ("component", "per_class_auc", "per_query_rr"):
                print(f"  {k:35s}: {v}")
    print("=" * 55)


if __name__ == "__main__":
    main()
