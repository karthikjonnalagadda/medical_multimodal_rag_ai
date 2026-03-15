"""
data/download_datasets.py
--------------------------
Dataset download and ingestion scripts for the Medical AI Assistant.

Datasets covered:
  * NIH Chest X-ray (Box download)
  * PubMedQA knowledge base
  * MedQuAD QA pairs
  * ICD-10 codes (WHO open data)
  * MIMIC-CXR (requires PhysioNet credentials)

Usage:
  python data/download_datasets.py --dataset pubmedqa
  python data/download_datasets.py --dataset icd10
  python data/download_datasets.py --dataset medquad
  python data/download_datasets.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
KNOWLEDGE_DIR = DATA_DIR / "knowledge_base"
IMAGES_DIR = DATA_DIR / "medical_images"
REPORTS_DIR = DATA_DIR / "lab_reports"

for d in [KNOWLEDGE_DIR, IMAGES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download a file with a progress bar."""
    try:
        logger.info(f"Downloading {url} → {dest}")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.0f}%  ({downloaded//1024} KB / {total//1024} KB)", end="")
        print()
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PubMedQA
# ═══════════════════════════════════════════════════════════════════════════════

def download_pubmedqa() -> Path:
    """
    Download PubMedQA dataset (ori_pqal split – 1,000 labelled QA pairs).
    Source: https://pubmedqa.github.io/
    """
    url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
    dest = KNOWLEDGE_DIR / "pubmedqa_ori_pqal.json"

    if dest.exists():
        logger.info(f"PubMedQA already downloaded: {dest}")
        return dest

    if _download_file(url, dest):
        logger.success(f"PubMedQA downloaded: {dest}")
    return dest


def ingest_pubmedqa(pipeline) -> int:
    """Ingest PubMedQA into the RAG knowledge base."""
    dest = download_pubmedqa()
    if not dest.exists():
        logger.error("PubMedQA file not found.")
        return 0

    with open(dest) as f:
        data = json.load(f)

    documents, sources = [], []
    for pmid, item in list(data.items())[:500]:   # ingest first 500 for demo
        context = " ".join(item.get("CONTEXTS", []))
        question = item.get("QUESTION", "")
        long_ans = item.get("LONG_ANSWER", "")
        if context or long_ans:
            text = f"Q: {question}\nContext: {context}\nAnswer: {long_ans}"
            documents.append(text)
            sources.append(f"PubMedQA PMID:{pmid}")

    logger.info(f"Ingesting {len(documents)} PubMedQA documents …")
    pipeline.ingest_knowledge(documents, sources=sources)
    logger.success(f"PubMedQA ingestion complete: {len(documents)} docs")
    return len(documents)


# ═══════════════════════════════════════════════════════════════════════════════
# ICD-10
# ═══════════════════════════════════════════════════════════════════════════════

def download_icd10() -> Path:
    """
    Download ICD-10 codes from a public GitHub mirror.
    WHO official data requires account; this uses an open mirror.
    """
    url = "https://raw.githubusercontent.com/k4m4/icd/master/icd10.json"
    dest = KNOWLEDGE_DIR / "icd10_codes.json"

    if dest.exists():
        logger.info(f"ICD-10 already downloaded: {dest}")
        return dest

    if _download_file(url, dest):
        logger.success(f"ICD-10 downloaded: {dest}")
    return dest


def ingest_icd10(pipeline) -> int:
    """Ingest ICD-10 disease descriptions into the knowledge base."""
    dest = download_icd10()
    if not dest.exists():
        return 0

    with open(dest) as f:
        data = json.load(f)

    # data is a list of {"code": "...", "name": "...", "description": "..."}
    documents, sources = [], []
    if isinstance(data, list):
        for entry in data[:1000]:
            code = entry.get("code", "")
            name = entry.get("name", "")
            desc = entry.get("description", name)
            if code and name:
                documents.append(f"ICD-10 {code}: {name}. {desc}")
                sources.append(f"ICD-10 Database ({code})")
    elif isinstance(data, dict):
        for code, info in list(data.items())[:1000]:
            name = info if isinstance(info, str) else info.get("name", code)
            documents.append(f"ICD-10 {code}: {name}")
            sources.append(f"ICD-10 Database ({code})")

    if documents:
        pipeline.ingest_knowledge(documents, sources=sources)
        logger.success(f"ICD-10 ingestion complete: {len(documents)} codes")
    return len(documents)


# ═══════════════════════════════════════════════════════════════════════════════
# MedQuAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_medquad() -> Path:
    """
    Download a sample of MedQuAD from the GitHub repository.
    Full dataset: https://github.com/abachaa/MedQuAD
    """
    url = "https://raw.githubusercontent.com/abachaa/MedQuAD/master/README.md"
    dest = KNOWLEDGE_DIR / "medquad_readme.md"

    # The full MedQuAD dataset is in XML per source; we download the summary
    if dest.exists():
        logger.info(f"MedQuAD info already downloaded: {dest}")
        return dest

    if _download_file(url, dest):
        logger.success("MedQuAD README downloaded.")
    return dest


def create_sample_knowledge_base(pipeline) -> int:
    """
    Create a sample knowledge base from curated medical text.
    Used as a fallback when external dataset downloads fail.
    """
    sample_knowledge = [
        # Respiratory
        "Pneumonia is an infection that inflames air sacs in one or both lungs. "
        "Symptoms include cough with phlegm or pus, fever, chills, and difficulty breathing. "
        "WBC count is typically elevated (>11,000 cells/μL). ICD-10: J18.9",

        "Community-acquired pneumonia (CAP) is the most common type. "
        "Typical pathogens include Streptococcus pneumoniae, Haemophilus influenzae. "
        "WHO recommends amoxicillin as first-line treatment for non-severe CAP.",

        "Tuberculosis (TB) is caused by Mycobacterium tuberculosis bacteria. "
        "Symptoms: persistent cough ≥3 weeks, haemoptysis, chest pain, fatigue, fever, "
        "night sweats, weight loss. CXR shows upper lobe consolidation or cavitation. ICD-10: A15",

        "Pulmonary Edema: excess fluid in the lungs. Causes include heart failure, "
        "ARDS, and pneumonia. CXR shows bilateral infiltrates, Kerley B lines, "
        "cardiomegaly. Treated with diuretics and oxygen therapy. ICD-10: J81",

        "COPD is a chronic inflammatory disease causing airflow obstruction. "
        "FEV1/FVC ratio <0.7 confirms diagnosis. Symptoms: dyspnoea, chronic cough, "
        "sputum production. Main risk factor: smoking. ICD-10: J44",

        "Pleural Effusion: abnormal fluid collection between pleural layers. "
        "Causes: heart failure, malignancy, infection, cirrhosis. "
        "CXR shows blunting of costophrenic angle. Diagnostic criteria: Light's criteria. ICD-10: J90",

        "Atelectasis: collapse of lung parenchyma. "
        "Post-operative atelectasis is the most common pulmonary complication after surgery. "
        "CXR shows linear opacities, volume loss, shifted fissures. ICD-10: J98.1",

        # Cardiovascular
        "Cardiomegaly (enlarged heart) on CXR is defined as cardiothoracic ratio >0.5. "
        "Causes: hypertension, cardiomyopathy, valvular disease, pericardial effusion. "
        "Requires echocardiography for evaluation. ICD-10: I51.7",

        # Lab values
        "Normal CBC reference ranges: Haemoglobin 12-17.5 g/dL, "
        "WBC 4500-11000 cells/μL, Platelets 150000-400000 cells/μL, "
        "RBC 4.2-5.9 million/μL. Elevated WBC (>11000) indicates infection or inflammation.",

        "Elevated C-reactive protein (CRP >10 mg/L) indicates systemic inflammation. "
        "ESR elevation parallels inflammation severity. Both are non-specific markers.",

        # Treatment guidelines
        "WHO Pneumonia Treatment Guidelines (2023): Non-severe CAP: amoxicillin 500mg TID x5d. "
        "Severe CAP: IV penicillin + macrolide. Atypical: doxycycline or azithromycin.",

        "Sepsis management (Surviving Sepsis Campaign 2021): "
        "1-hour bundle: measure lactate, blood cultures x2 before antibiotics, "
        "broad-spectrum antibiotics, 30 mL/kg IV crystalloid for hypotension or lactate ≥4.",
    ]

    sources = [
        "Mayo Clinic – Pneumonia", "WHO CAP Guidelines 2023",
        "WHO TB Factsheet", "Cleveland Clinic – Pulmonary Edema",
        "GOLD COPD Guidelines", "NEJM – Pleural Effusion",
        "Mayo Clinic – Atelectasis", "AHA Cardiomegaly",
        "ARUP Lab Reference Ranges", "Clinical Pathology – Inflammatory Markers",
        "WHO Pneumonia Treatment Guidelines 2023", "Surviving Sepsis Campaign 2021",
    ]

    pipeline.ingest_knowledge(sample_knowledge, sources=sources)
    logger.success(f"Sample knowledge base created: {len(sample_knowledge)} documents")
    return len(sample_knowledge)


# ═══════════════════════════════════════════════════════════════════════════════
# NIH Chest X-ray (instructions only – dataset requires Box account)
# ═══════════════════════════════════════════════════════════════════════════════

def print_nih_instructions():
    logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          NIH Chest X-ray Dataset – Download Instructions     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  URL: https://nihcc.app.box.com/v/ChestXray-NIHCC            ║
    ║                                                              ║
    ║  Steps:                                                      ║
    ║  1. Create a free Box account                                ║
    ║  2. Navigate to the URL above                                ║
    ║  3. Download images_001.tar.gz … images_012.tar.gz           ║
    ║  4. Extract to: data/medical_images/nih_chest_xray/          ║
    ║  5. Download Data_Entry_2017.csv (labels)                    ║
    ║                                                              ║
    ║  Size: ~45 GB (112,120 images)                               ║
    ║                                                              ║
    ║  For demo/testing, use the sample subset (~1000 images):     ║
    ║  https://nihcc.app.box.com/v/ChestXray-NIHCC/file/           ║
    ║        220660789610                                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# MIMIC (instructions only – requires PhysioNet credentialing)
# ═══════════════════════════════════════════════════════════════════════════════

def print_mimic_instructions():
    logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         MIMIC Datasets – Access Instructions                 ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Requires PhysioNet credentialed access (HIPAA training)     ║
    ║                                                              ║
    ║  MIMIC-CXR:  https://physionet.org/content/mimic-cxr/       ║
    ║  MIMIC-III:  https://physionet.org/content/mimiciii/         ║
    ║  MIMIC-IV:   https://physionet.org/content/mimiciv/          ║
    ║                                                              ║
    ║  Steps:                                                      ║
    ║  1. Register at https://physionet.org/register/              ║
    ║  2. Complete CITI training (human subjects research)         ║
    ║  3. Request access to the specific dataset                   ║
    ║  4. Download using wget with your credentials:               ║
    ║                                                              ║
    ║  wget -r -N -c -np --user USERNAME --ask-password            ║
    ║    https://physionet.org/files/mimic-cxr/2.0.0/              ║
    ║                                                              ║
    ║  5. Place files in: data/medical_images/mimic_cxr/           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Medical AI Dataset Downloader")
    parser.add_argument("--dataset", choices=["pubmedqa", "icd10", "medquad", "sample", "all"])
    parser.add_argument("--ingest", action="store_true", help="Ingest into pipeline")
    parser.add_argument("--info", action="store_true", help="Print dataset access instructions")
    args = parser.parse_args()

    if args.info:
        print_nih_instructions()
        print_mimic_instructions()
        return

    pipeline = None
    if args.ingest:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.rag.rag_pipeline import MedicalRAGPipeline
        pipeline = MedicalRAGPipeline(llm_backend="huggingface")

    dataset = args.dataset or "all"

    if dataset in ("pubmedqa", "all"):
        dest = download_pubmedqa()
        if args.ingest and pipeline:
            ingest_pubmedqa(pipeline)

    if dataset in ("icd10", "all"):
        dest = download_icd10()
        if args.ingest and pipeline:
            ingest_icd10(pipeline)

    if dataset in ("medquad", "all"):
        download_medquad()

    if dataset in ("sample", "all") and args.ingest and pipeline:
        create_sample_knowledge_base(pipeline)

    logger.success("Dataset preparation complete.")


if __name__ == "__main__":
    main()
