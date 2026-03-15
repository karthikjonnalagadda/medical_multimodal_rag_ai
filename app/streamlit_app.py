"""
app/streamlit_app.py
--------------------
Research-grade dashboard for the multimodal medical AI assistant.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMBEDDING_CONFIG, KNOWLEDGE_BASE_DIR, VECTOR_DB_CONFIG, VISION_CONFIG, get_llm_runtime_config


st.set_page_config(
    page_title="Medical Multimodal AI",
    page_icon=":stethoscope:",
    layout="wide",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
      html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(185, 217, 235, 0.9), transparent 30%),
          radial-gradient(circle at top right, rgba(245, 223, 198, 0.8), transparent 26%),
          linear-gradient(180deg, #f4f7fb 0%, #eef3f7 52%, #f9fafb 100%);
      }
      .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(13, 38, 59, 0.96), rgba(23, 81, 117, 0.92));
        color: #f7fbff;
        box-shadow: 0 18px 50px rgba(13, 38, 59, 0.20);
        margin-bottom: 1rem;
      }
      .hero h1 { margin: 0; font-size: 2.2rem; letter-spacing: -0.03em; }
      .hero p { margin: 0.45rem 0 0; opacity: 0.92; max-width: 56rem; }
      .card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(13, 38, 59, 0.08);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 34px rgba(13, 38, 59, 0.07);
      }
      .finding {
        background: linear-gradient(180deg, #ffffff, #f7fafc);
        border: 1px solid rgba(13, 38, 59, 0.10);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
      }
      .mono { font-family: 'IBM Plex Mono', monospace; }
      .muted { color: #4f6475; }
    </style>
    """,
    unsafe_allow_html=True,
)


for key in [
    "analysis_result",
    "ocr_result",
    "vision_result",
    "evidence_results",
    "elapsed_s",
    "gradcam_result",
    "original_xray",
]:
    st.session_state.setdefault(key, None)


@st.cache_resource(show_spinner="Loading retrieval and reasoning stack...")
def load_pipeline():
    from src.embeddings.embedding_model import MedicalEmbeddingModel
    from src.rag.rag_pipeline import MedicalRAGPipeline, load_sample_knowledge_base
    from src.rag.knowledge_ingestion import ensure_medical_knowledge_base_loaded
    from src.vector_db.faiss_store import create_vector_store

    llm_config = get_llm_runtime_config()
    embedding_model = MedicalEmbeddingModel(
        backend=EMBEDDING_CONFIG["backend"],
        model_name=EMBEDDING_CONFIG["model_name"],
        batch_size=EMBEDDING_CONFIG["batch_size"],
        device=EMBEDDING_CONFIG["device"],
    )
    vector_store = create_vector_store(
        backend=VECTOR_DB_CONFIG["backend"],
        dim=embedding_model.get_embedding_dim(),
        faiss_index_path=VECTOR_DB_CONFIG["faiss_index_path"],
        chroma_persist_dir=VECTOR_DB_CONFIG["chroma_persist_dir"],
        collection_name=VECTOR_DB_CONFIG["collection_name"],
    )
    pipeline = MedicalRAGPipeline(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_backend=llm_config["backend"],
        llm_model=llm_config["model"],
        openai_api_key=llm_config["api_key"] if llm_config["backend"] == "openai" else "",
        xai_api_key=llm_config["api_key"] if llm_config["backend"] == "xai" else "",
        groq_api_key=llm_config["api_key"] if llm_config["backend"] == "groq" else "",
        xai_base_url=llm_config["base_url"] or "https://api.x.ai/v1",
        groq_base_url=llm_config["base_url"] or "https://api.groq.com/openai/v1",
    )
    ensure_medical_knowledge_base_loaded(
        knowledge_dir=KNOWLEDGE_BASE_DIR,
        persist_dir=VECTOR_DB_CONFIG["chroma_persist_dir"],
        collection_name=VECTOR_DB_CONFIG["collection_name"],
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    if pipeline.vector_store.count() == 0:
        load_sample_knowledge_base(pipeline)
    return pipeline


@st.cache_resource(show_spinner="Loading OCR engine...")
def load_ocr():
    from src.ocr.extract_lab_text import MedicalOCR
    return MedicalOCR(engine="easyocr", gpu=False)


@st.cache_resource(show_spinner="Loading medical vision backbone...")
def load_vision():
    from src.vision.xray_analysis import MedicalImageAnalyser
    return MedicalImageAnalyser(
        confidence_threshold=VISION_CONFIG["confidence_threshold"],
        backend_name=VISION_CONFIG["backend_name"],
        model_path=VISION_CONFIG["pretrained_weights"],
        model_id=VISION_CONFIG.get("hf_model_id") or None,
        device=VISION_CONFIG["device"],
        enable_gradcam=VISION_CONFIG.get("enable_gradcam", True),
    )


def render_finding_cards(result):
    for index, condition in enumerate(result.conditions[:3], start=1):
        pct = condition.confidence * 100
        st.markdown(
            f"""
            <div class="finding">
              <div class="mono muted">Differential {index}</div>
              <h4 style="margin:0.1rem 0 0.2rem;">{condition.name}</h4>
              <div><strong>{pct:.0f}%</strong> confidence</div>
              <div class="muted">{condition.icd_code or 'ICD pending mapping'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_vision_summary(result):
    if not result:
        return
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        if result.gradcam_overlay is not None:
            st.image(result.gradcam_overlay, caption=f"Grad-CAM attention for {result.top_finding.label}", use_container_width=True)
        else:
            st.info("Heatmap unavailable for the selected backend or checkpoint.")
    with right:
        st.metric("Backend", result.backend_name)
        st.metric("Primary finding", result.top_finding.label if result.top_finding else "None")
        st.metric("Image quality", result.image_quality)
        if result.differential_diagnosis:
            st.markdown("**Top-3 differential from image model**")
            for label, score in result.differential_diagnosis:
                st.write(f"- {label}: {score:.0%}")
        if result.calibration_note:
            st.caption(result.calibration_note)


def render_evidence_table(evidence_results):
    if not evidence_results:
        st.info("No retrieved evidence chunks available yet.")
        return
    for idx, item in enumerate(evidence_results, start=1):
        title = f"{idx}. {item.document.source or 'Retrieved evidence'}"
        with st.expander(title, expanded=(idx == 1)):
            st.write(item.document.text)
            st.caption(
                f"Fused={item.fused_score:.3f} | Dense={item.dense_score:.3f} | Lexical={item.lexical_score:.3f}"
            )
            if item.document.metadata:
                st.json(item.document.metadata)


st.markdown(
    """
    <div class="hero">
      <h1>Medical Multimodal AI Platform</h1>
      <p>Research-oriented assistant that combines medical vision backbones, OCR, hybrid retrieval, and LLM reasoning with explainable outputs for radiology and document-heavy workflows.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Runtime")
    confidence_threshold = st.slider("Finding threshold", 0.10, 0.95, float(VISION_CONFIG["confidence_threshold"]), 0.05)
    top_k = st.slider("Retrieved evidence", 2, 12, 6)
    st.markdown("### Stack")
    st.write(f"- Vision: `{VISION_CONFIG['backend_name']}`")
    llm_runtime = get_llm_runtime_config()
    st.write(f"- LLM: `{llm_runtime['backend']}` / `{llm_runtime['model']}`")
    st.caption("Outputs are for research use only and require clinician confirmation.")


analysis_tab, vision_tab, ocr_tab, kb_tab, guide_tab = st.tabs(
    ["Multimodal Analysis", "Vision Explainability", "OCR Review", "Knowledge Evidence", "Implementation Guide"]
)


with analysis_tab:
    input_col, output_col = st.columns([1.05, 1.15], gap="large")

    with input_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Inputs")
        lab_file = st.file_uploader("Lab report or PDF", type=["pdf", "png", "jpg", "jpeg"], key="lab_file")
        image_file = st.file_uploader("X-ray, CT, MRI, or DICOM", type=["png", "jpg", "jpeg", "dcm"], key="img_file")
        symptoms_text = st.text_area("Symptoms", placeholder="fever, productive cough, pleuritic chest pain")
        patient_notes = st.text_area("Clinical notes", placeholder="Age, history, medications, exam, prior imaging")
        run_analysis = st.button("Run multimodal analysis", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with output_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Structured output")
        if run_analysis:
            start = time.time()
            pipeline = load_pipeline()
            pipeline.top_k = top_k

            lab_text = ""
            if lab_file is not None:
                ocr = load_ocr()
                ocr_result = ocr.extract_from_bytes(lab_file.read(), filename=lab_file.name)
                st.session_state.ocr_result = ocr_result
                lab_text = ocr_result.raw_text

            image_findings = []
            if image_file is not None:
                vision = load_vision()
                vision.confidence_threshold = confidence_threshold
                image_bytes = image_file.read()
                st.session_state.original_xray = image_bytes
                vision_result = vision.analyse(image_bytes)
                st.session_state.vision_result = vision_result
                from src.vision.grad_cam import generate_gradcam

                st.session_state.gradcam_result = generate_gradcam(
                    image_bytes,
                    analyser=vision,
                    analysis_result=vision_result,
                )
                if vision_result.top_finding:
                    image_findings.append(
                        f"{vision_result.top_finding.label} ({vision_result.top_finding.confidence:.0%})"
                    )

            symptoms = [item.strip() for item in symptoms_text.split(",") if item.strip()]
            query_text = "\n".join(filter(None, [lab_text, symptoms_text, patient_notes, "; ".join(image_findings)]))
            st.session_state.evidence_results = pipeline.search_evidence(query_text, top_k=top_k)
            result = pipeline.diagnose(
                lab_text=lab_text,
                image_findings=image_findings,
                symptoms=symptoms,
                patient_notes=patient_notes,
                retrieved_docs_override=[item.document for item in st.session_state.evidence_results],
            )
            st.session_state.analysis_result = result
            st.session_state.elapsed_s = time.time() - start

        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            st.metric("Processing time", f"{(st.session_state.elapsed_s or 0):.1f}s")
            render_finding_cards(result)

            st.markdown("**Explanation**")
            st.write(result.explanation or "LLM explanation unavailable.")

            st.markdown("**Recommendation**")
            st.info(result.recommendation)

            if result.evidence:
                st.markdown("**Evidence cues**")
                for item in result.evidence:
                    st.write(f"- {item}")

            st.download_button(
                "Download report JSON",
                data=json.dumps(result.to_dict(), indent=2),
                file_name="medical_ai_report.json",
                mime="application/json",
            )
        else:
            st.caption("Run the pipeline to see the structured differential diagnosis.")
        st.markdown("</div>", unsafe_allow_html=True)


with vision_tab:
    st.subheader("Medical image explainability")
    if st.session_state.vision_result:
        render_vision_summary(st.session_state.vision_result)

        gradcam_result = st.session_state.gradcam_result or {}
        image_col, heatmap_col = st.columns(2, gap="large")
        with image_col:
            st.markdown("**Original X-ray**")
            if gradcam_result.get("original_image") is not None:
                st.image(gradcam_result["original_image"], use_container_width=True)
            elif st.session_state.original_xray is not None:
                st.image(st.session_state.original_xray, use_container_width=True)
        with heatmap_col:
            st.markdown("**Grad-CAM heatmap**")
            if gradcam_result.get("heatmap") is not None:
                st.image(gradcam_result["heatmap"], use_container_width=True)
            else:
                st.info("Grad-CAM heatmap unavailable for this study.")

        top_predictions_col, differential_col = st.columns(2, gap="large")
        with top_predictions_col:
            st.markdown("**Top predictions**")
            top_predictions = gradcam_result.get("top_findings") or []
            if top_predictions:
                for item in top_predictions:
                    st.write(f"- {item['label']}: {item['confidence']:.0%}")
            else:
                st.caption("No Grad-CAM prediction summary available.")
        with differential_col:
            st.markdown("**Differential diagnosis**")
            if st.session_state.analysis_result and st.session_state.analysis_result.conditions:
                for condition in st.session_state.analysis_result.conditions[:3]:
                    st.write(f"- {condition.name}: {condition.confidence:.0%}")
            else:
                st.caption("Run the multimodal analysis tab to populate the diagnostic differential.")

        st.markdown("**RAG evidence sources**")
        evidence_results = st.session_state.evidence_results or []
        if evidence_results:
            seen_sources = []
            for item in evidence_results:
                source = item.document.source or item.document.metadata.get("source", "medical_reference")
                if source not in seen_sources:
                    seen_sources.append(source)
                    st.write(f"- {source}")
        else:
            st.caption("No retrieved evidence sources yet.")
    else:
        st.info("Upload an image in the Multimodal Analysis tab to inspect Grad-CAM and the top differential.")


with ocr_tab:
    st.subheader("Lab report OCR")
    result = st.session_state.ocr_result
    if result:
        left, right = st.columns([0.8, 1.2], gap="large")
        with left:
            st.metric("OCR confidence", f"{result.confidence:.0%}")
            st.metric("Extracted metrics", len(result.metrics))
            if result.patient_info:
                st.json(result.patient_info)
        with right:
            if result.metrics:
                st.markdown("**Structured lab metrics**")
                for metric in result.metrics[:12]:
                    flag = "normal" if metric.in_normal_range else "out-of-range"
                    st.write(f"- {metric.name}: {metric.value} {metric.unit} ({flag})")
            with st.expander("Raw OCR text", expanded=False):
                st.code(result.raw_text[:4000])
    else:
        st.info("Upload a lab report in the analysis tab to inspect OCR output.")


with kb_tab:
    st.subheader("Retrieved medical evidence")
    pipeline = load_pipeline()
    doc_count = pipeline.vector_store.count()
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Indexed documents", doc_count)
    metric_col2.metric("Hybrid mode", "BM25 + Dense")

    search_query = st.text_input("Search the medical knowledge base", placeholder="pneumonia lower lobe consolidation treatment")
    if st.button("Search evidence"):
        if search_query.strip():
            st.session_state.evidence_results = pipeline.search_evidence(search_query, top_k=top_k)

    render_evidence_table(st.session_state.evidence_results)


with guide_tab:
    st.subheader("Implementation roadmap")
    st.markdown(
        """
        1. Replace the vision backbone with `torchxrayvision` as the default medical classifier and keep a CheXNet checkpoint path for supervised fine-tunes on CheXpert or MIMIC-CXR.
        2. Use MedCLIP or BioViL style encoders when you need image-text embeddings for multimodal retrieval instead of only disease logits.
        3. Ingest PubMed, local guidelines, ICD-10, SNOMED CT, and UMLS exports through the chunker and hybrid retriever so lexical codes and dense semantics both match.
        4. Calibrate model probabilities on a held-out validation split before surfacing confidence values to users.
        5. Keep the generated response structured as findings, explanation, retrieved evidence, and a human-review recommendation.
        """
    )
    st.code(
        "pip install torchxrayvision rank-bm25 sentence-transformers transformers pymupdf pydicom",
        language="bash",
    )
