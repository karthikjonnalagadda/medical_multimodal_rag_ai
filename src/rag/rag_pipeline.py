"""
rag/rag_pipeline.py
--------------------
Retrieval-Augmented Generation pipeline for medical diagnosis assistance.

Flow:
  multi-modal inputs → text processing → embedding → vector retrieval
  → context assembly → LLM → structured diagnosis response
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

# Internal imports (relative-safe)
try:
    from src.embeddings.embedding_model import MedicalEmbeddingModel
    from src.vector_db.faiss_store import VectorStore, Document, create_vector_store
    from src.preprocessing.text_cleaning import MedicalTextProcessor, ProcessedMedicalText
    from src.rag.hybrid_retriever import HybridMedicalRetriever
    from src.rag.multimodal_reasoning import reason_multimodal_case
except ImportError:
    # Allow running module standalone during development
    pass


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Condition:
    name: str
    confidence: float       # 0.0 – 1.0
    icd_code: Optional[str] = None
    description: Optional[str] = None


@dataclass
class DiagnosisResult:
    """Structured output of the RAG pipeline."""
    query: str
    conditions: list[Condition] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    explanation: str = ""
    recommended_tests: list[str] = field(default_factory=list)
    recommendation: str = "Radiologist or clinician confirmation is recommended before any action."
    disclaimer: str = (
        "⚠️  This output is AI-generated and intended for informational purposes only. "
        "It does NOT constitute medical advice. Always consult a qualified healthcare "
        "professional for diagnosis and treatment."
    )
    raw_llm_response: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "possible_conditions": [
                {
                    "name": c.name,
                    "confidence": f"{c.confidence:.0%}",
                    "icd_code": c.icd_code,
                }
                for c in self.conditions
            ],
            "evidence": self.evidence,
            "references": self.references,
            "explanation": self.explanation,
            "recommended_tests": self.recommended_tests,
            "recommendation": self.recommendation,
            "disclaimer": self.disclaimer,
        }

    def format_text(self) -> str:
        lines = ["=" * 55, "  MEDICAL AI ASSISTANT – DIAGNOSTIC INSIGHTS", "=" * 55, ""]
        lines += ["Possible Findings:", "-" * 30]
        for i, c in enumerate(self.conditions, 1):
            lines.append(f"  {i}. {c.name}  (Confidence: {c.confidence:.0%})")
            if c.icd_code:
                lines.append(f"     ICD-10: {c.icd_code}")
        lines += ["", "Evidence:", "-" * 30]
        for ev in self.evidence:
            lines.append(f"  • {ev}")
        if self.references:
            lines += ["", "References:", "-" * 30]
            for ref in self.references:
                lines.append(f"  • {ref}")
        if self.explanation:
            lines += ["", "Explanation:", "-" * 30, self.explanation]
        if self.recommended_tests:
            lines += ["", "Recommended Tests:", "-" * 30]
            for test in self.recommended_tests:
                lines.append(f"  • {test}")
        if self.recommendation:
            lines += ["", "Recommendation:", "-" * 30, self.recommendation]
        lines += ["", self.disclaimer]
        return "\n".join(lines)


# ── Prompt templates ───────────────────────────────────────────────────────────

DIAGNOSIS_SYSTEM_PROMPT = """You are a clinical decision-support AI. Analyse the provided medical \
data and retrieved evidence to generate a structured diagnostic response.

Rules:
- Provide 2-4 possible conditions ranked by likelihood.
- Express confidence as a percentage (0-100%).
- List key evidence supporting each condition.
- Reference the sources provided.
- Use uncertainty-aware language in the explanation (e.g., "may", "could", "is consistent with").
- Do NOT claim definitive diagnoses or definitive etiologies. If multiple causes are plausible, list them as possibilities.
- ALWAYS include the disclaimer that this is for informational purposes only.
- Return ONLY valid JSON (no markdown, no backticks, no extra text outside the JSON object).
- Respond ONLY with valid JSON matching the schema below.

Required JSON schema:
{
  "conditions": [
    {"name": "...", "confidence": 85, "icd_code": "J18.9", "description": "..."}
  ],
  "evidence": ["...", "..."],
  "references": ["...", "..."],
  "explanation": "A concise clinical reasoning paragraph.",
  "recommended_tests": ["...", "..."],
  "recommendation": "Recommended next clinical step."
}"""

DIAGNOSIS_USER_TEMPLATE = """Patient Data:
{patient_data}

Retrieved Medical Knowledge:
{retrieved_context}

Generate a structured diagnostic response for the above data."""


# ── LLM adapters ──────────────────────────────────────────────────────────────

class _LLMAdapter:
    """Base class for LLM backends."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class _HuggingFaceLLM(_LLMAdapter):
    """Local HuggingFace model (BioGPT / ClinicalBERT / Mistral)."""

    def __init__(self, model_name: str, max_new_tokens: int = 512):
        logger.info(f"Loading HuggingFace LLM: {model_name}")
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            self._pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_new_tokens=max_new_tokens,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype="auto",
            )
            logger.success(f"HuggingFace LLM loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace LLM: {e}")
            self._pipe = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._pipe is None:
            return self._mock_response()
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        result = self._pipe(prompt)
        return result[0]["generated_text"].split("<|assistant|>")[-1].strip()

    def _mock_response(self) -> str:
        """Return a mock structured response when model is unavailable."""
        return json.dumps({
            "conditions": [
                {"name": "Community-acquired Pneumonia", "confidence": 78,
                 "icd_code": "J18.9", "description": "Infection of lung parenchyma."},
                {"name": "Acute Bronchitis", "confidence": 45,
                 "icd_code": "J20.9", "description": "Inflammation of bronchial mucosa."},
            ],
            "evidence": [
                "Elevated WBC count indicating bacterial infection",
                "Lung opacity detected on imaging",
                "Productive cough and fever symptoms",
            ],
            "references": [
                "WHO Pneumonia Treatment Guidelines (2023)",
                "Harrison's Principles of Internal Medicine, 21st Ed.",
            ],
            "explanation": (
                "The combination of elevated WBC, fever, productive cough, and "
                "radiographic consolidation is consistent with community-acquired "
                "pneumonia. Acute bronchitis is considered as a differential given "
                "the cough and absence of confirmed lobar consolidation."
            ),
            "recommendation": "Radiology review and correlation with vitals, labs, and oxygenation are recommended.",
        })


class _OpenAICompatibleLLM(_LLMAdapter):
    """OpenAI-compatible LLM client for OpenAI or xAI endpoints."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
        provider_name: str = "OpenAI",
    ):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.provider_name = provider_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        def _fallback_json(err: Exception) -> str:
            # Avoid returning just "{}" which renders as an empty answer in the UI.
            msg = f"{self.provider_name} LLM error: {type(err).__name__}"
            if str(err):
                msg += f": {str(err)}"

            # Try to return a schema-compatible JSON based on the prompt being used.
            if '"diagnosis"' in system_prompt and '"possible_conditions"' in system_prompt:
                return json.dumps(
                    {
                        "diagnosis": "",
                        "confidence": "",
                        "possible_conditions": [],
                        "explanation": msg,
                        "recommended_tests": [],
                        "next_steps": [],
                    }
                )

            if '"conditions"' in system_prompt and '"evidence"' in system_prompt:
                return json.dumps(
                    {
                        "conditions": [],
                        "evidence": [],
                        "references": [],
                        "explanation": msg,
                        "recommended_tests": [],
                        "recommendation": "Please retry after verifying LLM configuration.",
                        "disclaimer": "This is for informational purposes only and not medical advice.",
                    }
                )

            return "{}"

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Enforce JSON output using OpenAI-compatible JSON mode where supported.
            # NOTE: Some providers (Groq) may hard-fail the request if JSON validation fails.
            # We handle that by retrying once without JSON mode and then falling back to a
            # schema-compatible JSON object.
            use_strict_json_mode = self.provider_name.lower() in {"openai", "xai", "groq"}

            def _ensure_json_object(text: str) -> str:
                candidate = (text or "").strip()
                if not candidate:
                    return "{}"
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    pass
                match = re.search(r"\{.*\}", candidate, re.DOTALL)
                if match:
                    try:
                        json.loads(match.group(0))
                        return match.group(0)
                    except Exception:
                        return _fallback_json(ValueError("Invalid JSON returned by LLM"))
                return _fallback_json(ValueError("Invalid JSON returned by LLM"))

            try:
                kwargs: dict = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": 0.1,
                }
                if use_strict_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                return _ensure_json_object(resp.choices[0].message.content or "{}")
            except Exception as e:
                # If strict JSON mode caused the failure, retry once without it.
                if use_strict_json_mode and "json_validate_failed" in str(e):
                    logger.warning(f"{self.provider_name} JSON mode validation failed; retrying without JSON mode: {e}")
                if use_strict_json_mode:
                    try:
                        resp = client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=self.max_tokens,
                            temperature=0.1,
                        )
                        return _ensure_json_object(resp.choices[0].message.content or "{}")
                    except Exception as e2:
                        logger.error(f"{self.provider_name} API error (retry): {e2}")
                        return _fallback_json(e2)
                logger.error(f"{self.provider_name} API error: {e}")
                return _fallback_json(e)
        except Exception as e:
            logger.error(f"{self.provider_name} API error: {e}")
            return _fallback_json(e)


# ── RAG Pipeline ───────────────────────────────────────────────────────────────

class MedicalRAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline for medical diagnosis.

    Usage
    -----
    >>> pipeline = MedicalRAGPipeline()
    >>> result = pipeline.diagnose(
    ...     lab_text="Hemoglobin: 9.5 g/dL  WBC: 15000",
    ...     image_findings=["Lung opacity in lower right lobe"],
    ...     symptoms=["fever", "cough", "shortness of breath"],
    ... )
    >>> print(result.format_text())
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[MedicalEmbeddingModel] = None,
        text_processor: Optional[MedicalTextProcessor] = None,
        llm_backend: str = "huggingface",
        llm_model: str = "microsoft/BioGPT-Large-PubMedQA",
        openai_api_key: str = "",
        xai_api_key: str = "",
        xai_base_url: str = "https://api.x.ai/v1",
        groq_api_key: str = "",
        groq_base_url: str = "https://api.groq.com/openai/v1",
        top_k_retrieval: int = 5,
    ):
        # Text processor
        self.text_processor = text_processor or MedicalTextProcessor()

        # Embedding model
        self.embedding_model = embedding_model or MedicalEmbeddingModel()

        # Vector store
        if vector_store is None:
            logger.warning("No vector store provided – creating default persistent ChromaDB store.")
            self.vector_store = create_vector_store("chromadb")
        else:
            self.vector_store = vector_store

        self.top_k = top_k_retrieval
        self.document_registry: dict[str, Document] = {}
        self.hybrid_retriever = HybridMedicalRetriever(self.vector_store, self.embedding_model)

        # LLM
        if llm_backend == "groq":
            self.llm = _OpenAICompatibleLLM(
                model=llm_model,
                api_key=groq_api_key,
                base_url=groq_base_url,
                provider_name="Groq",
            )
        elif llm_backend == "xai":
            self.llm = _OpenAICompatibleLLM(
                model=llm_model,
                api_key=xai_api_key,
                base_url=xai_base_url,
                provider_name="xAI",
            )
        elif llm_backend == "openai":
            self.llm = _OpenAICompatibleLLM(
                model=llm_model,
                api_key=openai_api_key,
                provider_name="OpenAI",
            )
        else:
            self.llm = _HuggingFaceLLM(model_name=llm_model)

        logger.info(f"MedicalRAGPipeline ready | llm={llm_backend} | top_k={top_k_retrieval}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def diagnose(
        self,
        lab_text: str = "",
        image_findings: list[str] | None = None,
        symptoms: list[str] | None = None,
        patient_notes: str = "",
        retrieved_docs_override: list["Document"] | None = None,
    ) -> DiagnosisResult:
        """
        Run the full RAG pipeline and return a structured diagnosis.

        Parameters
        ----------
        lab_text       : OCR-extracted lab report text
        image_findings : list of findings from image analysis
        symptoms       : list of patient-reported symptoms
        patient_notes  : optional free-text clinical notes

        Returns
        -------
        DiagnosisResult
        """
        logger.info("Starting RAG diagnosis pipeline …")

        # 1. Text processing
        processed: ProcessedMedicalText = self.text_processor.process(
            lab_text=lab_text,
            image_findings=image_findings,
            symptoms=symptoms,
            patient_notes=patient_notes,
        )

        # 2. Generate query embedding
        query_text = processed.query_ready or processed.cleaned_text[:1000]
        query_embedding = self.embedding_model.embed(query_text)

        # 3. Retrieve relevant knowledge
        retrieved_docs: list[Document] = []
        if retrieved_docs_override is not None:
            retrieved_docs = retrieved_docs_override
            logger.info(f"Using {len(retrieved_docs)} retrieved documents supplied by external retriever.")
        elif self.vector_store.count() > 0:
            retrieved_docs = self.vector_store.search_similar(query_embedding, top_k=self.top_k)
            logger.info(f"Retrieved {len(retrieved_docs)} knowledge documents.")
        else:
            logger.warning("Vector store is empty – proceeding without retrieval.")

        # 4. Build context
        retrieved_context = self._build_context(retrieved_docs)
        patient_data = processed.cleaned_text

        reasoning_bundle = None
        try:
            reasoning_bundle = reason_multimodal_case(
                symptoms=symptoms or [],
                clinical_notes="\n\n".join(part for part in [patient_notes, lab_text] if part),
                vision_findings=image_findings or [],
                retrieved_docs=retrieved_docs,
                llm=self.llm,
            )
        except Exception as exc:
            logger.warning("Multimodal reasoning helper failed, falling back to base prompt: {}", exc)

        if reasoning_bundle and reasoning_bundle.get("differential_diagnosis"):
            result = self._result_from_multimodal_reasoning(reasoning_bundle, query_text, retrieved_docs)
        else:
            # 5. LLM generation
            user_prompt = DIAGNOSIS_USER_TEMPLATE.format(
                patient_data=patient_data,
                retrieved_context=retrieved_context,
            )
            llm_response = self.llm.generate(DIAGNOSIS_SYSTEM_PROMPT, user_prompt)
            logger.debug(f"LLM response length: {len(llm_response)} chars")

            # 6. Parse response
            result = self._parse_llm_response(llm_response, query_text)
            result.references += [d.source for d in retrieved_docs if d.source]
            if not result.evidence:
                result.evidence = self._extract_evidence_from_docs(retrieved_docs)

        result.references = list(dict.fromkeys([ref for ref in result.references if ref]))

        logger.success("RAG diagnosis complete.")
        return result

    def ingest_knowledge(
        self,
        documents: list[str],
        sources: list[str] | None = None,
        metadata: list[dict] | None = None,
    ) -> None:
        """
        Add documents to the knowledge base.

        Parameters
        ----------
        documents : list of text chunks from medical literature
        sources   : list of source identifiers (e.g. "WHO Guidelines 2023")
        metadata  : list of metadata dicts
        """
        import uuid
        sources = sources or ["" for _ in documents]
        metadata = metadata or [{} for _ in documents]

        embeddings = self.embedding_model.embed_batch(documents)
        docs = [
            Document(
                id=str(uuid.uuid4()),
                text=text,
                embedding=embeddings[i],
                metadata=meta,
                source=src,
            )
            for i, (text, src, meta) in enumerate(zip(documents, sources, metadata))
        ]
        self.vector_store.add_documents(docs)
        for doc in docs:
            self.document_registry[doc.id] = doc
        self.hybrid_retriever.add_documents(docs)
        logger.success(f"Ingested {len(docs)} documents into knowledge base.")

    def search_evidence(
        self,
        query_text: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ):
        return self.hybrid_retriever.search(
            query_text=query_text,
            top_k=top_k or self.top_k,
            metadata_filter=metadata_filter,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_context(self, docs: list[Document]) -> str:
        if not docs:
            return "No relevant documents retrieved."
        parts = []
        for i, doc in enumerate(docs, 1):
            source_label = f" [{doc.source}]" if doc.source else ""
            metadata = doc.metadata or {}
            disease = metadata.get("disease", "unknown")
            symptoms = metadata.get("symptoms", "n/a")
            parts.append(
                f"[{i}]{source_label} disease={disease} symptoms={symptoms}\n{doc.text}"
            )
        return "\n\n".join(parts)

    def _parse_llm_response(self, response: str, query: str) -> DiagnosisResult:
        """Parse the JSON LLM response into a DiagnosisResult."""
        result = DiagnosisResult(query=query, raw_llm_response=response)

        # Extract JSON block
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            logger.warning("Could not find JSON in LLM response.")
            result.explanation = response
            return result

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            result.explanation = response
            return result

        # Map conditions
        for cond_data in data.get("conditions", []):
            confidence_raw = cond_data.get("confidence", 50)
            # Handle both 0.85 and 85 formats
            confidence = float(confidence_raw)
            if confidence > 1.0:
                confidence /= 100.0
            result.conditions.append(Condition(
                name=cond_data.get("name", "Unknown"),
                confidence=confidence,
                icd_code=cond_data.get("icd_code"),
                description=cond_data.get("description"),
            ))

        result.evidence = data.get("evidence", [])
        result.references = [r for r in data.get("references", []) if r]
        result.explanation = data.get("explanation", "")
        result.recommended_tests = data.get("recommended_tests", [])
        result.recommendation = data.get(
            "recommendation",
            "Clinical confirmation with radiology and treating physician review is recommended.",
        )

        # Sort by confidence
        result.conditions.sort(key=lambda c: c.confidence, reverse=True)
        return result

    def _result_from_multimodal_reasoning(
        self,
        reasoning_bundle: dict,
        query_text: str,
        retrieved_docs: list[Document],
    ) -> DiagnosisResult:
        result = DiagnosisResult(
            query=query_text,
            raw_llm_response=json.dumps(reasoning_bundle),
        )
        for item in reasoning_bundle.get("differential_diagnosis", []):
            result.conditions.append(
                Condition(
                    name=item.get("disease", "Unknown"),
                    confidence=float(item.get("confidence", 0.0)),
                )
            )

        result.conditions.sort(key=lambda c: c.confidence, reverse=True)
        result.explanation = reasoning_bundle.get("explanation", "")
        result.recommended_tests = reasoning_bundle.get("recommended_tests", [])
        if result.recommended_tests:
            result.recommendation = "Recommended follow-up tests: " + ", ".join(result.recommended_tests)
        result.references = [doc.source for doc in retrieved_docs if doc.source]
        result.evidence = self._extract_evidence_from_docs(retrieved_docs)
        return result

    def _extract_evidence_from_docs(self, docs: list[Document]) -> list[str]:
        evidence = []
        for doc in docs[:4]:
            metadata = doc.metadata or {}
            source = doc.source or metadata.get("source", "medical_reference")
            disease = metadata.get("disease")
            prefix = f"{source}"
            if disease:
                prefix += f" ({disease})"
            evidence.append(f"{prefix}: {doc.text[:180].strip()}")
        return evidence


# ── Convenience function ───────────────────────────────────────────────────────

def load_sample_knowledge_base(pipeline: MedicalRAGPipeline) -> None:
    """
    Load a small built-in sample knowledge base for testing / demo.
    Replaces real PubMed/WHO document ingestion in offline mode.
    """
    sample_docs = [
        "Pneumonia is an infection that inflames the air sacs in one or both lungs. "
        "The air sacs may fill with fluid or pus. Symptoms include cough with phlegm, "
        "fever, chills, and difficulty breathing. WBC count is typically elevated.",
        "Tuberculosis (TB) is caused by Mycobacterium tuberculosis. Symptoms include "
        "persistent cough lasting 3+ weeks, coughing blood, chest pain, fatigue, fever, "
        "night sweats, and unexplained weight loss. Chest X-ray shows upper lobe infiltrates.",
        "Pulmonary edema is a condition caused by excess fluid in the lungs. This fluid "
        "collects in the air sacs, making it difficult to breathe. It may be caused by "
        "heart problems or non-cardiac causes such as pneumonia.",
        "Pleural effusion refers to the accumulation of excess fluid between the layers "
        "of the pleura surrounding the lungs. Symptoms include shortness of breath and "
        "chest pain. It can be caused by heart failure, malignancy, or infection.",
        "COPD (Chronic Obstructive Pulmonary Disease) is a chronic inflammatory lung "
        "disease that causes obstructed airflow from the lungs. Symptoms include breathing "
        "difficulty, cough, mucus production and wheezing. Long-term exposure to irritating "
        "gases, most often from cigarette smoke, causes COPD.",
        "Atelectasis is a complete or partial collapse of the entire lung or area (lobe) "
        "of the lung. It occurs when the tiny air sacs (alveoli) within the lung become "
        "deflated or possibly filled with alveolar fluid.",
        "According to WHO guidelines, community-acquired pneumonia should be treated with "
        "antibiotics in most cases. Amoxicillin is the recommended first-line treatment. "
        "Patients with severe pneumonia should be hospitalised.",
        "ICD-10 Code J18.9 – Pneumonia, unspecified organism. "
        "ICD-10 Code A15 – Respiratory tuberculosis. "
        "ICD-10 Code J81 – Pulmonary edema.",
    ]
    sources = [
        "Mayo Clinic – Pneumonia Overview",
        "WHO TB Factsheet",
        "Cleveland Clinic – Pulmonary Edema",
        "NEJM – Pleural Effusion Review",
        "Mayo Clinic – COPD Overview",
        "Mayo Clinic – Atelectasis",
        "WHO Pneumonia Treatment Guidelines 2023",
        "ICD-10 Disease Database",
    ]
    pipeline.ingest_knowledge(sample_docs, sources=sources)
    logger.success("Sample knowledge base loaded.")
