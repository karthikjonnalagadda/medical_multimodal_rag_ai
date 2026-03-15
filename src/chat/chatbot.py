"""Medical Chatbot - Fast text-only medical conversation with RAG and Groq LLM.

Optimized for speed:
- Embedding cache for repeated queries
- Limited RAG retrieval (top_k=3)
- Short context window
- Lightweight prompts
- Async-friendly design

Target response time: 1-2 seconds per query.
"""

import json
import logging
import re
from typing import Any, Optional
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.config import EMBEDDING_CONFIG, VECTOR_DB_CONFIG, get_llm_runtime_config
from src.embeddings.embedding_model import MedicalEmbeddingModel
from src.rag.rag_pipeline import MedicalRAGPipeline
from src.vector_db.faiss_store import create_vector_store

logger = logging.getLogger(__name__)


CLINICAL_JSON_SCHEMA: dict[str, Any] = {
    "diagnosis": "",
    "confidence": "",
    "possible_conditions": [],
    "explanation": "",
    "recommended_tests": [],
    "next_steps": [],
}


def normalize_clinical_json_response(text: str) -> dict[str, Any]:
    """
    Normalize an LLM response into a strict clinical JSON schema.

    Some LLM backends return a raw JSON object, others may wrap JSON in Markdown,
    and fallback models may return free text. This extracts the best-effort JSON
    object and coerces it into the required schema.
    """
    raw = (text or "").strip()
    payload: dict[str, Any] = {}

    if raw:
        # 1) Direct JSON object.
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}

        # 2) JSON embedded in surrounding text/codefences.
        if not payload:
            start = raw.find("{")
            end = raw.rfind("}")
            if 0 <= start < end:
                candidate = raw[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = {}

        # 3) Regex fallback (may still fail if braces are unbalanced).
        if not payload:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = {}

    normalized: dict[str, Any] = dict(CLINICAL_JSON_SCHEMA)

    def as_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        return str(value).strip()

    def as_list_of_strings(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        if isinstance(value, dict):
            value = value.get("items") or value.get("values") or value.get("list") or []
        if not isinstance(value, list):
            text_value = as_str(value)
            return [text_value] if text_value else []

        out: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                if item.strip():
                    out.append(item.strip())
                continue
            if isinstance(item, dict):
                name = item.get("name") or item.get("condition") or item.get("title") or ""
                if name:
                    out.append(as_str(name))
                    continue
            text_item = as_str(item)
            if text_item:
                out.append(text_item)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in out:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    diagnosis = payload.get("diagnosis") or payload.get("primary_diagnosis") or ""
    confidence = payload.get("confidence") or payload.get("diagnosis_confidence") or ""
    possible_conditions = payload.get("possible_conditions") or payload.get("conditions") or payload.get("differential") or []
    explanation = payload.get("explanation") or payload.get("analysis") or payload.get("answer") or ""
    recommended_tests = payload.get("recommended_tests") or payload.get("tests") or []
    next_steps = payload.get("next_steps") or payload.get("follow_up") or payload.get("followup") or []

    normalized["diagnosis"] = as_str(diagnosis)
    normalized["confidence"] = as_str(confidence)
    normalized["possible_conditions"] = as_list_of_strings(possible_conditions)
    normalized["explanation"] = as_str(explanation)
    normalized["recommended_tests"] = as_list_of_strings(recommended_tests)
    normalized["next_steps"] = as_list_of_strings(next_steps)

    if not normalized["diagnosis"] and normalized["possible_conditions"]:
        normalized["diagnosis"] = normalized["possible_conditions"][0]

    if not payload and raw and not normalized["explanation"]:
        normalized["explanation"] = raw

    return normalized


def clinical_json_to_structured_sections(clinical: dict[str, Any]) -> dict[str, Any]:
    """Convert strict clinical JSON into the existing frontend-friendly `structured` shape."""
    diagnosis = str(clinical.get("diagnosis") or "").strip()
    confidence = str(clinical.get("confidence") or "").strip()
    possible_conditions = clinical.get("possible_conditions") or []
    explanation = str(clinical.get("explanation") or "").strip()
    recommended_tests = clinical.get("recommended_tests") or []
    next_steps = clinical.get("next_steps") or []

    sections: list[dict[str, str]] = []
    if diagnosis or confidence:
        line = diagnosis
        if confidence:
            line = f"{diagnosis} ({confidence})" if diagnosis else confidence
        sections.append({"title": "Diagnosis", "content": line})

    if possible_conditions:
        sections.append(
            {
                "title": "Possible Conditions",
                "content": "\n".join(f"- {item}" for item in possible_conditions if str(item).strip()),
            }
        )

    if recommended_tests:
        sections.append(
            {
                "title": "Recommended Tests",
                "content": "\n".join(f"- {item}" for item in recommended_tests if str(item).strip()),
            }
        )

    if next_steps:
        sections.append(
            {
                "title": "Next Steps",
                "content": "\n".join(f"- {item}" for item in next_steps if str(item).strip()),
            }
        )

    structured: dict[str, Any] = {"summary": explanation or None, "sections": sections, "clinical_json": clinical}
    if not sections and explanation:
        structured["full_text"] = explanation
    return structured


@dataclass
class ChatResponse:
    """Structured chat response from the medical chatbot."""
    
    answer: str
    sources: list[dict]
    query: str
    retrieval_time: float = 0.0
    llm_time: float = 0.0
    total_time: float = 0.0
    structured: dict | None = None  # Frontend-friendly sections
    clinical: dict | None = None  # Strict clinical JSON schema


class MedicalChatbot:
    """Fast, lightweight medical chatbot using RAG and Groq LLM.
    
    Features:
    - Embedding caching for fast repeated queries
    - Minimal RAG retrieval (top_k=3 by default)
    - Compact context for speed
    - Streamlined prompts
    - Minimal token consumption on Groq
    """
    
    SYSTEM_PROMPT = """You are a medical AI assistant providing educational information.
Respond ONLY with valid JSON matching this exact schema (no markdown, no extra keys):
{
  "diagnosis": "",
  "confidence": "",
  "possible_conditions": [],
  "explanation": "",
  "recommended_tests": [],
  "next_steps": []
}
Use uncertainty-aware language (e.g., "may", "could", "is consistent with") and do not claim definitive diagnoses.
If you are uncertain, leave diagnosis/confidence empty and explain the uncertainty in "explanation"."""

    CHAT_PROMPT_TEMPLATE = """User question: {question}

Retrieved medical context:
{context}

Return ONLY a JSON object following the schema from the system prompt.
Rules:
- possible_conditions: list of short condition names (strings)
- recommended_tests / next_steps: lists of short actionable strings
- confidence: a short string like "low", "medium", "high" or "85%"
"""

    def __init__(
        self,
        vector_store=None,
        embedding_model: Optional[MedicalEmbeddingModel] = None,
        llm_backend: str = "groq",
        llm_model: str = "mixtral-8x7b-32768",
        groq_api_key: str = "",
        groq_base_url: str = "https://api.groq.com/openai/v1",
        openai_api_key: str = "",
        xai_api_key: str = "",
        xai_base_url: str = "https://api.x.ai/v1",
        top_k: int = 3,
        max_tokens: int = 256,
        temperature: float = 0.1,
    ):
        """Initialize the medical chatbot.
        
        Args:
            vector_store: Vector store instance (ChromaDB/FAISS). If None, creates one.
            embedding_model: Embedding model. If None, creates one.
            llm_backend: "groq", "openai", "xai", or "huggingface"
            llm_model: Model name for the selected backend
            groq_api_key: Groq API key
            groq_base_url: Groq API base URL
            openai_api_key: OpenAI API key
            xai_api_key: xAI API key
            xai_base_url: xAI API base URL
            top_k: Number of documents to retrieve (default 3 for speed)
            max_tokens: Max tokens in LLM response (default 256 for speed)
            temperature: LLM temperature (default 0.1 for consistency)
        """
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize embedding model if not provided
        if embedding_model is None:
            logger.info("Loading embedding model...")
            embedding_model = MedicalEmbeddingModel(
                backend=EMBEDDING_CONFIG.get("backend", "sentence_transformer"),
                model_name=EMBEDDING_CONFIG.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                batch_size=EMBEDDING_CONFIG.get("batch_size", 32),
                device=EMBEDDING_CONFIG.get("device", "cpu"),
            )
        self.embedding_model = embedding_model
        
        # Initialize vector store if not provided
        if vector_store is None:
            logger.info("Initializing vector store...")
            vector_store = create_vector_store(
                backend=VECTOR_DB_CONFIG.get("backend", "chromadb"),
                dim=embedding_model.get_embedding_dim(),
                faiss_index_path=VECTOR_DB_CONFIG.get("faiss_index_path"),
                chroma_persist_dir=VECTOR_DB_CONFIG.get("chroma_persist_dir"),
                collection_name=VECTOR_DB_CONFIG.get("collection_name", "medical_knowledge"),
            )
        self.vector_store = vector_store
        
        # Initialize RAG pipeline for retrieval and LLM
        logger.info(f"Initializing RAG pipeline with backend={llm_backend}, model={llm_model}")
        self.rag_pipeline = MedicalRAGPipeline(
            vector_store=vector_store,
            embedding_model=embedding_model,
            llm_backend=llm_backend,
            llm_model=llm_model,
            openai_api_key=openai_api_key,
            xai_api_key=xai_api_key,
            xai_base_url=xai_base_url,
            groq_api_key=groq_api_key,
            groq_base_url=groq_base_url,
            top_k_retrieval=top_k,
        )
        
        # Configure LLM for fast responses
        if hasattr(self.rag_pipeline.llm, 'max_tokens'):
            self.rag_pipeline.llm.max_tokens = max_tokens
        if hasattr(self.rag_pipeline.llm, 'temperature'):
            self.rag_pipeline.llm.temperature = temperature
    
    @lru_cache(maxsize=256)
    def _embed_cached(self, text: str) -> np.ndarray:
        """Cached embedding lookup to avoid re-embedding repeated queries.
        
        Note: This caches embeddings for the exact text. In a production system,
        you might want semantic caching or query normalization.
        """
        return self.embedding_model.embed(text)
    
    def _build_compact_context(self, documents: list) -> str:
        """Build a compact context string from retrieved documents.
        
        Optimized for minimal token usage while maintaining relevance.
        """
        if not documents:
            return "No relevant medical knowledge found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(documents[:self.top_k], 1):
            # Extract just the essential content
            text = doc.text.strip()
            if len(text) > 200:
                text = text[:200] + "..."
            context_parts.append(f"[{i}] {text}")
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, message: str) -> ChatResponse:
        """Generate a chatbot response to a user message.
        
        Pipeline:
        1. Embed the user message (with caching)
        2. Retrieve top-k relevant documents from vector store
        3. Build compact context
        4. Call Groq LLM to generate response
        5. Return response with sources
        
        Target: 1-2 seconds total response time.
        
        Args:
            message: User's medical question
            
        Returns:
            ChatResponse with answer, sources, and timing information
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Embed message (with cache)
            query_embedding = self._embed_cached(message)
            retrieval_start = time.time()
            
            # Step 2: Retrieve relevant documents
            retrieved_docs = self.vector_store.search_similar(
                query_embedding,
                top_k=self.top_k
            )
            retrieval_time = time.time() - retrieval_start
            
            # Step 3: Build compact context
            context = self._build_compact_context(retrieved_docs)
            
            # Step 4: Call Groq LLM
            llm_start = time.time()
            user_prompt = self.CHAT_PROMPT_TEMPLATE.format(
                question=message,
                context=context
            )
            
            answer = self.rag_pipeline.llm.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            llm_time = time.time() - llm_start

            clinical = normalize_clinical_json_response(answer)
            
            # Step 5: Build sources list
            sources = [
                {
                    "text": doc.text[:150] + "..." if len(doc.text) > 150 else doc.text,
                    "metadata": doc.metadata or {},
                }
                for doc in retrieved_docs[:self.top_k]
            ]
            
            # Step 6: Produce frontend-friendly structure + attach strict clinical JSON
            structured = clinical_json_to_structured_sections(clinical)
            
            total_time = time.time() - start_time
            
            response = ChatResponse(
                answer=clinical.get("explanation") or answer,
                sources=sources,
                query=message,
                retrieval_time=retrieval_time,
                llm_time=llm_time,
                total_time=total_time,
                structured=structured,
                clinical=clinical,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return ChatResponse(
                answer=f"Error processing your question: {str(e)}",
                sources=[],
                query=message,
                retrieval_time=0.0,
                llm_time=0.0,
                total_time=time.time() - start_time,
                structured=None,
            )
    
    def get_knowledge_base_stats(self) -> dict:
        """Get statistics about the knowledge base."""
        return {
            "total_documents": self.vector_store.count(),
            "embedding_model": self.embedding_model.model_name,
            "vector_store_backend": self.vector_store.__class__.__name__,
        }


# Global singleton instance
_chatbot: Optional[MedicalChatbot] = None


def load_chatbot() -> MedicalChatbot:
    """Load or create the global chatbot instance.
    
    Uses singleton pattern for efficiency in Streamlit caching.
    Call this once at startup and reuse the instance.
    
    Returns:
        MedicalChatbot instance
    """
    global _chatbot
    
    if _chatbot is None:
        logger.info("Initializing medical chatbot...")
        
        # Get current LLM configuration
        llm_config = get_llm_runtime_config()
        
        _chatbot = MedicalChatbot(
            llm_backend=llm_config.get("backend", "groq"),
            llm_model=llm_config.get("model", "mixtral-8x7b-32768"),
            groq_api_key=llm_config.get("api_key", "") if llm_config.get("backend") == "groq" else "",
            groq_base_url=llm_config.get("base_url", "https://api.groq.com/openai/v1") if llm_config.get("backend") == "groq" else "",
            openai_api_key=llm_config.get("api_key", "") if llm_config.get("backend") == "openai" else "",
            xai_api_key=llm_config.get("api_key", "") if llm_config.get("backend") == "xai" else "",
            xai_base_url=llm_config.get("base_url", "https://api.x.ai/v1") if llm_config.get("backend") == "xai" else "",
            top_k=VECTOR_DB_CONFIG.get("top_k", 3) if VECTOR_DB_CONFIG.get("top_k", 5) <= 3 else 3,  # Cap at 3 for speed
            max_tokens=256,
            temperature=0.1,
        )
        
        logger.info(f"Medical chatbot initialized with {llm_config.get('backend')} backend")
    
    return _chatbot
