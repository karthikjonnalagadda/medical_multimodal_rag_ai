#!/usr/bin/env python3
"""
Quick Setup & Test Script for Medical Chatbot

This script verifies the chatbot installation and performs basic tests.

Usage:
    python test_chatbot.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set."""
    logger.info("🔍 Checking environment variables...")
    
    import os
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.warning("⚠️ GROQ_API_KEY not set. Set it with:")
        logger.warning('   $env:GROQ_API_KEY = "your-key"  (PowerShell)')
        logger.warning('   export GROQ_API_KEY="your-key"  (Bash)')
        return False
    
    logger.info(f"✅ GROQ_API_KEY is set")
    
    groq_model = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    logger.info(f"✅ GROQ_MODEL = {groq_model}")
    
    return True


def check_imports():
    """Check if all required packages are importable."""
    logger.info("🔍 Checking required imports...")
    
    required_packages = [
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("openai", "OpenAI Python Client"),
    ]
    
    all_ok = True
    for pkg, name in required_packages:
        try:
            __import__(pkg)
            logger.info(f"✅ {name} installed")
        except ImportError:
            logger.error(f"❌ {name} NOT installed. Run: pip install {pkg}")
            all_ok = False
    
    return all_ok


def check_chatbot_module():
    """Check if chatbot modules can be imported."""
    logger.info("🔍 Checking chatbot modules...")
    
    try:
        from src.chat.chatbot import MedicalChatbot, load_chatbot, ChatResponse
        logger.info("✅ Chatbot module imports OK")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import chatbot: {e}")
        return False


def test_embedding_model():
    """Test if embedding model loads correctly."""
    logger.info("🔍 Testing embedding model...")
    
    try:
        from src.embeddings.embedding_model import MedicalEmbeddingModel
        from src.config import EMBEDDING_CONFIG
        
        logger.info("Loading embedding model (this may take a moment)...")
        model = MedicalEmbeddingModel(
            backend=EMBEDDING_CONFIG.get("backend", "sentence_transformer"),
            model_name=EMBEDDING_CONFIG.get("model_name"),
            device=EMBEDDING_CONFIG.get("device", "cpu"),
        )
        
        # Test embedding
        test_text = "What is pneumonia?"
        embedding = model.embed(test_text)
        logger.info(f"✅ Embedding model working (dimension: {embedding.shape[0]})")
        return True
    
    except Exception as e:
        logger.error(f"❌ Embedding model failed: {e}")
        return False


def test_vector_store():
    """Test if vector store initializes correctly."""
    logger.info("🔍 Testing vector store...")
    
    try:
        from src.vector_db.faiss_store import create_vector_store
        from src.config import VECTOR_DB_CONFIG
        from src.embeddings.embedding_model import MedicalEmbeddingModel
        from src.config import EMBEDDING_CONFIG
        
        # Create embedding model for dimension
        embedding_model = MedicalEmbeddingModel(
            backend=EMBEDDING_CONFIG.get("backend", "sentence_transformer"),
            model_name=EMBEDDING_CONFIG.get("model_name"),
            device=EMBEDDING_CONFIG.get("device", "cpu"),
        )
        
        # Create vector store
        vector_store = create_vector_store(
            backend=VECTOR_DB_CONFIG.get("backend", "chromadb"),
            dim=embedding_model.get_embedding_dim(),
            faiss_index_path=VECTOR_DB_CONFIG.get("faiss_index_path"),
            chroma_persist_dir=VECTOR_DB_CONFIG.get("chroma_persist_dir"),
            collection_name=VECTOR_DB_CONFIG.get("collection_name"),
        )
        
        doc_count = vector_store.count()
        logger.info(f"✅ Vector store initialized ({doc_count} documents)")
        return True
    
    except Exception as e:
        logger.error(f"❌ Vector store failed: {e}")
        return False


def test_chatbot_initialization():
    """Test full chatbot initialization."""
    logger.info("🔍 Testing chatbot initialization...")
    
    try:
        from src.chat.chatbot import load_chatbot
        
        logger.info("Loading chatbot (this may take a moment)...")
        chatbot = load_chatbot()
        
        logger.info("✅ Chatbot initialized successfully")
        
        # Print stats
        stats = chatbot.get_knowledge_base_stats()
        logger.info(f"   Total documents: {stats.get('total_documents', 'N/A')}")
        logger.info(f"   Embedding model: {stats.get('embedding_model', 'N/A')}")
        logger.info(f"   Vector store: {stats.get('vector_store_backend', 'N/A')}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Chatbot initialization failed: {e}")
        return False


def test_chat_response():
    """Test generating a chat response."""
    logger.info("🔍 Testing chat response generation...")
    
    try:
        from src.chat.chatbot import load_chatbot
        
        chatbot = load_chatbot()
        
        logger.info("Generating test response...")
        response = chatbot.generate_response("What is pneumonia?")
        
        logger.info(f"✅ Chat response generated")
        logger.info(f"   Response time: {response.total_time:.2f}s")
        logger.info(f"   Answer length: {len(response.answer)} chars")
        logger.info(f"   Sources: {len(response.sources)}")
        logger.info(f"\n   Preview of answer:\n   {response.answer[:200]}...")
        
        if response.total_time > 2.0:
            logger.warning(f"⚠️ Response slower than target (2.0s > 1-2s)")
        else:
            logger.info(f"✅ Response time within target (1-2s)")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Chat response failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    logger.info("=" * 60)
    logger.info("Medical Chatbot - Setup & Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Environment Variables", check_environment),
        ("Required Packages", check_imports),
        ("Chatbot Module", check_chatbot_module),
        ("Embedding Model", test_embedding_model),
        ("Vector Store", test_vector_store),
        ("Chatbot Initialization", test_chatbot_initialization),
        ("Chat Response Generation", test_chat_response),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info("")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status:8} {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"Result: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("✅ All tests passed! Your chatbot is ready to use.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Start the FastAPI backend:")
        logger.info("   python -m uvicorn src.api.main:app --reload --port 8000")
        logger.info("")
        logger.info("2. In another terminal, start Streamlit:")
        logger.info("   streamlit run app/streamlit_app.py")
        logger.info("")
        logger.info("3. Visit http://localhost:8501/medical_chatbot")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
