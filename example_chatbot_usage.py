#!/usr/bin/env python3
"""
Example Usage of Medical Chatbot

Shows how to use the chatbot programmatically without Streamlit or FastAPI.

Usage:
    python example_chatbot_usage.py
"""

import logging
from src.chat.chatbot import load_chatbot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """Example 1: Basic chatbot usage."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Usage")
    logger.info("=" * 60)
    
    # Load the chatbot
    logger.info("Loading chatbot...")
    chatbot = load_chatbot()
    
    # Ask a question
    question = "What are the main symptoms of diabetes?"
    logger.info(f"\nQuestion: {question}")
    
    # Generate response
    response = chatbot.generate_response(question)
    
    # Display results
    logger.info(f"\nAnswer:\n{response.answer}")
    logger.info(f"\nResponse time: {response.total_time:.2f}s")
    logger.info(f"  - Retrieval: {response.retrieval_time:.3f}s")
    logger.info(f"  - LLM: {response.llm_time:.3f}s")
    
    if response.sources:
        logger.info(f"\nSources ({len(response.sources)}):")
        for i, source in enumerate(response.sources, 1):
            logger.info(f"  [{i}] {source['text'][:100]}...")


def example_2_multiple_questions():
    """Example 2: Multiple questions (demonstrates caching)."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Multiple Questions (with caching)")
    logger.info("=" * 60)
    
    chatbot = load_chatbot()
    
    questions = [
        "What is hypertension?",
        "How is hypertension treated?",
        "What are the risk factors for hypertension?",
    ]
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n[{i}] {question}")
        response = chatbot.generate_response(question)
        logger.info(f"Response time: {response.total_time:.2f}s")
        logger.info(f"Answer: {response.answer[:200]}...")


def example_3_knowledge_base_stats():
    """Example 3: Get knowledge base statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Knowledge Base Statistics")
    logger.info("=" * 60)
    
    chatbot = load_chatbot()
    stats = chatbot.get_knowledge_base_stats()
    
    logger.info("\nKnowledge Base Information:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


def example_4_custom_chatbot():
    """Example 4: Creating a custom chatbot instance."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Custom Chatbot Instance")
    logger.info("=" * 60)
    
    from src.chat.chatbot import MedicalChatbot
    
    # Create custom chatbot with different settings
    custom_chatbot = MedicalChatbot(
        top_k=5,              # Retrieve more documents
        max_tokens=512,       # Allow longer responses
        temperature=0.3,      # More creative responses
    )
    
    logger.info("Custom chatbot created with:")
    logger.info("  - top_k=5 (retrieve 5 documents)")
    logger.info("  - max_tokens=512 (longer responses)")
    logger.info("  - temperature=0.3 (more variety)")
    
    question = "Explain the pathophysiology of asthma."
    logger.info(f"\nQuestion: {question}")
    
    response = custom_chatbot.generate_response(question)
    logger.info(f"\nAnswer:\n{response.answer}")
    logger.info(f"\nResponse time: {response.total_time:.2f}s")


def example_5_error_handling():
    """Example 5: Error handling."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Error Handling")
    logger.info("=" * 60)
    
    chatbot = load_chatbot()
    
    # Test with empty question
    logger.info("\nTesting with empty question...")
    try:
        response = chatbot.generate_response("")
        logger.warning("Empty question was accepted (implementation allows it)")
    except Exception as e:
        logger.info(f"Exception caught: {type(e).__name__}: {e}")
    
    # Test with very long question
    logger.info("\nTesting with very long question...")
    long_question = "What are " + "the " * 100 + "complications?"
    response = chatbot.generate_response(long_question)
    logger.info(f"Long question handled. Response time: {response.total_time:.2f}s")
    
    # Test with special characters
    logger.info("\nTesting with special characters...")
    special_question = "What is COVID-19 (SARS-CoV-2)? #virus @health"
    response = chatbot.generate_response(special_question)
    logger.info(f"Special characters handled. Response time: {response.total_time:.2f}s")


def example_6_embedding_cache():
    """Example 6: Embedding cache efficiency."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Embedding Cache Efficiency")
    logger.info("=" * 60)
    
    import time
    chatbot = load_chatbot()
    
    question = "What is the treatment for pneumonia?"
    
    # First call (not cached)
    logger.info(f"\nFirst call (not cached): {question}")
    start = time.time()
    response1 = chatbot.generate_response(question)
    time1 = time.time() - start
    logger.info(f"Time: {time1:.3f}s")
    
    # Second call with same question (cached)
    logger.info(f"\nSecond call (cached): {question}")
    start = time.time()
    response2 = chatbot.generate_response(question)
    time2 = time.time() - start
    logger.info(f"Time: {time2:.3f}s")
    
    logger.info(f"\nCache benefit: {time1/time2:.1f}x faster (compared to first call)")


def example_7_api_usage():
    """Example 7: Using the chatbot via REST API (async example)."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 7: REST API Usage Guide")
    logger.info("=" * 60)
    
    logger.info("""
To use the chatbot via REST API:

1. Start the FastAPI server:
   python -m uvicorn src.api.main:app --reload --port 8000

2. Make a request to the /chat endpoint:
   POST http://localhost:8000/chat
   Content-Type: application/json
   
   {
     "message": "What is pneumonia?"
   }

3. Response example:
   {
     "answer": "Pneumonia is an infection that inflames the air sacs...",
     "sources": [
       {
         "text": "Pneumonia is characterized by inflammation...",
         "metadata": {}
       }
     ],
     "query": "What is pneumonia?",
     "retrieval_time": 0.15,
     "llm_time": 0.92,
     "total_time": 1.07
   }

4. Get knowledge base stats:
   GET http://localhost:8000/chat/stats

5. Use API documentation:
   http://localhost:8000/docs (Swagger UI)
   http://localhost:8000/redoc (ReDoc)
    """)


def main():
    """Run all examples."""
    try:
        # Run examples
        example_1_basic_usage()
        example_2_multiple_questions()
        example_3_knowledge_base_stats()
        example_4_custom_chatbot()
        example_5_error_handling()
        example_6_embedding_cache()
        example_7_api_usage()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ All examples completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
