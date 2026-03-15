"""FastAPI endpoint for the Medical Chatbot.

Fast, lightweight endpoint for text-only medical chat.
POST /chat - Generate medical chatbot responses with RAG + Groq.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from src.chat.chatbot import load_chatbot

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request model for chatbot endpoint."""
    
    message: str = Field(..., min_length=1, max_length=1000, description="User's medical question")


class ChatSource(BaseModel):
    """Source document from chat response."""
    
    text: str = Field(..., description="Excerpt from source document")
    metadata: dict = Field(default_factory=dict, description="Document metadata")


class ChatApiResponse(BaseModel):
    """Response model for chatbot endpoint."""
    
    answer: str = Field(..., description="AI chatbot response")
    sources: list[ChatSource] = Field(default_factory=list, description="Retrieved source documents")
    query: str = Field(..., description="Original user query")
    retrieval_time: float = Field(default=0.0, description="Time spent retrieving documents (seconds)")
    llm_time: float = Field(default=0.0, description="Time spent generating response (seconds)")
    total_time: float = Field(default=0.0, description="Total response time (seconds)")
    structured: Optional[dict] = Field(default=None, description="Structured answer with sections")
    clinical: Optional[dict] = Field(
        default=None,
        description="Strict clinical JSON: diagnosis/confidence/possible_conditions/explanation/recommended_tests/next_steps",
    )


def setup_chat_routes(app):
    """Register chat routes to a FastAPI app.
    
    Args:
        app: FastAPI application instance
        
    Routes:
        POST /chat - Generate a chatbot response
        GET /chat/stats - Get knowledge base statistics
    """
    
    @app.post("/chat", response_model=ChatApiResponse, tags=["Chat"])
    async def chat(request: ChatRequest):
        """Generate a medical chatbot response.
        
        Fast, text-only medical question answering using RAG + Groq LLM.
        
        Request:
            message (str): User's medical question (e.g., "What are symptoms of pneumonia?")
        
        Response:
            answer (str): AI-generated medical explanation
            sources (list): Retrieved knowledge base documents
            query (str): Original query
            timing: Retrieval, LLM, and total response times
        
        Example:
            POST /chat
            {
              "message": "What causes chest pain?"
            }
        
        Response:
            200 OK
            {
              "answer": "Chest pain can have multiple causes...",
              "sources": [
                {
                  "text": "Chest pain is a symptom...",
                  "metadata": {...}
                }
              ],
              "query": "What causes chest pain?",
              "retrieval_time": 0.12,
              "llm_time": 0.88,
              "total_time": 1.05
            }
        """
        try:
            logger.info(f"Chat request: {request.message[:100]}...")
            
            # Load chatbot instance
            chatbot = load_chatbot()
            
            # Generate response
            chat_response = chatbot.generate_response(request.message)
            
            logger.info(f"Chat response generated in {chat_response.total_time:.2f}s")
            
            # Convert to API response
            return ChatApiResponse(
                answer=chat_response.answer,
                sources=[
                    ChatSource(
                        text=source["text"],
                        metadata=source["metadata"]
                    )
                    for source in chat_response.sources
                ],
                query=chat_response.query,
                retrieval_time=chat_response.retrieval_time,
                llm_time=chat_response.llm_time,
                total_time=chat_response.total_time,
                structured=chat_response.structured,
                clinical=chat_response.clinical,
            )
            
        except Exception as e:
            logger.error(f"Chat endpoint error: {e}", exc_info=True)
            return ChatApiResponse(
                answer=f"Error processing your question. Please try again. Error: {str(e)}",
                sources=[],
                query=request.message,
                retrieval_time=0.0,
                llm_time=0.0,
                total_time=0.0,
            )
    
    @app.get("/chat/stats", tags=["Chat"])
    async def chat_stats():
        """Get medical knowledge base statistics.
        
        Returns information about the loaded knowledge base.
        
        Response:
            {
              "total_documents": 1500,
              "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
              "vector_store_backend": "ChromaVectorStore"
            }
        """
        try:
            chatbot = load_chatbot()
            stats = chatbot.get_knowledge_base_stats()
            
            return {
                **stats,
                "status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"Chat stats error: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
