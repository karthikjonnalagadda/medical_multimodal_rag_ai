"""Medical AI Chatbot Page - Text-only medical conversation with RAG + Groq LLM.

Fast, lightweight Streamlit interface for medical Q&A.
Features:
- Real-time chat interface
- Streaming responses
- Source attribution
- Performance metrics
- Response time tracking
"""

import streamlit as st
import logging
from datetime import datetime

from src.chat.chatbot import load_chatbot

logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Medical AI Chatbot",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better chat UI
st.markdown("""
<style>
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 4px solid #1976d2;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 4px solid #666;
    }
    
    .source-box {
        background-color: #fff9e6;
        padding: 10px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 3px solid #fbc02d;
        font-size: 0.9em;
    }
    
    .response-metrics {
        background-color: #f0f0f0;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85em;
        color: #666;
        margin-top: 8px;
    }
    
    .disclaimer {
        background-color: #fff3e0;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 16px 0;
        font-size: 0.9em;
        color: #e65100;
    }
    
    .empty-state {
        text-align: center;
        color: #999;
        padding: 40px 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading medical chatbot...")
def get_chatbot():
    """Load the medical chatbot instance with caching.
    
    This is cached at the session level to avoid reloading embeddings
    and the vector store on each interaction.
    """
    return load_chatbot()


def render_chat_message(role: str, content: str):
    """Render a single chat message with styling.
    
    Args:
        role: "user" or "assistant"
        content: Message content
    """
    if role == "user":
        st.markdown(f'<div class="user-message"><b>You:</b> {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message"><b>Medical AI:</b> {content}</div>', unsafe_allow_html=True)


def render_sources(sources: list):
    """Render source documents with metadata.
    
    Args:
        sources: List of source documents with text and metadata
    """
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)})"):
        for idx, source in enumerate(sources, 1):
            st.markdown(f'<div class="source-box"><b>Source {idx}:</b><br>{source["text"]}</div>', unsafe_allow_html=True)
            if source.get("metadata"):
                st.caption(f"Metadata: {source['metadata']}")


def render_response_metrics(response):
    """Render response time metrics.
    
    Args:
        response: ChatResponse object with timing information
    """
    metrics_html = f"""
    <div class="response-metrics">
        ⏱️ Retrieval: {response['retrieval_time']:.3f}s | 
        LLM: {response['llm_time']:.3f}s | 
        Total: {response['total_time']:.3f}s
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)


def main():
    """Main Medical Chatbot page."""
    
    # Page Header
    st.title(":speech_balloon: Medical AI Chatbot")
    st.markdown("Ask medical questions and get AI-powered insights from our knowledge base.")
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Educational Information Disclaimer:</b> This chatbot provides educational medical information and is NOT a substitute for professional medical advice. Always consult with qualified healthcare professionals for diagnosis, treatment, or medical decisions. For emergencies, contact emergency services immediately.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "kb_stats" not in st.session_state:
        st.session_state.kb_stats = None
    
    # Get chatbot instance
    try:
        chatbot = get_chatbot()
        
        # Get knowledge base stats (cached in session)
        if st.session_state.kb_stats is None:
            st.session_state.kb_stats = chatbot.get_knowledge_base_stats()
        
        # Knowledge base info in sidebar
        with st.sidebar:
            st.header("📊 Knowledge Base")
            stats = st.session_state.kb_stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get("total_documents", "N/A"))
            with col2:
                st.metric("Status", "Ready ✅")
            
            st.divider()
            
            st.subheader("Embedding Model")
            st.caption(stats.get("embedding_model", "N/A"))
            
            st.subheader("Vector Store")
            st.caption(stats.get("vector_store_backend", "N/A"))
            
            st.divider()
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
    except Exception as e:
        st.error(f"Failed to load chatbot: {str(e)}")
        logger.error(f"Chatbot loading error: {e}", exc_info=True)
        return
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
            <h3>👋 Welcome to Medical AI Chatbot</h3>
            <p>Ask me anything about medical topics, symptoms, treatments, conditions, and more.</p>
            <br>
            <b>Example questions:</b><br>
            • What are the symptoms of pneumonia?<br>
            • How is diabetes treated?<br>
            • What causes chest pain?<br>
            • What is hypertension?
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.chat_history:
            render_chat_message(message["role"], message["content"])
            
            # Display sources and metrics for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])
            
            if message["role"] == "assistant" and message.get("metrics"):
                render_response_metrics(message["metrics"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input section with columns for better layout
    st.divider()
    
    # Chat input
    user_input = st.chat_input(
        placeholder="Ask a medical question...",
        key="chat_input"
    )
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Show user message
        render_chat_message("user", user_input)
        
        # Generate response
        with st.spinner("🤖 Generating response..."):
            try:
                response = chatbot.generate_response(user_input)
                
                # Add assistant message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "metrics": {
                        "retrieval_time": response.retrieval_time,
                        "llm_time": response.llm_time,
                        "total_time": response.total_time,
                    },
                    "timestamp": datetime.now().isoformat(),
                })
                
                # Show assistant response
                render_chat_message("assistant", response.answer)
                
                # Show sources
                if response.sources:
                    render_sources(response.sources)
                
                # Show metrics
                render_response_metrics({
                    "retrieval_time": response.retrieval_time,
                    "llm_time": response.llm_time,
                    "total_time": response.total_time,
                })
                
                # Performance feedback
                if response.total_time > 2.0:
                    st.warning(f"⏱️ Response took {response.total_time:.2f}s (slower than target 1-2s)")
                else:
                    st.success(f"⚡ Fast response: {response.total_time:.2f}s")
                
                # Rerun to show messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.error(f"Response generation error: {e}", exc_info=True)
                
                # Remove failed message from history
                st.session_state.chat_history.pop()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.85em; margin-top: 20px;'>
        <p>Medical AI Chatbot • Powered by Groq LLM + RAG • Built with ❤️</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
