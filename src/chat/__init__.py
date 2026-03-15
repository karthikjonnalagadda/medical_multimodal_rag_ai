"""Medical Chatbot Module

Fast, lightweight text-only medical conversation using RAG and Groq LLM.
"""

from .chatbot import MedicalChatbot, ChatResponse, load_chatbot

__all__ = ["MedicalChatbot", "ChatResponse", "load_chatbot"]
