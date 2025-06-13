"""
RAG (Retrieval-Augmented Generation) approach for answering audit questions.

This module provides LangGraph-based workflow implementation for audit question answering
with improved state management and tool integration capabilities.
"""

from .core.audit_qa import RAGLangGraphAuditQA
from .core.models import AuditQuestion, AuditResponse, AnswerType

# Make LangGraph implementation the default
AuditQA = RAGLangGraphAuditQA

__all__ = ["AuditQA", "AuditQuestion", "AuditResponse", "AnswerType"] 