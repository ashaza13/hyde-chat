"""
Hypothetical Document Embeddings (HyDE) approach for answering audit questions.

This module provides LangGraph-based workflow implementation for audit question answering
with improved state management and tool integration capabilities.
"""
__version__ = "0.1.0"

from .core.langgraph_audit_qa import HyDELangGraphAuditQA
from .core.models import AuditQuestion, AuditResponse, AnswerType

# Make LangGraph implementation the default
AuditQA = HyDELangGraphAuditQA

__all__ = ["AuditQA", "AuditQuestion", "AuditResponse", "AnswerType"]
