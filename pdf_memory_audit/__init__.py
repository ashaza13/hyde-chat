"""
PDF Memory approach for answering audit questions using the full document as context.

This module provides LangGraph-based workflow implementation for audit question answering
with improved state management and tool integration capabilities.
"""

from .core.langgraph_audit_qa import PDFMemoryLangGraphAuditQA
from .core.models import AuditQuestion, AuditResponse, AnswerType

# Make LangGraph implementation the default
AuditQA = PDFMemoryLangGraphAuditQA

__all__ = ["AuditQA", "AuditQuestion", "AuditResponse", "AnswerType"] 