"""
Hypothetical Document Embeddings (HyDE) approach for answering audit questions.
"""
__version__ = "0.1.0"

from .core.models import AuditResponse, AuditQuestion
from .core.audit_qa import AuditQA

__all__ = ["AuditQA", "AuditResponse", "AuditQuestion"]
