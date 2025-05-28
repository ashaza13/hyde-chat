"""Core modules shared across audit approaches."""

from .bedrock_client import BedrockClient, BedrockModelConfig
from .document_processor import DocumentProcessor

__all__ = ["BedrockClient", "BedrockModelConfig", "DocumentProcessor"] 