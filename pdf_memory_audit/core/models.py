from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class AnswerType(str, Enum):
    YES = "Yes"
    NO = "No"
    NA = "N/A"


class AuditQuestion(BaseModel):
    """Model for audit questions with optional context."""
    question: str = Field(..., description="The audit question to be answered")
    context: Optional[str] = Field(None, description="Additional context for the question")
    
    @validator('question')
    def question_must_be_audit_related(cls, v):
        if len(v) < 5:
            raise ValueError('Question is too short')
        return v


class AuditResponse(BaseModel):
    """Model for structured audit responses."""
    answer: AnswerType = Field(..., description="The answer must be 'Yes', 'No', or 'N/A'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    explanation: str = Field(..., description="Explanation for the provided answer")
    sources: Optional[List[str]] = Field(None, description="Sources or references supporting the answer")
    
    @validator('explanation')
    def explanation_must_be_detailed(cls, v):
        if len(v) < 10:
            raise ValueError('Explanation is too short')
        return v


class BedrockModelConfig(BaseModel):
    """Configuration for AWS Bedrock models."""
    model_id: str = Field(..., description="The Bedrock model ID to use")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Temperature for model generation")
    max_tokens: int = Field(1024, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top p sampling parameter")
    
    class Config:
        arbitrary_types_allowed = True 