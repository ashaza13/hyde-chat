import json
import re
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from .bedrock_client import BedrockClient
from .document_processor import DocumentProcessor
from .models import AuditQuestion, AuditResponse, AnswerType, BedrockModelConfig


class AuditQA:
    """
    Main class for answering audit questions using the full PDF document as context.
    """
    
    def __init__(
        self,
        aws_region: str = "us-gov-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        model_config: Optional[BedrockModelConfig] = None
    ):
        """
        Initialize the AuditQA service.
        
        Args:
            aws_region: AWS region to use
            aws_access_key_id: AWS access key ID (optional if using IAM roles)
            aws_secret_access_key: AWS secret access key (optional if using IAM roles)
            model_config: Configuration for the Bedrock model
        """
        # Initialize the Bedrock client
        self.bedrock_client = BedrockClient(
            aws_region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            model_config=model_config
        )
        
        # Initialize the document processor
        self.document_processor = DocumentProcessor()
    
    def load_document(self, bucket_name: str, key: str) -> bool:
        """
        Load a document for analysis.
        
        Args:
            bucket_name: Name of the S3 bucket
            key: Key of the PDF file in the S3 bucket
            
        Returns:
            True if successful, False otherwise
        """
        return self.document_processor.load_document(bucket_name, key)
    
    def answer_question(self, question_text: str, context: Optional[str] = None) -> AuditResponse:
        """
        Answer an audit question using the full document as context.
        
        Args:
            question_text: The audit question text
            context: Optional additional context for the question
            
        Returns:
            An AuditResponse object with the answer, confidence, and explanation
        """
        # Create question model
        question = AuditQuestion(question=question_text, context=context)
        
        # Get the full document text
        document_text = self.document_processor.get_document_text()
        
        # Combine provided context with document text
        if question.context:
            full_context = f"{question.context}\n\n{document_text}"
        else:
            full_context = document_text
        
        # Get the final answer using the LLM
        answer = self._get_final_answer(question.question, full_context)
        return answer
    
    def _get_final_answer(self, question: str, context: str) -> AuditResponse:
        """
        Get the final answer from the LLM using the question and context.
        
        Args:
            question: The audit question
            context: The context for answering the question
            
        Returns:
            An AuditResponse object
        """
        # Craft prompt for the final answer
        prompt = f"""You are an expert auditor specializing in higher education financial statements. You are tasked with answering questions about financial audits with ONLY "Yes", "No", or "N/A" (if the question is not applicable or cannot be determined from the available information).

I will provide you with a question and relevant context from financial statements or audit reports.

You must first analyze the context and then provide your answer in the following JSON format:
{{
  "answer": "Yes" | "No" | "N/A",
  "confidence": <float between 0 and 1>,
  "explanation": "<detailed explanation supporting your answer>"
}}

Question: {question}

Context:
{context if context else "No specific context provided"}

Provide your response in the exact JSON format specified above. Your explanation should be detailed and reference specific parts of the context when applicable.
"""
        
        # Get response from LLM
        response_text = self.bedrock_client.invoke_model(prompt)
        
        # Parse the JSON response
        try:
            # Extract JSON object using regex (in case there's other text)
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
                response_dict = json.loads(json_str)
                
                # Create AuditResponse object
                answer = AnswerType(response_dict.get("answer", "N/A"))
                confidence = float(response_dict.get("confidence", 0.5))
                explanation = response_dict.get("explanation", "No explanation provided")
                
                return AuditResponse(
                    answer=answer,
                    confidence=confidence,
                    explanation=explanation
                )
            else:
                # Fallback if JSON cannot be extracted
                return self._parse_unstructured_response(response_text)
                
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback for parsing errors
            return self._parse_unstructured_response(response_text)
    
    def _parse_unstructured_response(self, response_text: str) -> AuditResponse:
        """
        Attempt to parse an unstructured response into an AuditResponse.
        
        Args:
            response_text: The unstructured response text
            
        Returns:
            An AuditResponse object
        """
        # Look for Yes/No/N/A in the response
        response_lower = response_text.lower()
        
        if "yes" in response_lower:
            answer = AnswerType.YES
        elif "no" in response_lower:
            answer = AnswerType.NO
        else:
            answer = AnswerType.NA
        
        # Default confidence
        confidence = 0.5
        
        # Use the response text as explanation
        explanation = response_text
        
        return AuditResponse(
            answer=answer,
            confidence=confidence,
            explanation=explanation
        ) 