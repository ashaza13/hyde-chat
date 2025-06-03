import json
import re
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path

from common.core import BedrockClient, BedrockModelConfig
from textract_processor import TextChunk

class AuditQA:
    """
    Main class for answering audit questions using the full PDF document as context.
    """
    
    def __init__(
        self,
        aws_region: str = "us-gov-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        model_config: Optional[BedrockModelConfig] = None
    ):
        """
        Initialize the AuditQA service.
        
        Args:
            aws_region: AWS region to use
            aws_access_key_id: AWS access key ID (optional if using IAM roles)
            aws_secret_access_key: AWS secret access key (optional if using IAM roles)
            aws_session_token: AWS session token (optional, used for temporary credentials)
            model_config: Configuration for the Bedrock model
        """
        # Initialize the Bedrock client
        self.bedrock_client = BedrockClient(
            aws_region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            model_config=model_config
        )
        
        # Document text storage
        self.document_text = ""
        self.chunks_with_metadata = []
        self.has_metadata = False
    
    def set_document_text(self, document_text: str) -> None:
        """
        Set the document text to use as context.
        
        Args:
            document_text: The full text of the document
        """
        self.document_text = document_text
        # Clear metadata when setting plain text
        self.chunks_with_metadata = []
        self.has_metadata = False
    
    def set_document_chunks_with_metadata(self, chunks: List[TextChunk]) -> None:
        """
        Set document chunks with page metadata.
        
        Args:
            chunks: List of TextChunk objects with page metadata
        """
        self.chunks_with_metadata = chunks
        self.has_metadata = True
        # Also set the document text for backward compatibility
        self.document_text = "\n\n".join(chunk.text for chunk in chunks)
    
    def answer_question(self, question_text: str, context: Optional[str] = None) -> 'AuditResponse':
        """
        Answer an audit question using the full document as context.
        
        Args:
            question_text: The audit question text
            context: Optional additional context for the question
            
        Returns:
            An AuditResponse object with the answer, confidence, and explanation
        """
        # Import here to avoid circular imports
        from .models import AuditQuestion, AuditResponse
        
        # Create question model
        question = AuditQuestion(question=question_text, context=context)
        
        # Use metadata-aware processing if available
        if self.has_metadata and self.chunks_with_metadata:
            return self._answer_with_metadata(question)
        else:
            # Fallback to original approach
            document_text = self.document_text
            
            # Combine provided context with document text
            if question.context:
                full_context = f"{question.context}\n\n{document_text}"
            else:
                full_context = document_text
            
            # Get the final answer using the LLM
            answer = self._get_final_answer(question.question, full_context)
            return answer
    
    def _answer_with_metadata(self, question) -> 'AuditResponse':
        """
        Answer a question using chunks with page metadata.
        
        Args:
            question: The audit question
            
        Returns:
            An AuditResponse object with page citations
        """
        # For memory approach, use all chunks but with citation support
        context_parts = []
        citations = []
        page_references = set()
        
        if question.context:
            context_parts.append(f"Additional Context: {question.context}")
        
        # Add all chunks with citations (memory approach uses full document)
        for i, chunk in enumerate(self.chunks_with_metadata):
            citation_id = i + 1
            context_parts.append(f"[{citation_id}] {chunk.text}")
            
            citations.append({
                "citation_id": citation_id,
                "page_range": chunk.get_page_range_str(),
                "pages": chunk.page_numbers
            })
            
            page_references.update(chunk.page_numbers)
        
        full_context = "\n\n".join(context_parts)
        
        # Get the final answer with citation instructions
        answer = self._get_final_answer_with_citations(
            question.question, 
            full_context, 
            citations
        )
        
        return answer
    
    def _get_final_answer_with_citations(
        self, 
        question: str, 
        context: str, 
        citations: List[Dict[str, Any]]
    ) -> 'AuditResponse':
        """
        Get the final answer from the LLM with citation support.
        
        Args:
            question: The audit question
            context: The context with citation markers
            citations: List of citation information
            
        Returns:
            An AuditResponse object with page citations in the explanation
        """
        # Import here to avoid circular imports
        from .models import AuditResponse, AnswerType
        
        # Build citation reference for the prompt
        citation_info = ""
        if citations:
            citation_lines = []
            for citation in citations:
                citation_lines.append(f"[{citation['citation_id']}] = {citation['page_range']}")
            citation_info = f"\n\nCitation References:\n" + "\n".join(citation_lines)
            
        prompt = f"""You are an expert auditor specializing in higher education financial statements. You are tasked with answering questions about financial audits with ONLY "Yes", "No", or "N/A" (if the question is not applicable or cannot be determined from the available information).

I will provide you with a question and the full context from financial statements or audit reports. The context includes numbered citations [1], [2], etc. that correspond to specific page references.{citation_info}

You must first analyze the context and then provide your answer in the following JSON format:
{{
  "answer": "Yes" | "No" | "N/A",
  "confidence": <float between 0 and 1>,
  "explanation": "<detailed explanation supporting your answer, including page references when citing specific information>"
}}

When referencing specific information in your explanation, include the citation number and page reference, for example: "Based on information from [1] (Page 15), the financial performance shows..."

Question: {question}

Context:
{context if context else "No specific context provided"}

Provide your response in the exact JSON format specified above. Your explanation should be detailed and include page references when citing specific information from the context.
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
    
    def _get_final_answer(self, question: str, context: str) -> 'AuditResponse':
        """
        Get the final answer from the LLM using the question and context.
        
        Args:
            question: The audit question
            context: The context for answering the question
            
        Returns:
            An AuditResponse object
        """
        # Import here to avoid circular imports
        from .models import AuditResponse, AnswerType
        
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
    
    def _parse_unstructured_response(self, response_text: str) -> 'AuditResponse':
        """
        Attempt to parse an unstructured response into an AuditResponse.
        
        Args:
            response_text: The unstructured response text
            
        Returns:
            An AuditResponse object
        """
        # Import here to avoid circular imports
        from .models import AuditResponse, AnswerType
        
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