"""
LangGraph-based PDF Memory audit implementation.
"""

import json
import re
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path

from common.core import BaseLangGraphAuditWorkflow, AuditWorkflowState, ChromaVectorStore, BedrockModelConfig
from textract_processor import TextChunk
from .models import AuditResponse, AnswerType

class PDFMemoryLangGraphAuditQA(BaseLangGraphAuditWorkflow):
    """
    LangGraph-based PDF Memory audit QA implementation that uses the full document as context.
    """
    
    def __init__(
        self,
        aws_region: str = "us-gov-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        model_config: Optional[BedrockModelConfig] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the PDF Memory LangGraph workflow.
        
        Args:
            aws_region: AWS region for Bedrock
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_session_token: AWS session token
            model_config: Bedrock model configuration
            vector_store: Vector store instance
            embedding_model_name: Embedding model name
        """
        super().__init__(
            aws_region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            model_config=model_config,
            vector_store=vector_store,
            embedding_model_name=embedding_model_name
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
        
        # Clear vector store when setting plain text
        if self.vector_store:
            try:
                self.vector_store.clear()
            except Exception as e:
                print(f"Warning: Could not clear vector store: {e}")
    
    def set_document_chunks_with_metadata(self, chunks: List[TextChunk]) -> None:
        """
        Set document chunks with page metadata and add them to the vector store.
        
        Args:
            chunks: List of TextChunk objects with page metadata
        """
        self.chunks_with_metadata = chunks
        self.has_metadata = True
        # Also set the document text for backward compatibility
        self.document_text = "\n\n".join(chunk.text for chunk in chunks)
        
        # Clear and re-populate vector store for consistency (even though Memory approach uses all chunks)
        if self.vector_store:
            try:
                # Clear existing data to avoid stale chunks
                self.vector_store.clear()
                # Add new chunks
                if chunks:
                    self.vector_store.add_chunks(chunks)
            except Exception as e:
                print(f"Warning: Could not update vector store: {e}")
    
        def answer_question(self, question_text: str, context: Optional[str] = None) -> 'AuditResponse':
        """
        Answer an audit question using the PDF Memory LangGraph workflow.
        
        Args:
            question_text: The audit question text
            context: Optional additional context for the question
             
        Returns:
            An AuditResponse object with the answer, confidence, and explanation
        """
        # Import here to avoid circular imports
        from .models import AuditResponse, AnswerType
        
        # Run the workflow
        results = self.run_workflow(
            question=question_text,
            context=context,
            document_chunks=getattr(self, 'chunks_with_metadata', [])
        )
        
        # Convert to AuditResponse format
        answer_type = AnswerType.YES if results["answer"] == "Yes" else \
                     AnswerType.NO if results["answer"] == "No" else \
                     AnswerType.NA
        
        return AuditResponse(
            answer=answer_type,
            confidence=results.get("confidence", 0.0),
            explanation=results.get("explanation", ""),
            page_references=results.get("page_references", [])
        )
    
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
        from langchain.schema import HumanMessage
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        response_text = response.content
        
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
        from langchain.schema import HumanMessage
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        response_text = response.content
        
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

    def _build_workflow(self) -> StateGraph:
        """
        Build the PDF Memory-specific workflow graph.
        
        Returns:
            StateGraph: The compiled workflow graph
        """
        workflow = StateGraph(AuditWorkflowState)
        
        # Add nodes for PDF Memory approach (uses all chunks)
        workflow.add_node("process_question", self._process_question)
        workflow.add_node("retrieve_all_context", self._retrieve_all_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("format_response", self._format_response)
        
        # Add edges for PDF Memory workflow
        workflow.set_entry_point("process_question")
        workflow.add_edge("process_question", "retrieve_all_context")
        workflow.add_edge("retrieve_all_context", "generate_answer")
        workflow.add_edge("generate_answer", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow
    
    def _retrieve_all_context(self, state: AuditWorkflowState) -> AuditWorkflowState:
        """
        Retrieve all document chunks as context (PDF Memory approach).
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        # PDF Memory approach uses all chunks for comprehensive context
        state["retrieved_chunks"] = state.get("document_chunks", [])
        state["next_action"] = "generate_answer"
        
        return state
    
    def _create_answer_prompt(self, question: str, context: str) -> str:
        """
        Create a PDF Memory-specific prompt for answer generation.
        
        Args:
            question: The audit question
            context: The context with citations
            
        Returns:
            Formatted prompt string
        """
        return f"""You are an expert auditor specializing in higher education financial statements. You are tasked with answering questions about financial audits with ONLY "Yes", "No", or "N/A" (if the question is not applicable or cannot be determined from the available information).

This analysis uses the PDF Memory approach, where the full document content is provided as context to ensure comprehensive coverage.

I will provide you with a question and the complete context from financial statements or audit reports. The context includes numbered citations [1], [2], etc. that correspond to specific page references.

You must provide your answer in the following JSON format:
{{
  "answer": "Yes" | "No" | "N/A",
  "confidence": <float between 0 and 1>,
  "explanation": "<detailed explanation supporting your answer, including page references when citing specific information>"
}}

Question: {question}

Context:
{context if context else "No specific context provided"}

Provide your response in the exact JSON format specified above. Your explanation should reference the page citations when discussing specific information.""" 