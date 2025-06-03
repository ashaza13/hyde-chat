import json
import re
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path

from common.core import BedrockClient, BedrockModelConfig
from textract_processor import TextChunk
from .models import AuditQuestion, AuditResponse, AnswerType


class AuditQA:
    """
    Main class for answering audit questions using the RAG (Retrieval-Augmented Generation) approach.
    """
    
    def __init__(
        self,
        aws_region: str = "us-gov-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        model_config: Optional[BedrockModelConfig] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        use_query_rewriting: bool = False
    ):
        """
        Initialize the AuditQA service.
        
        Args:
            aws_region: AWS region to use
            aws_access_key_id: AWS access key ID (optional if using IAM roles)
            aws_secret_access_key: AWS secret access key (optional if using IAM roles)
            aws_session_token: AWS session token (optional, used for temporary credentials)
            model_config: Configuration for the Bedrock model
            embedding_model_name: Name of the embedding model to use
            use_query_rewriting: Whether to use query rewriting for better vector DB retrieval
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
        
        # Query rewriting flag
        self.use_query_rewriting = use_query_rewriting
    
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
    
    def get_relevant_chunks(self, query: str, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        """
        Get the most relevant chunks for a query using simple keyword matching.
        In a real implementation, this would use vector embeddings.
        
        Args:
            query: The query to find relevant chunks for
            top_k: Number of top chunks to return
            
        Returns:
            List of tuples with (TextChunk, relevance_score)
        """
        if not self.chunks_with_metadata:
            return []
        
        # Simple keyword-based relevance scoring
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.chunks_with_metadata:
            chunk_words = set(chunk.text.lower().split())
            # Score based on word overlap
            overlap = len(query_words.intersection(chunk_words))
            score = overlap / len(query_words) if query_words else 0
            scored_chunks.append((chunk, score))
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
    
    def rewrite_query_for_vector_search(self, original_query: str) -> str:
        """
        Rewrite a query to be more effective for vector database retrieval.
        
        This function uses the LLM to transform the original query into a form that's
        more conducive to semantic vector retrieval by:
        1. Expanding with relevant keywords that may appear in the target text
        2. Removing conversational elements
        3. Focusing on core information needs
        4. Adding domain-specific terminology
        
        Args:
            original_query: The original query text
            
        Returns:
            A rewritten query optimized for vector retrieval
        """
        prompt = f"""You are an expert at optimizing queries for semantic vector search retrieval in the financial auditing domain.

Given an original query, rewrite it to be more effective for retrieving relevant financial audit information from a vector database.

Guidelines for rewriting:
1. Expand with relevant financial audit terminology and keywords that might appear in financial statements
2. Remove conversational elements, questions, and filler words
3. Focus on the core information needs and entities
4. Use specific financial statement terminology where appropriate
5. Maintain all important concepts from the original query
6. Keep the rewritten query concise (typically 1-3 sentences)

Original query: {original_query}

Rewritten query for vector search (just return the rewritten query without explanation or additional text):"""

        # Get response from LLM
        rewritten_query = self.bedrock_client.invoke_model(prompt)
        
        # Clean up the response (remove quotes, extra whitespace, etc.)
        rewritten_query = rewritten_query.strip().strip('"\'')
        
        return rewritten_query
    
    def answer_question(self, question_text: str, context: Optional[str] = None, use_query_rewriting: Optional[bool] = None) -> AuditResponse:
        """
        Answer an audit question using the RAG approach.
        
        Args:
            question_text: The audit question text
            context: Optional additional context for the question
            use_query_rewriting: Whether to use query rewriting (overrides the instance setting if provided)
            
        Returns:
            An AuditResponse object with the answer, confidence, and explanation
        """
        # Create question model
        question = AuditQuestion(question=question_text, context=context)
        
        # Determine whether to use query rewriting
        should_rewrite = use_query_rewriting if use_query_rewriting is not None else self.use_query_rewriting
        
        # Rewrite the query if enabled
        search_query = question.question
        if should_rewrite:
            search_query = self.rewrite_query_for_vector_search(question.question)
            print(f"Original query: {question.question}")
            print(f"Rewritten query: {search_query}")
        
        # Use metadata-aware processing if available
        if self.has_metadata and self.chunks_with_metadata:
            return self._answer_with_metadata(question, search_query if should_rewrite else None)
        else:
            # Fallback to original approach
            document_text = self.document_text
            
            # Combine all context
            if question.context:
                full_context = f"{question.context}\n\n{document_text}"
            else:
                full_context = document_text
            
            # Get the final answer using the LLM
            answer = self._get_final_answer(question.question, full_context, search_query if should_rewrite else None)
            return answer
    
    def _answer_with_metadata(self, question: AuditQuestion, rewritten_query: Optional[str] = None) -> AuditResponse:
        """
        Answer a question using chunks with page metadata.
        
        Args:
            question: The audit question
            rewritten_query: The rewritten query used for retrieval (if query rewriting was used)
            
        Returns:
            An AuditResponse object with page citations
        """
        # Get relevant chunks
        query_text = rewritten_query if rewritten_query else question.question
        relevant_chunks = self.get_relevant_chunks(query_text, top_k=3)
        
        if not relevant_chunks:
            # No relevant chunks found, use all chunks
            relevant_chunks = [(chunk, 0.0) for chunk in self.chunks_with_metadata[:3]]
        
        # Build context with citations
        context_parts = []
        citations = []
        page_references = set()
        
        if question.context:
            context_parts.append(f"Additional Context: {question.context}")
        
        for i, (chunk, score) in enumerate(relevant_chunks):
            citation_id = i + 1
            context_parts.append(f"[{citation_id}] {chunk.text}")
            
            citations.append({
                "citation_id": citation_id,
                "page_range": chunk.get_page_range_str(),
                "pages": chunk.page_numbers,
                "confidence": score
            })
            
            page_references.update(chunk.page_numbers)
        
        full_context = "\n\n".join(context_parts)
        
        # Get the final answer with citation instructions
        answer = self._get_final_answer_with_citations(
            question.question, 
            full_context, 
            citations,
            rewritten_query
        )
        
        return answer
    
    def _get_final_answer_with_citations(
        self, 
        question: str, 
        context: str, 
        citations: List[Dict[str, Any]], 
        rewritten_query: Optional[str] = None
    ) -> AuditResponse:
        """
        Get the final answer from the LLM with citation support.
        
        Args:
            question: The audit question
            context: The context with citation markers
            citations: List of citation information
            rewritten_query: The rewritten query used for retrieval (if query rewriting was used)
            
        Returns:
            An AuditResponse object with page citations in the explanation
        """
        # Build citation reference for the prompt
        citation_info = ""
        if citations:
            citation_lines = []
            for citation in citations:
                citation_lines.append(f"[{citation['citation_id']}] = {citation['page_range']}")
            citation_info = f"\n\nCitation References:\n" + "\n".join(citation_lines)
        
        query_info = ""
        if rewritten_query:
            query_info = f"\nNote: To find the most relevant information, I rewrote your question as: '{rewritten_query}'"
            
        prompt = f"""You are an expert auditor specializing in higher education financial statements. You are tasked with answering questions about financial audits with ONLY "Yes", "No", or "N/A" (if the question is not applicable or cannot be determined from the available information).

I will provide you with a question and relevant context from financial statements or audit reports. The context includes numbered citations [1], [2], etc. that correspond to specific page references.{query_info}{citation_info}

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
    
    def _get_final_answer(self, question: str, context: str, rewritten_query: Optional[str] = None) -> AuditResponse:
        """
        Get the final answer from the LLM using the question and context.
        
        Args:
            question: The audit question
            context: The context for answering the question
            rewritten_query: The rewritten query used for retrieval (if query rewriting was used)
            
        Returns:
            An AuditResponse object
        """
        # Craft prompt for the final answer
        query_info = ""
        if rewritten_query:
            query_info = f"\nNote: To find the most relevant information, I rewrote your question as: '{rewritten_query}'"
            
        prompt = f"""You are an expert auditor specializing in higher education financial statements. You are tasked with answering questions about financial audits with ONLY "Yes", "No", or "N/A" (if the question is not applicable or cannot be determined from the available information).

I will provide you with a question and relevant context from financial statements or audit reports.{query_info}

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