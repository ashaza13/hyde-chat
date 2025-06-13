import json
import re
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path

from common.core import BedrockModelConfig
from textract_processor import TextChunk
from .models import AuditQuestion, AuditResponse, AnswerType
from langgraph.graph import StateGraph, END
from common.core import BaseLangGraphAuditWorkflow, AuditWorkflowState, ChromaVectorStore


class HyDELangGraphAuditQA(BaseLangGraphAuditWorkflow):
    """
    LangGraph-based HyDE audit QA implementation using hypothetical document embeddings.
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
        Initialize the HyDE LangGraph workflow.
        
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
        
        # Clear and re-populate vector store for semantic search
        if self.vector_store:
            try:
                # Clear existing data to avoid stale chunks
                self.vector_store.clear()
                # Add new chunks
                if chunks:
                    self.vector_store.add_chunks(chunks)
            except Exception as e:
                print(f"Warning: Could not update vector store: {e}")
    
    def get_relevant_chunks_for_hypothetical(self, hypothetical_doc: str, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        """
        Get the most relevant chunks for a hypothetical document using vector store semantic search.
        
        Args:
            hypothetical_doc: The hypothetical document to find relevant chunks for
            top_k: Number of top chunks to return
            
        Returns:
            List of tuples with (TextChunk, relevance_score)
        """
        # Use vector store for semantic search if available
        if self.vector_store and self.has_metadata:
            try:
                return self.vector_store.similarity_search(hypothetical_doc, k=top_k)
            except Exception as e:
                print(f"Vector search failed, falling back to keyword matching: {e}")
        
        # Fallback to keyword-based matching if vector store is not available
        if not self.chunks_with_metadata:
            return []
        
        # Simple keyword-based relevance scoring as fallback
        hyp_words = set(hypothetical_doc.lower().split())
        scored_chunks = []
        
        for chunk in self.chunks_with_metadata:
            chunk_words = set(chunk.text.lower().split())
            # Score based on word overlap
            overlap = len(hyp_words.intersection(chunk_words))
            score = overlap / len(hyp_words) if hyp_words else 0
            scored_chunks.append((chunk, score))
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
    
    def generate_hypothetical_document(self, question: str) -> str:
        """
        Generate a hypothetical document that would answer the question.
        This is the core of the HyDE approach.
        
        Args:
            question: The audit question
            
        Returns:
            A hypothetical document that answers the question
        """
        # Craft prompt for generating hypothetical document
        prompt = f"""You are an expert auditor specializing in higher education financial statements.
        
I'm going to ask you a question about a financial audit, and I'd like you to generate a hypothetical extract from a financial statement or audit report that would help answer this question.
        
The extract should be detailed, specific, and realistic - as if it came from an actual higher education institution's financial statement.
        
Question: {question}
        
Generate a detailed, realistic extract from a financial statement or audit report that would help answer this question:"""
        
        # Generate the hypothetical document
        from langchain.schema import HumanMessage
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        hypothetical_doc = response.content
        return hypothetical_doc
    
    def answer_question(self, question_text: str, context: Optional[str] = None) -> AuditResponse:
        """
        Answer an audit question using the HYDE LangGraph workflow.
        
        Args:
            question_text: The audit question text
            context: Optional additional context for the question
            
        Returns:
            An AuditResponse object with the answer, confidence, and explanation
        """
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
    
    def _answer_with_metadata(self, question: AuditQuestion, hypothetical_doc: str) -> AuditResponse:
        """
        Answer a question using chunks with page metadata and HyDE approach.
        
        Args:
            question: The audit question
            hypothetical_doc: The generated hypothetical document
            
        Returns:
            An AuditResponse object with page citations
        """
        # Get relevant chunks based on hypothetical document
        relevant_chunks = self.get_relevant_chunks_for_hypothetical(hypothetical_doc, top_k=3)
        
        if not relevant_chunks:
            # No relevant chunks found, use first few chunks
            relevant_chunks = [(chunk, 0.0) for chunk in self.chunks_with_metadata[:3]]
        
        # Build context with citations
        context_parts = []
        citations = []
        page_references = set()
        
        if question.context:
            context_parts.append(f"Additional Context: {question.context}")
        
        context_parts.append(f"Hypothetical Document (for reference): {hypothetical_doc}")
        
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
            citations
        )
        
        return answer
    
    def _get_final_answer_with_citations(
        self, 
        question: str, 
        context: str, 
        citations: List[Dict[str, Any]]
    ) -> AuditResponse:
        """
        Get the final answer from the LLM with citation support.
        
        Args:
            question: The audit question
            context: The context with citation markers
            citations: List of citation information
            
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
            
        prompt = f"""You are an expert auditor specializing in higher education financial statements. You are tasked with answering questions about financial audits with ONLY "Yes", "No", or "N/A" (if the question is not applicable or cannot be determined from the available information).

I will provide you with a question, a hypothetical document (generated using HyDE approach), and relevant context from financial statements or audit reports. The context includes numbered citations [1], [2], etc. that correspond to specific page references.{citation_info}

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

    def _build_workflow(self) -> StateGraph:
        """
        Build the HYDE-specific workflow graph.
        
        Returns:
            StateGraph: The compiled workflow graph
        """
        workflow = StateGraph(AuditWorkflowState)
        
        # Add nodes specific to HYDE approach
        workflow.add_node("process_question", self._process_question)
        workflow.add_node("generate_hypothetical_doc", self._generate_hypothetical_doc)
        workflow.add_node("retrieve_context", self._retrieve_context_hyde)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("format_response", self._format_response)
        
        # Add edges for HYDE workflow
        workflow.set_entry_point("process_question")
        workflow.add_edge("process_question", "generate_hypothetical_doc")
        workflow.add_edge("generate_hypothetical_doc", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow
    
    def _generate_hypothetical_doc(self, state: AuditWorkflowState) -> AuditWorkflowState:
        """
        Generate a hypothetical document that would answer the question.
        This is the core of the HYDE approach.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            # Create prompt for generating hypothetical document
            prompt = f"""You are an expert auditor specializing in higher education financial statements.

I'm going to ask you a question about a financial audit, and I'd like you to generate a hypothetical extract from a financial statement or audit report that would help answer this question.

The extract should be detailed, specific, and realistic - as if it came from an actual higher education institution's financial statement.

Question: {state['question']}

Generate a detailed, realistic extract from a financial statement or audit report that would help answer this question:"""

            # Generate the hypothetical document
            from langchain.schema import HumanMessage
            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            hypothetical_doc = response.content
            state["hypothetical_doc"] = hypothetical_doc
            state["next_action"] = "retrieve_context"
            
        except Exception as e:
            state["error"] = f"Error generating hypothetical document: {str(e)}"
            state["hypothetical_doc"] = None
        
        return state
    
    def _retrieve_context_hyde(self, state: AuditWorkflowState) -> AuditWorkflowState:
        """
        Retrieve relevant context using the hypothetical document for better retrieval.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            if state.get("hypothetical_doc") and state.get("document_chunks"):
                # Use hypothetical document for vector search
                search_results = self.vector_store.similarity_search(
                    query=state["hypothetical_doc"],
                    k=3
                )
                
                # Extract chunks and scores
                retrieved_chunks = []
                for chunk, score in search_results:
                    retrieved_chunks.append(chunk)
                
                state["retrieved_chunks"] = retrieved_chunks
            else:
                # Fallback: use question directly for search or use all chunks
                if state.get("document_chunks"):
                    search_results = self.vector_store.similarity_search(
                        query=state["question"],
                        k=3
                    )
                    state["retrieved_chunks"] = [chunk for chunk, score in search_results]
                else:
                    state["retrieved_chunks"] = []
            
            state["next_action"] = "generate_answer"
            
        except Exception as e:
            # Fallback to using all chunks if vector search fails
            state["retrieved_chunks"] = state.get("document_chunks", [])[:3]
            state["error"] = f"Vector search failed, using fallback: {str(e)}"
        
        return state
    
    def _create_answer_prompt(self, question: str, context: str) -> str:
        """
        Create a HYDE-specific prompt for answer generation.
        
        Args:
            question: The audit question
            context: The context with citations
            
        Returns:
            Formatted prompt string
        """
        return f"""You are an expert auditor specializing in higher education financial statements. You are tasked with answering questions about financial audits with ONLY "Yes", "No", or "N/A" (if the question is not applicable or cannot be determined from the available information).

This analysis uses the HYDE (Hypothetical Document Embeddings) approach, where relevant context was retrieved based on a hypothetical document that would answer your question.

I will provide you with a question and context from financial statements or audit reports. The context includes numbered citations [1], [2], etc. that correspond to specific page references.

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