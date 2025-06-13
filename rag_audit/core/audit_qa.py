import json
import re
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path

from common.core import BedrockModelConfig
from textract_processor import TextChunk
from .models import AuditQuestion, AuditResponse, AnswerType
from langgraph.graph import StateGraph, END
from common.core import BaseLangGraphAuditWorkflow, AuditWorkflowState, ChromaVectorStore


class RAGLangGraphAuditQA(BaseLangGraphAuditWorkflow):
    """
    LangGraph-based RAG audit QA implementation with query rewriting capabilities.
    """
    
    def __init__(
        self,
        aws_region: str = "us-gov-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        model_config: Optional[BedrockModelConfig] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        embedding_model_name: str = "amazon.titan-embed-text-v2:0",
        use_query_rewriting: bool = False
    ):
        """
        Initialize the RAG LangGraph workflow.
        
        Args:
            aws_region: AWS region for Bedrock
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_session_token: AWS session token
            model_config: Bedrock model configuration
            vector_store: Vector store instance
            embedding_model_name: Embedding model name
            use_query_rewriting: Whether to use query rewriting for better retrieval
        """
        self.use_query_rewriting = use_query_rewriting
        
        # Initialize metadata attributes
        self.has_metadata = False
        self.chunks_with_metadata = []
        
        super().__init__(
            aws_region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            model_config=model_config,
            vector_store=vector_store,
            embedding_model_name=embedding_model_name
        )

    def _build_workflow(self) -> StateGraph:
        """
        Build the RAG-specific workflow graph.
        
        Returns:
            StateGraph: The compiled workflow graph
        """
        workflow = StateGraph(AuditWorkflowState)
        
        # Add nodes for RAG approach
        workflow.add_node("process_question", self._process_question)
        
        if self.use_query_rewriting:
            workflow.add_node("rewrite_query", self._rewrite_query)
            workflow.add_node("retrieve_context", self._retrieve_context_with_rewritten_query)
        else:
            workflow.add_node("retrieve_context", self._retrieve_context_rag)
        
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("format_response", self._format_response)
        
        # Add edges for RAG workflow
        workflow.set_entry_point("process_question")
        
        if self.use_query_rewriting:
            workflow.add_edge("process_question", "rewrite_query")
            workflow.add_edge("rewrite_query", "retrieve_context")
        else:
            workflow.add_edge("process_question", "retrieve_context")
        
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow
    
    def _rewrite_query(self, state: AuditWorkflowState) -> AuditWorkflowState:
        """
        Rewrite the query for better vector retrieval.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            prompt = f"""You are an expert at optimizing queries for semantic vector search retrieval in the financial auditing domain.

Given an original query, rewrite it to be more effective for retrieving relevant financial audit information from a vector database.

Guidelines for rewriting:
1. Expand with relevant financial audit terminology and keywords that might appear in financial statements
2. Remove conversational elements, questions, and filler words
3. Focus on the core information needs and entities
4. Use specific financial statement terminology where appropriate
5. Maintain all important concepts from the original query
6. Keep the rewritten query concise (typically 1-3 sentences)

Original query: {state['question']}

Rewritten query for vector search (just return the rewritten query without explanation or additional text):"""

            # Get response from LLM
            from langchain.schema import HumanMessage
            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            rewritten_query = response.content
            
            # Clean up the response
            rewritten_query = rewritten_query.strip().strip('"\'')
            
            # Store both queries in state
            state["rewritten_question"] = rewritten_query
            state["next_action"] = "retrieve_context"
            
        except Exception as e:
            state["error"] = f"Error rewriting query: {str(e)}"
            state["rewritten_question"] = state["question"]  # Fallback
        
        return state
    
    def _retrieve_context_rag(self, state: AuditWorkflowState) -> AuditWorkflowState:
        """
        Retrieve relevant context using direct vector search.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            # Check if we have chunks available (either in state or in instance)
            has_chunks = (state.get("document_chunks") or 
                         (hasattr(self, 'chunks_with_metadata') and self.chunks_with_metadata))
            
            if has_chunks:
                # Use direct question for vector search
                search_results = self.vector_store.similarity_search(
                    query=state["question"],
                    k=3
                )
                
                # Extract chunks and scores
                retrieved_chunks = []
                for chunk, score in search_results:
                    retrieved_chunks.append(chunk)
                
                state["retrieved_chunks"] = retrieved_chunks
            else:
                # No chunks available
                state["retrieved_chunks"] = []
                state["error"] = "No document chunks available for retrieval"
            
            state["next_action"] = "generate_answer"
            
        except Exception as e:
            # Fallback to using chunks from state or instance attributes
            fallback_chunks = (state.get("document_chunks", []) or 
                             getattr(self, 'chunks_with_metadata', []))
            state["retrieved_chunks"] = fallback_chunks[:3]
            state["error"] = f"Vector search failed, using fallback: {str(e)}"
        
        return state
    
    def _retrieve_context_with_rewritten_query(self, state: AuditWorkflowState) -> AuditWorkflowState:
        """
        Retrieve relevant context using the rewritten query.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            # Check if we have chunks available (either in state or in instance)
            has_chunks = (state.get("document_chunks") or 
                         (hasattr(self, 'chunks_with_metadata') and self.chunks_with_metadata))
            
            if has_chunks:
                # Use rewritten query for vector search
                search_query = state.get("rewritten_question", state["question"])
                search_results = self.vector_store.similarity_search(
                    query=search_query,
                    k=3
                )
                
                # Extract chunks and scores
                retrieved_chunks = []
                for chunk, score in search_results:
                    retrieved_chunks.append(chunk)
                
                state["retrieved_chunks"] = retrieved_chunks
            else:
                # No chunks available
                state["retrieved_chunks"] = []
                state["error"] = "No document chunks available for retrieval"
            
            state["next_action"] = "generate_answer"
            
        except Exception as e:
            # Fallback to using chunks from state or instance attributes
            fallback_chunks = (state.get("document_chunks", []) or 
                             getattr(self, 'chunks_with_metadata', []))
            state["retrieved_chunks"] = fallback_chunks[:3]
            state["error"] = f"Vector search failed, using fallback: {str(e)}"
        
        return state
    
    def _create_answer_prompt(self, question: str, context: str) -> str:
        """
        Create a RAG-specific prompt for answer generation.
        
        Args:
            question: The audit question
            context: The context with citations
            
        Returns:
            Formatted prompt string
        """
        approach_description = "This analysis uses the RAG (Retrieval-Augmented Generation) approach"
        if self.use_query_rewriting:
            approach_description += " with query rewriting for optimized vector search"
        approach_description += ", where the most relevant document sections are retrieved based on semantic similarity."
        
        return f"""You are an expert auditor specializing in higher education financial statements. You are tasked with answering questions about financial audits with ONLY "Yes", "No", or "N/A" (if the question is not applicable or cannot be determined from the available information).

{approach_description}

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
        print(f"🔍 RAG DEBUG - Setting {len(chunks)} chunks with metadata")
        
        self.chunks_with_metadata = chunks
        self.has_metadata = True
        # Also set the document text for backward compatibility
        self.document_text = "\n\n".join(chunk.text for chunk in chunks)
        
        print(f"🔍 RAG DEBUG - Set has_metadata={self.has_metadata}, chunks count={len(self.chunks_with_metadata)}")
        
        # Clear and re-populate vector store for semantic search
        if self.vector_store:
            try:
                print(f"🔍 RAG DEBUG - Clearing vector store before adding new chunks")
                # Clear existing data to avoid stale chunks
                self.vector_store.clear()
                # Add new chunks
                if chunks:
                    print(f"🔍 RAG DEBUG - Adding {len(chunks)} chunks to vector store")
                    self.vector_store.add_chunks(chunks)
                    print(f"✅ RAG DEBUG - Successfully added chunks to vector store")
                else:
                    print(f"⚠️ RAG DEBUG - No chunks to add to vector store")
            except Exception as e:
                print(f"❌ RAG DEBUG - Could not update vector store: {e}")
                print(f"Warning: Could not update vector store: {e}")
        else:
            print(f"⚠️ RAG DEBUG - No vector store available")
    
    def get_relevant_chunks(self, query: str, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        """
        Get the most relevant chunks for a query using vector store semantic search.
        
        Args:
            query: The query to find relevant chunks for
            top_k: Number of top chunks to return
            
        Returns:
            List of tuples with (TextChunk, relevance_score)
        """
        print(f"🔍 RAG RETRIEVAL DEBUG - Query: '{query[:50]}...', top_k={top_k}")
        print(f"🔍 RAG RETRIEVAL DEBUG - has_metadata={getattr(self, 'has_metadata', False)}")
        print(f"🔍 RAG RETRIEVAL DEBUG - chunks_with_metadata count={len(getattr(self, 'chunks_with_metadata', []))}")
        print(f"🔍 RAG RETRIEVAL DEBUG - vector_store available={self.vector_store is not None}")
        
        # Use vector store for semantic search if available
        if self.vector_store and self.has_metadata:
            try:
                print(f"🔍 RAG RETRIEVAL DEBUG - Using vector store for semantic search")
                results = self.vector_store.similarity_search(query, k=top_k)
                print(f"🔍 RAG RETRIEVAL DEBUG - Vector search returned {len(results)} results")
                return results
            except Exception as e:
                print(f"❌ RAG RETRIEVAL DEBUG - Vector search failed: {e}")
                print(f"Vector search failed, falling back to keyword matching: {e}")
        
        # Fallback to keyword-based matching if vector store is not available
        if not self.chunks_with_metadata:
            print(f"⚠️ RAG RETRIEVAL DEBUG - No chunks_with_metadata available, returning empty list")
            return []
        
        print(f"🔍 RAG RETRIEVAL DEBUG - Using keyword-based fallback search")
        # Simple keyword-based relevance scoring as fallback
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
        results = scored_chunks[:top_k]
        print(f"🔍 RAG RETRIEVAL DEBUG - Keyword search returned {len(results)} results")
        return results
    
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
        from langchain.schema import HumanMessage
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        rewritten_query = response.content
        
        # Clean up the response (remove quotes, extra whitespace, etc.)
        rewritten_query = rewritten_query.strip().strip('"\'')
        
        return rewritten_query
    
    def answer_question(self, question_text: str, context: Optional[str] = None, use_query_rewriting: Optional[bool] = None) -> AuditResponse:
        """
        Answer an audit question using the RAG LangGraph workflow.
        
        Args:
            question_text: The audit question text
            context: Optional additional context for the question
            use_query_rewriting: Whether to use query rewriting (overrides the instance setting if provided)
            
        Returns:
            An AuditResponse object with the answer, confidence, and explanation
        """
        # Temporarily override query rewriting setting if provided
        original_query_rewriting = self.use_query_rewriting
        if use_query_rewriting is not None:
            self.use_query_rewriting = use_query_rewriting
            # Rebuild workflow with new setting
            self.workflow = self._build_workflow()
            self.app = self.workflow.compile()
        
        try:
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
        
        finally:
            # Restore original query rewriting setting
            if use_query_rewriting is not None:
                self.use_query_rewriting = original_query_rewriting
                self.workflow = self._build_workflow()
                self.app = self.workflow.compile()
    
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