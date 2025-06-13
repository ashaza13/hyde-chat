"""
Base LangGraph workflow implementation for audit question answering.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_aws import ChatBedrockConverse

from .vector_store import ChromaVectorStore
from textract_processor import TextChunk


class BedrockModelConfig:
    """Configuration for AWS Bedrock models."""
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 0.9
    ):
        """
        Initialize the model configuration.
        
        Args:
            model_id: The Bedrock model ID to use
            temperature: Temperature for model generation
            max_tokens: Maximum tokens to generate
            top_p: Top p sampling parameter
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p


class AuditWorkflowState(TypedDict):
    """State for the audit workflow."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    question: str
    context: Optional[str]
    document_chunks: List[TextChunk]
    retrieved_chunks: List[TextChunk]
    hypothetical_doc: Optional[str]
    answer: Optional[str]
    confidence: Optional[float]
    explanation: Optional[str]
    page_references: List[str]
    next_action: Optional[str]
    error: Optional[str]


class BaseLangGraphAuditWorkflow:
    """
    Base class for LangGraph-based audit workflows.
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
        Initialize the base workflow.
        
        Args:
            aws_region: AWS region for Bedrock
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_session_token: AWS session token
            model_config: Bedrock model configuration
            vector_store: Vector store instance
            embedding_model_name: Embedding model name
        """
        # Set default model config if not provided
        if model_config is None:
            model_config = BedrockModelConfig()
        
        # Initialize ChatBedrockConverse client
        chat_kwargs = {
            "model": model_config.model_id,
            "region_name": aws_region,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
        }
        
        # Add credentials if provided
        if aws_access_key_id and aws_secret_access_key:
            chat_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
            if aws_session_token:
                chat_kwargs["aws_session_token"] = aws_session_token
        
        self.chat_model = ChatBedrockConverse(**chat_kwargs)
        self.model_config = model_config
        
        # Initialize vector store
        if vector_store is None:
            self.vector_store = ChromaVectorStore(
                embedding_model_name=embedding_model_name
            )
        else:
            self.vector_store = vector_store
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow. Should be overridden by subclasses.
        
        Returns:
            StateGraph: The compiled workflow graph
        """
        workflow = StateGraph(AuditWorkflowState)
        
        # Add nodes
        workflow.add_node("process_question", self._process_question)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("format_response", self._format_response)
        
        # Add edges
        workflow.set_entry_point("process_question")
        workflow.add_edge("process_question", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow
    
    def _process_question(self, state: AuditWorkflowState) -> AuditWorkflowState:
        """
        Process the initial question and set up the workflow state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        # Add the question as a human message if not already present
        if not state["messages"]:
            state["messages"] = [HumanMessage(content=state["question"])]
        
        # Initialize other state variables
        state["retrieved_chunks"] = []
        state["page_references"] = []
        state["next_action"] = "retrieve_context"
        
        return state
    
    def _retrieve_context(self, state: AuditWorkflowState) -> AuditWorkflowState:
        """
        Retrieve relevant context for the question. Should be overridden by subclasses.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        # Default implementation: use all chunks
        state["retrieved_chunks"] = state.get("document_chunks", [])
        state["next_action"] = "generate_answer"
        return state
    
    def _generate_answer(self, state: AuditWorkflowState) -> AuditWorkflowState:
        """
        Generate an answer based on the retrieved context.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            # Build context from retrieved chunks
            context_parts = []
            
            if state.get("context"):
                context_parts.append(f"Additional Context: {state['context']}")
            
            # Add retrieved chunks with citations
            for i, chunk in enumerate(state["retrieved_chunks"]):
                citation_id = i + 1
                context_parts.append(f"[{citation_id}] {chunk.text}")
                state["page_references"].append(chunk.get_page_range_str())
            
            full_context = "\n\n".join(context_parts)
            
            # Create prompt for answer generation
            prompt = self._create_answer_prompt(state["question"], full_context)
            
            # Generate answer using ChatBedrockConverse
            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            response_text = response.content
            
            # Parse the response (assuming JSON format)
            import json
            try:
                parsed_response = json.loads(response_text)
                state["answer"] = parsed_response.get("answer", "N/A")
                state["confidence"] = parsed_response.get("confidence", 0.0)
                state["explanation"] = parsed_response.get("explanation", "")
            except json.JSONDecodeError:
                # Fallback to plain text parsing
                state["answer"] = "N/A"
                state["confidence"] = 0.0
                state["explanation"] = response_text
            
            state["next_action"] = "format_response"
            
        except Exception as e:
            state["error"] = str(e)
            state["answer"] = "N/A"
            state["confidence"] = 0.0
            state["explanation"] = f"Error generating answer: {str(e)}"
        
        return state
    
    def _format_response(self, state: AuditWorkflowState) -> AuditWorkflowState:
        """
        Format the final response.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        # Add the AI response to messages
        if state.get("answer"):
            state["messages"].append(AIMessage(content=state["answer"]))
        
        return state
    
    def _create_answer_prompt(self, question: str, context: str) -> str:
        """
        Create a prompt for answer generation.
        
        Args:
            question: The audit question
            context: The relevant context
            
        Returns:
            Formatted prompt string
        """
        return f"""Based on the following context, please answer the audit question. 

Context:
{context}

Question: {question}

Please provide your answer in the following JSON format:
{{
    "answer": "Yes/No/N/A",
    "confidence": 0.0-1.0,
    "explanation": "Detailed explanation with reasoning and evidence from the context"
}}

If you reference information from the context, please include the citation numbers [1], [2], etc. in your explanation."""
    
    def set_document_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Set the document chunks for the workflow.
        
        Args:
            chunks: List of TextChunk objects
        """
        # Add chunks to vector store
        if self.vector_store and chunks:
            self.vector_store.clear()
            self.vector_store.add_chunks(chunks)
    
    def run_workflow(
        self,
        question: str,
        context: Optional[str] = None,
        document_chunks: Optional[List[TextChunk]] = None
    ) -> Dict[str, Any]:
        """
        Run the audit workflow.
        
        Args:
            question: The audit question
            context: Optional additional context
            document_chunks: Optional document chunks to use
            
        Returns:
            Dictionary containing the workflow results
        """
        # Initialize state
        initial_state = AuditWorkflowState(
            messages=[],
            question=question,
            context=context,
            document_chunks=document_chunks or [],
            retrieved_chunks=[],
            hypothetical_doc=None,
            answer=None,
            confidence=None,
            explanation=None,
            page_references=[],
            next_action=None,
            error=None
        )
        
        # Run the workflow
        final_state = self.app.invoke(initial_state)
        
        # Return results
        return {
            "answer": final_state.get("answer", "N/A"),
            "confidence": final_state.get("confidence", 0.0),
            "explanation": final_state.get("explanation", ""),
            "page_references": final_state.get("page_references", []),
            "error": final_state.get("error")
        } 