# LangGraph Implementation Guide

This document outlines the LangGraph-based workflow implementation and Langchain Chroma vector store integration in the Hyde Chat audit system.

## Overview

The system has been updated to use LangGraph-based implementations as the default approach. This provides:

- **Workflow-based architecture** using LangGraph for better state management
- **Langchain Chroma vector store** for improved document retrieval
- **Tool calling capabilities** for future agent hand-offs
- **Enhanced error handling** and workflow monitoring
- **Modular design** for easier extensibility

## Dependencies

The following dependencies have been added to `requirements.txt`:

```text
# LangGraph and LangChain dependencies
langgraph>=0.0.65
langchain>=0.1.0
langchain-community>=0.0.10
langchain-chroma>=0.1.0
chromadb>=0.4.0
# Additional dependencies for enhanced functionality
langchain-aws>=0.1.0
```

## Architecture Components

### 1. Vector Store Integration

A new `ChromaVectorStore` class has been added in `common/core/vector_store.py`:

```python
from common.core import ChromaVectorStore

# Initialize vector store
vector_store = ChromaVectorStore(
    collection_name="audit_documents",
    embedding_model_name="all-MiniLM-L6-v2",
    persist_directory="./chroma_db"
)

# Add document chunks
vector_store.add_chunks(chunks)

# Perform similarity search
results = vector_store.similarity_search("query", k=3)
```

### 2. Base LangGraph Workflow

A base workflow class `BaseLangGraphAuditWorkflow` provides common functionality:

```python
from common.core import BaseLangGraphAuditWorkflow, AuditWorkflowState

class CustomAuditWorkflow(BaseLangGraphAuditWorkflow):
    def _build_workflow(self) -> StateGraph:
        # Define custom workflow nodes and edges
        pass
```

### 3. Workflow State Management

All workflows use a shared state structure:

```python
class AuditWorkflowState(TypedDict):
    messages: List[BaseMessage]
    question: str
    context: Optional[str]
    document_chunks: List[TextChunk]
    retrieved_chunks: List[TextChunk]
    hypothetical_doc: Optional[str]  # For HyDE
    answer: Optional[str]
    confidence: Optional[float]
    explanation: Optional[str]
    page_references: List[str]
    next_action: Optional[str]
    error: Optional[str]
```

## Audit Implementations

### 1. HyDE Audit (Hypothetical Document Embeddings)

```python
from hyde_audit import AuditQA  # Now uses LangGraph implementation

hyde_qa = AuditQA(
    model_config=BedrockModelConfig(...),
    vector_store=vector_store  # Optional, creates own if not provided
)

# Set document chunks (automatically added to vector store)
hyde_qa.set_document_chunks_with_metadata(chunks)

# Answer questions
response = hyde_qa.answer_question("Is the financial position strong?")
```

**Workflow Steps:**
1. Process question
2. Generate hypothetical document
3. Retrieve context using hypothetical document
4. Generate answer
5. Format response

### 2. PDF Memory Audit

```python
from pdf_memory_audit import AuditQA  # Now uses LangGraph implementation

pdf_qa = AuditQA(
    model_config=BedrockModelConfig(...),
    vector_store=vector_store
)

pdf_qa.set_document_chunks_with_metadata(chunks)
response = pdf_qa.answer_question("Are there control weaknesses?")
```

**Workflow Steps:**
1. Process question
2. Retrieve all context (full document approach)
3. Generate answer
4. Format response

### 3. RAG Audit (Retrieval-Augmented Generation)

```python
from rag_audit import AuditQA  # Now uses LangGraph implementation

rag_qa = AuditQA(
    model_config=BedrockModelConfig(...),
    vector_store=vector_store,
    use_query_rewriting=True  # Optional query optimization
)

rag_qa.set_document_chunks_with_metadata(chunks)
response = rag_qa.answer_question("What was the revenue growth?")
```

**Workflow Steps (with query rewriting):**
1. Process question
2. Rewrite query for better vector search
3. Retrieve context using rewritten query
4. Generate answer
5. Format response

**Workflow Steps (without query rewriting):**
1. Process question
2. Retrieve context using original query
3. Generate answer
4. Format response

## Using the Implementation

### Standard Usage

```python
# Import the audit module
from hyde_audit import AuditQA  # LangGraph implementation is the default

# Initialize
qa = AuditQA(
    aws_region="us-gov-west-1",
    vector_store=ChromaVectorStore()  # Optional
)

# Set document chunks
qa.set_document_chunks_with_metadata(chunks)

# Answer questions
response = qa.answer_question("Question?")
```

## Tool Calling and Agent Hand-offs

The LangGraph implementations are designed to support tool calling and agent hand-offs:

### Adding Tools

```python
from langchain_core.tools import BaseTool

class CustomAuditTool(BaseTool):
    name = "custom_audit_tool"
    description = "Custom tool for audit analysis"
    
    def _run(self, query: str) -> str:
        # Tool implementation
        return "Tool result"

# Add to workflow
workflow.add_node("use_tool", ToolNode([CustomAuditTool()]))
```

### Agent Hand-offs

```python
# Define hand-off conditions in workflow
def should_hand_off(state: AuditWorkflowState) -> str:
    if state["confidence"] < 0.5:
        return "hand_off_to_specialist"
    return "continue_workflow"

workflow.add_conditional_edges(
    "generate_answer",
    should_hand_off,
    {
        "continue_workflow": "format_response",
        "hand_off_to_specialist": "specialist_agent"
    }
)
```

## Performance Considerations

### Vector Store Performance

- Chroma provides persistent storage and efficient similarity search
- Embeddings are computed once and reused
- Collection statistics help monitor performance

### Workflow Optimization

- State is managed efficiently through LangGraph
- Error handling prevents workflow failures
- Conditional edges enable dynamic workflows

## Future Enhancements

The LangGraph architecture enables:

1. **Multi-agent workflows** for complex audit scenarios
2. **Tool integration** for external data sources
3. **Streaming responses** for real-time feedback
4. **Workflow monitoring** and debugging
5. **Custom node types** for specialized processing

## Troubleshooting

### Common Issues

1. **Import errors:** Ensure all new dependencies are installed
2. **Vector store errors:** Check embedding model availability
3. **Workflow errors:** Review state structure and node definitions
4. **Memory issues:** Consider chunking strategies for large documents

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

```python
# Get vector store statistics
stats = vector_store.get_collection_stats()
print(f"Documents: {stats['document_count']}")
```

## Support

For issues or questions:
1. Check this implementation guide
2. Examine the workflow implementations in each audit module
3. Test with sample data

The LangGraph implementations provide a solid foundation for future enhancements including agent hand-offs and tool calling. 