"""
Example usage of audit implementations with LangGraph workflows.

This script demonstrates how to use the updated audit implementations that
now use LangGraph workflows and Chroma vector store by default.
"""

import os
from typing import List
from pathlib import Path

# Import audit implementations
from hyde_audit import AuditQA as HyDEAuditQA
from pdf_memory_audit import AuditQA as PDFMemoryAuditQA
from rag_audit import AuditQA as RAGAuditQA

# Import supporting classes
from common.core import ChromaVectorStore, BedrockModelConfig
from textract_processor import TextChunk


def create_sample_chunks() -> List[TextChunk]:
    """Create sample text chunks for demonstration."""
    sample_texts = [
        "The university reported total revenues of $150 million for fiscal year 2023, including $80 million in tuition and fees.",
        "Net assets increased by $25 million during the year, primarily due to investment gains and reduced operating expenses.",
        "The auditor's report confirms that the financial statements present fairly the financial position as of June 30, 2023.",
        "Endowment assets totaled $500 million at year-end, with a 12% return on investments for the fiscal year.",
        "The university maintains adequate internal controls over financial reporting with no material weaknesses identified."
    ]
    
    chunks = []
    for i, text in enumerate(sample_texts):
        chunk = TextChunk(
            text=text,
            page_numbers=[i + 1],  # Simulate different pages
            chunk_type="text"
        )
        chunks.append(chunk)
    
    return chunks


def main():
    """Main demonstration function."""
    print("Audit System with LangGraph Workflows")
    print("====================================")
    
    # Initialize shared vector store
    vector_store = ChromaVectorStore(
        collection_name="demo_audit_docs",
        embedding_model_name="amazon.titan-embed-text-v2:0"
    )
    
    # Create sample document chunks
    chunks = create_sample_chunks()
    vector_store.add_chunks(chunks)
    
    # Get collection statistics
    stats = vector_store.get_collection_stats()
    print("\nVector Store Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Note: The actual LLM calls require AWS Bedrock access
    # The following code is provided as an example of how to use the APIs
    
    print("\nAudit Implementation Example Usage:")
    print("----------------------------------")
    
    # Example 1: HyDE Audit approach
    print("\nExample 1: HyDE Audit")
    print("  # Initialize HyDE Audit")
    print("  hyde_qa = HyDEAuditQA(")
    print("      model_config=BedrockModelConfig(")
    print("          model_id=\"anthropic.claude-3-sonnet-20240229-v1:0\",")
    print("          max_tokens=4000")
    print("      ),")
    print("      vector_store=vector_store")
    print("  )")
    print("  hyde_qa.set_document_chunks_with_metadata(chunks)")
    print("  response = hyde_qa.answer_question(\"Did the university show positive financial performance?\")")
    
    # Example 2: PDF Memory Audit approach
    print("\nExample 2: PDF Memory Audit")
    print("  # Initialize PDF Memory Audit")
    print("  pdf_memory_qa = PDFMemoryAuditQA(")
    print("      model_config=BedrockModelConfig(")
    print("          model_id=\"anthropic.claude-3-sonnet-20240229-v1:0\",")
    print("          max_tokens=4000")
    print("      ),")
    print("      vector_store=vector_store")
    print("  )")
    print("  pdf_memory_qa.set_document_chunks_with_metadata(chunks)")
    print("  response = pdf_memory_qa.answer_question(\"Were there any material weaknesses in internal controls?\")")
    
    # Example 3: RAG Audit approach with query rewriting
    print("\nExample 3: RAG Audit with Query Rewriting")
    print("  # Initialize RAG Audit")
    print("  rag_qa = RAGAuditQA(")
    print("      model_config=BedrockModelConfig(")
    print("          model_id=\"anthropic.claude-3-sonnet-20240229-v1:0\",")
    print("          max_tokens=4000")
    print("      ),")
    print("      vector_store=vector_store,")
    print("      use_query_rewriting=True  # Enable query optimization")
    print("  )")
    print("  rag_qa.set_document_chunks_with_metadata(chunks)")
    print("  response = rag_qa.answer_question(\"What was the endowment performance?\")")
    
    print("\nKey Benefits of LangGraph Implementation:")
    print("  - Workflow-based architecture for better state management")
    print("  - Chroma vector store integration for improved retrieval")
    print("  - Support for tool calling and agent hand-offs")
    print("  - Better error handling and state persistence")
    print("  - Modular design for easier extensions")
    
    # Clean up
    try:
        vector_store.clear_collection()
        print("\nVector store collection cleared successfully.")
    except Exception as e:
        print(f"\nError clearing vector store collection: {e}")


if __name__ == "__main__":
    main() 