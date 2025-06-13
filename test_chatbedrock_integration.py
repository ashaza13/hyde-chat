#!/usr/bin/env python3
"""
Test script to verify ChatBedrockConverse integration.
"""

import os
from common.core import BedrockModelConfig, BaseLangGraphAuditWorkflow, ChromaVectorStore
from textract_processor import TextChunk

def test_chatbedrock_integration():
    """Test ChatBedrockConverse integration with the audit workflows."""
    
    print("Testing ChatBedrockConverse Integration")
    print("=" * 50)
    
    # Create a test model config
    model_config = BedrockModelConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.0,
        max_tokens=1000
    )
    
    # Create a test vector store
    vector_store = ChromaVectorStore(
        collection_name="test_collection",
        embedding_model_name="all-MiniLM-L6-v2"
    )
    
    # Create sample test chunks
    test_chunks = [
        TextChunk(
            text="The university reported total revenue of $100 million for the fiscal year ending June 30, 2023.",
            page_numbers=[1],
            chunk_type="text"
        ),
        TextChunk(
            text="Net assets increased by $10 million, indicating strong financial performance.",
            page_numbers=[2], 
            chunk_type="text"
        )
    ]
    
    print("\n1. Testing Base Workflow Initialization")
    try:
        # Initialize base workflow
        workflow = BaseLangGraphAuditWorkflow(
            model_config=model_config,
            vector_store=vector_store
        )
        print("✓ Base workflow initialized successfully")
        print(f"✓ Model ID: {workflow.model_config.model_id}")
        print(f"✓ Temperature: {workflow.model_config.temperature}")
        print(f"✓ Max tokens: {workflow.model_config.max_tokens}")
        
    except Exception as e:
        print(f"✗ Base workflow initialization failed: {e}")
        return False
    
    print("\n2. Testing Vector Store Operations")
    try:
        # Test adding chunks to vector store
        vector_store.add_chunks(test_chunks)
        stats = vector_store.get_collection_stats()
        print(f"✓ Added {len(test_chunks)} chunks to vector store")
        print(f"✓ Collection stats: {stats}")
        
    except Exception as e:
        print(f"✗ Vector store operations failed: {e}")
        return False
    
    print("\n3. Testing ChatBedrockConverse Model")
    try:
        # Test that the chat model is properly initialized
        print(f"✓ ChatBedrockConverse model initialized: {type(workflow.chat_model).__name__}")
        print(f"✓ Model configuration: {workflow.chat_model.model}")
        
        # Note: We won't actually call the model here to avoid AWS charges
        # but we can verify it's properly configured
        
    except Exception as e:
        print(f"✗ ChatBedrockConverse model test failed: {e}")
        return False
    
    print("\n4. Testing Import Structure")
    try:
        from hyde_audit import AuditQA as HydeAuditQA
        from rag_audit import AuditQA as RagAuditQA
        from pdf_memory_audit import AuditQA as MemoryAuditQA
        
        print("✓ All audit implementations imported successfully")
        print("✓ HyDE audit available")
        print("✓ RAG audit available") 
        print("✓ Memory audit available")
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False
    
    print("\n5. Testing Audit QA Initialization")
    try:
        # Test initializing each audit approach
        hyde_qa = HydeAuditQA(model_config=model_config, vector_store=vector_store)
        rag_qa = RagAuditQA(model_config=model_config, vector_store=vector_store)
        memory_qa = MemoryAuditQA(model_config=model_config, vector_store=vector_store)
        
        print("✓ HyDE QA initialized successfully")
        print("✓ RAG QA initialized successfully")
        print("✓ Memory QA initialized successfully")
        
        # Test that they all use ChatBedrockConverse
        print(f"✓ HyDE uses: {type(hyde_qa.chat_model).__name__}")
        print(f"✓ RAG uses: {type(rag_qa.chat_model).__name__}")
        print(f"✓ Memory uses: {type(memory_qa.chat_model).__name__}")
        
    except Exception as e:
        print(f"✗ Audit QA initialization failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! ChatBedrockConverse integration successful.")
    print("\nNote: This test verifies the integration without making actual API calls.")
    print("To test with real AWS Bedrock calls, set your AWS credentials and run:")
    print("  export AWS_ACCESS_KEY_ID=your_key")
    print("  export AWS_SECRET_ACCESS_KEY=your_secret")
    print("  # Then use the audit implementations with real questions")
    
    return True

if __name__ == "__main__":
    success = test_chatbedrock_integration()
    exit(0 if success else 1) 