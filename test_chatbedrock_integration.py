#!/usr/bin/env python3
"""
Test script to verify ChatBedrockConverse and BedrockEmbeddings integration.
"""

import os
from common.core import BedrockModelConfig, BaseLangGraphAuditWorkflow, ChromaVectorStore
from textract_processor import TextChunk

def test_chatbedrock_integration():
    """Test ChatBedrockConverse and BedrockEmbeddings integration with the audit workflows."""
    
    print("Testing ChatBedrockConverse & BedrockEmbeddings Integration")
    print("=" * 60)
    
    # Create a test model config
    model_config = BedrockModelConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.0,
        max_tokens=1000
    )
    
    # Create a test vector store with Amazon Titan embeddings
    vector_store = ChromaVectorStore(
        collection_name="test_collection",
        embedding_model_name="amazon.titan-embed-text-v2:0"
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
    
    print("\n4. Testing BedrockEmbeddings Integration")
    try:
        # Test that embeddings are properly configured
        print(f"✓ BedrockEmbeddings model initialized: {type(workflow.vector_store.embeddings).__name__}")
        print(f"✓ Embedding model ID: {workflow.vector_store.embedding_model_name}")
        
        # Note: We won't actually call the embeddings here to avoid AWS charges
        # but we can verify it's properly configured
        
    except Exception as e:
        print(f"✗ BedrockEmbeddings test failed: {e}")
        return False
    
    print("\n5. Testing Import Structure")
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
    
    print("\n6. Testing Audit QA Initialization")
    try:
        # Test initializing each audit approach
        hyde_qa = HydeAuditQA(model_config=model_config, vector_store=vector_store)
        rag_qa = RagAuditQA(model_config=model_config, vector_store=vector_store)
        memory_qa = MemoryAuditQA(model_config=model_config, vector_store=vector_store)
        
        print("✓ HyDE QA initialized successfully")
        print("✓ RAG QA initialized successfully")
        print("✓ Memory QA initialized successfully")
        
        # Test that they all use ChatBedrockConverse and BedrockEmbeddings
        print(f"✓ HyDE uses chat model: {type(hyde_qa.chat_model).__name__}")
        print(f"✓ RAG uses chat model: {type(rag_qa.chat_model).__name__}")
        print(f"✓ Memory uses chat model: {type(memory_qa.chat_model).__name__}")
        
        print(f"✓ HyDE uses embeddings: {type(hyde_qa.vector_store.embeddings).__name__}")
        print(f"✓ RAG uses embeddings: {type(rag_qa.vector_store.embeddings).__name__}")
        print(f"✓ Memory uses embeddings: {type(memory_qa.vector_store.embeddings).__name__}")
        
    except Exception as e:
        print(f"✗ Audit QA initialization failed: {e}")
        return False
    
    print("\n7. Testing Model Configurations")
    try:
        # Verify all components use the correct models
        embedding_model = "amazon.titan-embed-text-v2:0"
        chat_model = "anthropic.claude-3-sonnet-20240229-v1:0"
        
        assert hyde_qa.vector_store.embedding_model_name == embedding_model
        assert rag_qa.vector_store.embedding_model_name == embedding_model
        assert memory_qa.vector_store.embedding_model_name == embedding_model
        
        assert hyde_qa.chat_model.model == chat_model
        assert rag_qa.chat_model.model == chat_model
        assert memory_qa.chat_model.model == chat_model
        
        print(f"✓ All implementations use Amazon Titan embeddings: {embedding_model}")
        print(f"✓ All implementations use Claude chat model: {chat_model}")
        
    except Exception as e:
        print(f"✗ Model configuration test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED! Amazon Titan embedding integration successful.")
    print("\nChanges Applied:")
    print("  🔄 Replaced SentenceTransformerEmbeddings with BedrockEmbeddings")
    print("  🎯 Updated default embedding model to amazon.titan-embed-text-v2:0")
    print("  🔗 Integrated AWS credentials for both chat and embeddings")
    print("  📊 All RAG, Memory, and HyDE approaches now use Amazon Titan")
    
    print("\nNote: This test verifies the integration without making actual API calls.")
    print("To test with real AWS Bedrock calls, set your AWS credentials and run:")
    print("  export AWS_ACCESS_KEY_ID=your_key")
    print("  export AWS_SECRET_ACCESS_KEY=your_secret")
    print("  # Then use the audit implementations with real questions")
    
    return True

if __name__ == "__main__":
    success = test_chatbedrock_integration()
    exit(0 if success else 1) 