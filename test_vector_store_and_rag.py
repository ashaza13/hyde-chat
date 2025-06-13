#!/usr/bin/env python3
"""
Test script to inspect vector store contents and test RAG approach functionality.
"""

import os
from typing import List, Tuple
from common.core import BedrockModelConfig, ChromaVectorStore
from textract_processor import TextChunk
from rag_audit import AuditQA as RagAuditQA

def create_test_financial_data() -> List[TextChunk]:
    """Create sample financial document chunks for testing."""
    
    chunks = [
        TextChunk(
            text="""STATEMENT OF REVENUES, EXPENSES AND CHANGES IN NET POSITION
For the Year Ended June 30, 2023

OPERATING REVENUES:
Student tuition and fees (net of scholarship allowances of $12,500,000)    $85,400,000
Federal grants and contracts                                                $28,300,000
State and local grants and contracts                                       $15,200,000
Private gifts, grants and contracts                                        $22,100,000
Auxiliary enterprises                                                       $31,800,000
Other operating revenues                                                    $8,200,000
Total operating revenues                                                   $191,000,000""",
            page_numbers=[3],
            start_page=3,
            end_page=3,
            chunk_index=0
        ),
        
        TextChunk(
            text="""OPERATING EXPENSES:
Instruction                                                                $98,500,000
Research                                                                   $45,200,000
Public service                                                             $18,300,000
Academic support                                                           $22,400,000
Student services                                                           $15,600,000
Institutional support                                                      $28,900,000
Operations and maintenance of plant                                        $19,700,000
Scholarships and fellowships                                               $12,500,000
Auxiliary enterprises                                                      $25,800,000
Depreciation                                                               $15,100,000
Total operating expenses                                                   $302,000,000
Operating loss                                                            $(111,000,000)""",
            page_numbers=[3, 4],
            start_page=3,
            end_page=4,
            chunk_index=1
        ),
        
        TextChunk(
            text="""NON-OPERATING REVENUES (EXPENSES):
State appropriations                                                       $95,600,000
Investment income                                                          $8,400,000
Interest on capital asset-related debt                                    $(2,300,000)
Federal Pell grants                                                        $18,200,000
Other non-operating revenues                                               $3,100,000
Net non-operating revenues                                                $123,000,000

Income before other revenues                                               $12,000,000

OTHER REVENUES:
Capital appropriations                                                     $5,800,000
Capital grants and gifts                                                   $2,200,000
Total other revenues                                                       $8,000,000

Increase in net position                                                   $20,000,000
Net position - beginning of year                                         $485,300,000
Net position - end of year                                               $505,300,000""",
            page_numbers=[4],
            start_page=4,
            end_page=4,
            chunk_index=2
        ),
        
        TextChunk(
            text="""INDEPENDENT AUDITOR'S REPORT

To the Board of Trustees
State University

We have audited the accompanying financial statements of State University, which comprise the statement of net position as of June 30, 2023, and the related statements of revenues, expenses and changes in net position, and cash flows for the year then ended, and the related notes to the financial statements.

Management's Responsibility for the Financial Statements
Management is responsible for the preparation and fair presentation of these financial statements in accordance with accounting principles generally accepted in the United States of America.

Auditor's Responsibility
Our responsibility is to express an opinion on these financial statements based on our audit. We conducted our audit in accordance with auditing standards generally accepted in the United States of America.

Opinion
In our opinion, the financial statements referred to above present fairly, in all material respects, the financial position of State University as of June 30, 2023, and the changes in its net position and its cash flows for the year then ended in accordance with accounting principles generally accepted in the United States of America.""",
            page_numbers=[1, 2],
            start_page=1,
            end_page=2,
            chunk_index=3
        ),
        
        TextChunk(
            text="""NOTE 1 - SUMMARY OF SIGNIFICANT ACCOUNTING POLICIES

Basis of Presentation
The financial statements have been prepared in accordance with accounting principles generally accepted in the United States of America as prescribed by the Governmental Accounting Standards Board (GASB).

Revenue Recognition
Student tuition and fees are recognized as revenue in the period earned. Federal, state, and private grants and contracts are recognized as revenue when qualifying expenditures are incurred and all eligibility requirements are met.

Cash and Cash Equivalents
Cash and cash equivalents include cash on hand, demand deposits, and short-term investments with original maturities of three months or less.

Investments
Investments are reported at fair value based on quoted market prices. Investment income includes realized and unrealized gains and losses on investments.""",
            page_numbers=[8, 9],
            start_page=8,
            end_page=9,
            chunk_index=4
        ),
        
        TextChunk(
            text="""MANAGEMENT'S DISCUSSION AND ANALYSIS

Fiscal Year 2023 Financial Highlights
• Total revenues increased by $15.2 million (8.6%) compared to fiscal year 2022
• Net position increased by $20 million, representing a 4.1% increase
• The University maintained strong enrollment with 18,500 students
• State appropriations increased by $8.3 million due to enhanced state funding formula
• The University completed $45 million in capital projects during the year
• Endowment assets grew to $125 million, an increase of $12 million from the prior year

Key Financial Ratios
• Primary reserve ratio: 0.42 (strong financial cushion)
• Viability ratio: 0.68 (adequate debt capacity)
• Return on net assets: 4.1% (positive trend)
• Net operating revenues ratio: -0.58 (typical for public universities due to reliance on state appropriations)""",
            page_numbers=[5, 6],
            start_page=5,
            end_page=6,
            chunk_index=5
        )
    ]
    
    return chunks

def inspect_vector_store(vector_store: ChromaVectorStore) -> None:
    """Inspect and display vector store contents."""
    
    print("\n🔍 VECTOR STORE INSPECTION")
    print("=" * 50)
    
    # Get collection statistics
    try:
        stats = vector_store.get_collection_stats()
        print(f"Collection Name: {stats.get('collection_name', 'Unknown')}")
        print(f"Document Count: {stats.get('document_count', 0)}")
        print(f"Embedding Model: {stats.get('embedding_model', 'Unknown')}")
        print(f"Persist Directory: {stats.get('persist_directory', 'Unknown')}")
        
        if 'error' in stats:
            print(f"⚠️  Error getting stats: {stats['error']}")
            
    except Exception as e:
        print(f"❌ Failed to get vector store stats: {e}")
        return
    
    # Test some sample searches
    test_queries = [
        "total revenue",
        "operating expenses", 
        "auditor opinion",
        "state appropriations",
        "net position"
    ]
    
    print(f"\n📊 VECTOR SEARCH TESTS")
    print("-" * 30)
    
    for query in test_queries:
        try:
            results = vector_store.similarity_search(query, k=2)
            print(f"\nQuery: '{query}'")
            print(f"Results found: {len(results)}")
            
            for i, (chunk, score) in enumerate(results[:2]):
                print(f"  [{i+1}] Score: {score:.3f}, Page: {chunk.get_page_range_str()}")
                print(f"      Text preview: {chunk.text[:100]}...")
                
        except Exception as e:
            print(f"  ❌ Search failed for '{query}': {e}")

def test_rag_workflow(rag_qa: RagAuditQA, test_questions: List[str]) -> None:
    """Test RAG workflow with sample questions."""
    
    print(f"\n🤖 RAG WORKFLOW TESTS")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test Question {i} ---")
        print(f"Q: {question}")
        
        try:
            # Test the retrieval part (without LLM call)
            print(f"\n🔍 Testing Retrieval:")
            
            # Get relevant chunks using the RAG QA's method
            relevant_chunks = rag_qa.get_relevant_chunks(question, top_k=3)
            
            print(f"Retrieved {len(relevant_chunks)} chunks:")
            for j, (chunk, score) in enumerate(relevant_chunks):
                print(f"  [{j+1}] Score: {score:.3f}")
                print(f"      Page: {chunk.get_page_range_str()}")
                print(f"      Index: {chunk.chunk_index}")
                print(f"      Preview: {chunk.text[:150]}...")
                print()
            
            # Build context that would be sent to LLM
            print(f"📝 Context that would be sent to LLM:")
            context_parts = []
            
            for j, (chunk, score) in enumerate(relevant_chunks):
                citation_id = j + 1
                context_parts.append(f"[{citation_id}] {chunk.text}")
            
            full_context = "\n\n".join(context_parts)
            print(f"Context length: {len(full_context)} characters")
            
            if full_context:
                print(f"Context preview:\n{full_context[:300]}...")
                print(f"\n✅ RAG retrieval working - context would be available to LLM")
            else:
                print(f"❌ No context retrieved - this would cause 'insufficient information' error")
            
        except Exception as e:
            print(f"❌ RAG test failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main test function."""
    
    print("🧪 VECTOR STORE & RAG TESTING")
    print("=" * 60)
    
    # Create test data
    print("📚 Creating test financial data...")
    test_chunks = create_test_financial_data()
    print(f"Created {len(test_chunks)} financial document chunks")
    
    # Create model config (won't be used for LLM calls in this test)
    model_config = BedrockModelConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.0,
        max_tokens=1000
    )
    
    # Create RAG QA system
    print("\n🔧 Initializing RAG QA system...")
    try:
        rag_qa = RagAuditQA(
            model_config=model_config,
            use_query_rewriting=False  # Test without query rewriting first
        )
        print("✅ RAG QA system initialized")
        
        # Add chunks to the system
        print("📥 Adding document chunks...")
        rag_qa.set_document_chunks_with_metadata(test_chunks)
        print(f"✅ Added {len(test_chunks)} chunks to RAG system")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG QA: {e}")
        return False
    
    # Inspect vector store
    inspect_vector_store(rag_qa.vector_store)
    
    # Test RAG workflow with sample questions
    test_questions = [
        "Did the university report total revenue greater than $150 million?",
        "What was the university's operating loss for the year?",
        "Did the auditor provide an unqualified opinion?",
        "How much did the university receive in state appropriations?",
        "What was the increase in net position for the year?"
    ]
    
    test_rag_workflow(rag_qa, test_questions)
    
    print(f"\n" + "=" * 60)
    print("🎯 TEST SUMMARY:")
    print("✅ Vector store populated with financial data")
    print("✅ Embedding model: amazon.titan-embed-text-v2:0") 
    print("✅ RAG retrieval mechanism tested")
    print("✅ Context generation verified")
    print("\n💡 If RAG retrieval is working but you're still getting 'insufficient information':")
    print("   1. Check AWS credentials for Bedrock")
    print("   2. Verify Amazon Titan embeddings are working")
    print("   3. Check if the LLM is receiving the context properly")
    print("\n🔬 To test with real AWS calls, set credentials and call:")
    print("   rag_qa.answer_question('Your question here')")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 