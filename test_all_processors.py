#!/usr/bin/env python3
"""
Test script demonstrating how to use PDF Memory, RAG, and Hyde approaches
with the new Textract processor capabilities.
"""

import argparse
import time
from typing import List, Dict, Any

from pdf_memory_audit import AuditQA as MemoryAuditQA
from rag_audit import AuditQA as RagAuditQA
from hyde_audit import AuditQA as HydeAuditQA

def main():
    """Run tests for all three approaches."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test all three audit approaches')
    parser.add_argument('--bucket', required=True, help='S3 bucket containing the PDF')
    parser.add_argument('--key', required=True, help='S3 key for the PDF file')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--question', default='What is the total financial assets available within one year?', 
                       help='Question to ask')
    parser.add_argument('--approaches', default='all', 
                       help='Comma-separated list of approaches to test (memory,rag,hyde,all)')
    args = parser.parse_args()
    
    # Determine which approaches to test
    approaches = args.approaches.lower().split(',')
    test_memory = 'all' in approaches or 'memory' in approaches
    test_rag = 'all' in approaches or 'rag' in approaches
    test_hyde = 'all' in approaches or 'hyde' in approaches
    
    # Initialize clients
    clients = {}
    results = {}
    
    if test_memory:
        print("Initializing PDF Memory approach...")
        clients['memory'] = MemoryAuditQA(aws_region=args.region)
        
    if test_rag:
        print("Initializing RAG approach...")
        clients['rag'] = RagAuditQA(aws_region=args.region)
        
    if test_hyde:
        print("Initializing Hyde approach...")
        clients['hyde'] = HydeAuditQA(aws_region=args.region)
    
    # Load document for each approach
    for name, client in clients.items():
        print(f"Loading document for {name} approach...")
        start_time = time.time()
        success = client.load_document(args.bucket, args.key)
        elapsed = time.time() - start_time
        
        if success:
            print(f"✓ Document loaded successfully for {name} approach in {elapsed:.2f} seconds")
        else:
            print(f"✗ Failed to load document for {name} approach")
    
    # Test each approach with the question
    for name, client in clients.items():
        print(f"\nTesting {name.upper()} approach with question: '{args.question}'")
        print("-" * 80)
        
        start_time = time.time()
        response = client.answer_question(args.question)
        elapsed = time.time() - start_time
        
        print(f"Response time: {elapsed:.2f} seconds")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Explanation: {response.explanation}")
        
        # Save result
        results[name] = {
            'answer': str(response.answer),
            'confidence': response.confidence,
            'explanation': response.explanation,
            'time': elapsed
        }
        
        # Show sources for RAG if available
        if name == 'rag' and hasattr(response, 'sources') and response.sources:
            print("\nSources:")
            for i, source in enumerate(response.sources):
                print(f"  {i+1}. {source}")
    
    # Compare results
    if len(results) > 1:
        print("\nComparison of Results:")
        print("-" * 80)
        
        print(f"{'Approach':<10} | {'Answer':<5} | {'Conf.':<5} | {'Time (s)':<8}")
        print("-" * 40)
        
        for name, result in results.items():
            print(f"{name.capitalize():<10} | {result['answer']:<5} | {result['confidence']:<5.2f} | {result['time']:<8.2f}")

if __name__ == "__main__":
    main() 