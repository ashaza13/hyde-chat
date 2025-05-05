#!/usr/bin/env python3
"""
Example script demonstrating the use of the PDF Memory Audit and RAG Audit packages.
"""

import os
import argparse
from typing import Optional

# Import both packages
from pdf_memory_audit import AuditQA as MemoryAuditQA
from rag_audit import AuditQA as RagAuditQA


def main():
    parser = argparse.ArgumentParser(description="Answer audit questions using PDF Memory or RAG approach")
    
    parser.add_argument("--approach", choices=["memory", "rag"], default="rag", 
                        help="The approach to use: 'memory' or 'rag'")
    parser.add_argument("--bucket", type=str, required=True, 
                        help="S3 bucket containing the PDF document")
    parser.add_argument("--key", type=str, required=True, 
                        help="S3 key for the PDF document")
    parser.add_argument("--question", type=str, required=True, 
                        help="The audit question to answer")
    parser.add_argument("--region", type=str, default="us-gov-west-1",
                        help="AWS region for Bedrock")
    
    args = parser.parse_args()
    
    # Get AWS credentials from environment
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    # Initialize the appropriate AuditQA client
    if args.approach == "memory":
        print("Using PDF Memory Audit approach")
        audit_qa = MemoryAuditQA(
            aws_region=args.region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    else:  # args.approach == "rag"
        print("Using RAG Audit approach")
        audit_qa = RagAuditQA(
            aws_region=args.region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    
    # Load the document
    print(f"Loading document from s3://{args.bucket}/{args.key}")
    success = audit_qa.load_document(bucket_name=args.bucket, key=args.key)
    
    if not success:
        print("Failed to load document.")
        return
    
    # Ask the question
    print(f"Question: {args.question}")
    print("Processing...")
    
    response = audit_qa.answer_question(args.question)
    
    # Print the response
    print("\nRESULTS:")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Explanation: {response.explanation}")
    
    # Print sources if available (RAG approach)
    if hasattr(response, 'sources') and response.sources:
        print("\nSources:")
        for source in response.sources:
            print(f"- {source}")


if __name__ == "__main__":
    main() 