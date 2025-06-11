#!/usr/bin/env python3
"""
Module for processing audit questions using various approaches.
"""

import os
import argparse
from typing import Optional, List, Dict
import pandas as pd

from common.core import DocumentProcessor
from pdf_memory_audit import AuditQA as MemoryAuditQA
from rag_audit import AuditQA as RagAuditQA
from hyde_audit import AuditQA as HydeAuditQA
from question_structure import QuestionTree, AnswerType, Question


class AuditProcessor:
    """
    Processes audit questions using multiple approaches and maintains the results.
    """
    
    def __init__(
        self, 
        aws_region: str = "us-gov-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        use_rag_query_rewriting: bool = False
    ):
        """
        Initialize the AuditProcessor.
        
        Args:
            aws_region: AWS region for Bedrock
            aws_access_key_id: AWS access key ID (optional if using IAM roles)
            aws_secret_access_key: AWS secret access key (optional if using IAM roles)
            aws_session_token: AWS session token (optional, used for temporary credentials)
            use_rag_query_rewriting: Whether to use query rewriting for the RAG approach
        """
        self.aws_region = aws_region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.use_rag_query_rewriting = use_rag_query_rewriting
        
        # Initialize the central document processor
        self.document_processor = DocumentProcessor(
            aws_region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        
        # Initialize audit clients
        self.memory_qa = MemoryAuditQA(
            aws_region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        
        self.rag_qa = RagAuditQA(
            aws_region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            use_query_rewriting=use_rag_query_rewriting
        )
        
        self.hyde_qa = HydeAuditQA(
            aws_region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        
        # Document loading status
        self.document_loaded = False
        self.document_bucket = None
        self.document_key = None
        
        # Question tree
        self.question_tree = None
        
        # Processed questions cache (for dependency tracking)
        self.processed_questions: Dict[str, Dict[str, any]] = {}
    
    def load_document(self, bucket_name: str, key: str, force_reprocess: bool = False) -> bool:
        """
        Load a document using the centralized document processor and make it available to all approaches.
        
        Args:
            bucket_name: S3 bucket name
            key: S3 object key
            force_reprocess: Whether to force reprocessing even if a processed document exists
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Loading document s3://{bucket_name}/{key}...")
        
        # Process the document once using the centralized document processor
        success, document_text = self.document_processor.process_document(
            bucket_name=bucket_name,
            key=key,
            force_reprocess=force_reprocess
        )
        
        if not success:
            print("Failed to process document")
            return False
        
        # Store document details for future reference
        self.document_bucket = bucket_name
        self.document_key = key
        
        # Check if we have metadata available
        if self.document_processor.has_page_metadata():
            print("Document processed with page metadata - enabling enhanced citation support")
            
            # Get chunks with metadata
            chunks_with_metadata = self.document_processor.get_chunked_text_with_metadata(
                chunk_size=1000, 
                overlap=200
            )
            
            print(f"Created {len(chunks_with_metadata)} chunks with page metadata")
            
            # Set chunks with metadata for RAG and HYDE approaches (only they support vectordb metadata)
            self.rag_qa.set_document_chunks_with_metadata(chunks_with_metadata)
            self.hyde_qa.set_document_chunks_with_metadata(chunks_with_metadata)
            
            # Set document text for memory approach (no metadata support)
            self.memory_qa.set_document_text(document_text)
            
            # Print document summary
            summary = self.document_processor.get_document_summary()
            print(f"Document Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
        else:
            print("Document processed without detailed metadata - using standard approach")
            
            # Set the document text in each of the audit QA instances (original approach)
            self.memory_qa.set_document_text(document_text)
            self.rag_qa.set_document_text(document_text)
            self.hyde_qa.set_document_text(document_text)
        
        self.document_loaded = True
        return True
    
    def load_questions(self, csv_path: str) -> bool:
        """
        Load questions from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.question_tree = QuestionTree.from_csv(csv_path)
            print(f"Loaded {len(self.question_tree.questions)} questions from {csv_path}")
            return True
        except Exception as e:
            print(f"Error loading questions: {e}")
            return False
    
    def get_dependency_context(self, question_id: str, approach: str) -> str:
        """
        Build a context string based on the dependencies of a question.
        This context includes the dependent questions and their answers.
        
        Args:
            question_id: ID of the question
            approach: The approach being used ("rag", "memory", "hyde")
            
        Returns:
            A context string with information about dependencies
        """
        if not self.question_tree:
            return ""
        
        question = self.question_tree.get_question(question_id)
        if not question or not question.dependencies:
            return ""
        
        # Build context from dependencies
        context_parts = ["Before answering the main question, consider these related sub-questions and their answers:"]
        
        for dep_id in question.dependencies:
            if dep_id not in self.processed_questions or approach not in self.processed_questions[dep_id]:
                # If dependency hasn't been processed with this approach, we can't use it
                continue
                
            dep_question = self.question_tree.get_question(dep_id)
            if not dep_question:
                continue
                
            # Get the answer and explanation for this dependency based on the approach
            if approach == "rag":
                answer = dep_question.rag_answer
                confidence = dep_question.rag_confidence
                explanation = dep_question.rag_explanation
            elif approach == "memory":
                answer = dep_question.context_answer
                confidence = dep_question.context_confidence
                explanation = dep_question.context_explanation
            elif approach == "hyde":
                answer = dep_question.hyde_answer
                confidence = dep_question.hyde_confidence
                explanation = dep_question.hyde_explanation
            else:
                continue
                
            if not answer or not explanation:
                continue
                
            # Add this dependency's question and answer to the context
            context_parts.append(f"Sub-question {dep_id}: {dep_question.text}")
            context_parts.append(f"Answer: {answer.value}")
            context_parts.append(f"Confidence: {confidence if confidence else 'Unknown'}")
            context_parts.append(f"Explanation: {explanation}")
            context_parts.append("")  # Empty line for separation
        
        # Only return context if we have actual dependency information
        if len(context_parts) > 1:
            return "\n".join(context_parts)
        else:
            return ""
    
    def process_with_rag(self, question_id: str, use_query_rewriting: Optional[bool] = None) -> bool:
        """
        Process a question using the RAG approach.
        
        Args:
            question_id: ID of the question to process
            use_query_rewriting: Whether to use query rewriting (overrides the instance setting if provided)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.document_loaded:
            print("Error: Document not loaded")
            return False
        
        if not self.question_tree:
            print("Error: Questions not loaded")
            return False
        
        question = self.question_tree.get_question(question_id)
        if not question:
            print(f"Error: Question {question_id} not found")
            return False
        
        print(f"Processing question {question_id} with RAG approach...")
        print(f"Question: {question.text}")
        
        # Determine whether to use query rewriting
        should_rewrite = use_query_rewriting if use_query_rewriting is not None else self.use_rag_query_rewriting
        if should_rewrite:
            print("Using query rewriting for vector search optimization")
        
        # Get dependency context
        dependency_context = self.get_dependency_context(question_id, "rag")
        if dependency_context:
            print("Using context from dependencies")
        
        # Process the question with RAG
        response = self.rag_qa.answer_question(
            question.text, 
            context=dependency_context,
            use_query_rewriting=should_rewrite
        )
        
        # Update the question with the response
        if response.answer == "Yes":
            question.rag_answer = AnswerType.YES
        elif response.answer == "No":
            question.rag_answer = AnswerType.NO
        elif response.answer == "N/A":
            question.rag_answer = AnswerType.NA
        else:
            question.rag_answer = AnswerType.UNKNOWN
            
        question.rag_confidence = response.confidence
        question.rag_explanation = response.explanation
        
        print(f"RAG Answer: {question.rag_answer}")
        print(f"RAG Confidence: {question.rag_confidence}")
        
        # Store processed question for future dependency resolution
        if question_id not in self.processed_questions:
            self.processed_questions[question_id] = {}
        self.processed_questions[question_id]["rag"] = {
            "answer": question.rag_answer,
            "confidence": question.rag_confidence,
            "explanation": question.rag_explanation
        }
        
        return True
    
    def process_with_memory(self, question_id: str) -> bool:
        """
        Process a question using the Memory approach.
        
        Args:
            question_id: ID of the question to process
            
        Returns:
            True if successful, False otherwise
        """
        if not self.document_loaded:
            print("Error: Document not loaded")
            return False
        
        if not self.question_tree:
            print("Error: Questions not loaded")
            return False
        
        question = self.question_tree.get_question(question_id)
        if not question:
            print(f"Error: Question {question_id} not found")
            return False
        
        print(f"Processing question {question_id} with Memory approach...")
        print(f"Question: {question.text}")
        
        # Get dependency context
        dependency_context = self.get_dependency_context(question_id, "memory")
        if dependency_context:
            print("Using context from dependencies")
        
        # Process the question with Memory approach
        response = self.memory_qa.answer_question(question.text, context=dependency_context)
        
        # Update the question with the response
        if response.answer == "Yes":
            question.context_answer = AnswerType.YES
        elif response.answer == "No":
            question.context_answer = AnswerType.NO
        elif response.answer == "N/A":
            question.context_answer = AnswerType.NA
        else:
            question.context_answer = AnswerType.UNKNOWN
            
        question.context_confidence = response.confidence
        question.context_explanation = response.explanation
        
        print(f"Memory Answer: {question.context_answer}")
        print(f"Memory Confidence: {question.context_confidence}")
        
        # Store processed question for future dependency resolution
        if question_id not in self.processed_questions:
            self.processed_questions[question_id] = {}
        self.processed_questions[question_id]["memory"] = {
            "answer": question.context_answer,
            "confidence": question.context_confidence,
            "explanation": question.context_explanation
        }
        
        return True
    
    def process_with_hyde(self, question_id: str) -> bool:
        """
        Process a question using the HYDE approach.
        
        Args:
            question_id: ID of the question to process
            
        Returns:
            True if successful, False otherwise
        """
        if not self.document_loaded:
            print("Error: Document not loaded")
            return False
        
        if not self.question_tree:
            print("Error: Questions not loaded")
            return False
        
        question = self.question_tree.get_question(question_id)
        if not question:
            print(f"Error: Question {question_id} not found")
            return False
        
        print(f"Processing question {question_id} with HYDE approach...")
        print(f"Question: {question.text}")
        
        # Get dependency context
        dependency_context = self.get_dependency_context(question_id, "hyde")
        if dependency_context:
            print("Using context from dependencies")
        
        # Process the question with HYDE approach
        response = self.hyde_qa.answer_question(question.text, context=dependency_context)
        
        # Update the question with the response
        if response.answer == "Yes":
            question.hyde_answer = AnswerType.YES
        elif response.answer == "No":
            question.hyde_answer = AnswerType.NO
        elif response.answer == "N/A":
            question.hyde_answer = AnswerType.NA
        else:
            question.hyde_answer = AnswerType.UNKNOWN
            
        question.hyde_confidence = response.confidence
        question.hyde_explanation = response.explanation
        
        print(f"HYDE Answer: {question.hyde_answer}")
        print(f"HYDE Confidence: {question.hyde_confidence}")
        
        # Store processed question for future dependency resolution
        if question_id not in self.processed_questions:
            self.processed_questions[question_id] = {}
        self.processed_questions[question_id]["hyde"] = {
            "answer": question.hyde_answer,
            "confidence": question.hyde_confidence,
            "explanation": question.hyde_explanation
        }
        
        return True
    
    def process_question_batch(self, question_ids: List[str], approaches: List[str], use_rag_query_rewriting: Optional[bool] = None) -> bool:
        """
        Process a batch of questions with specified approaches.
        
        Args:
            question_ids: List of question IDs to process
            approaches: List of approaches to use ("rag", "memory", "hyde")
            use_rag_query_rewriting: Whether to use query rewriting for RAG (overrides the instance setting if provided)
            
        Returns:
            True if all processing was successful, False otherwise
        """
        all_successful = True
        
        for question_id in question_ids:
            for approach in approaches:
                if approach.lower() == "rag":
                    success = self.process_with_rag(question_id, use_query_rewriting=use_rag_query_rewriting)
                elif approach.lower() == "memory":
                    success = self.process_with_memory(question_id)
                elif approach.lower() == "hyde":
                    success = self.process_with_hyde(question_id)
                else:
                    print(f"Unknown approach: {approach}")
                    success = False
                
                all_successful = all_successful and success
        
        return all_successful
    
    def process_all_questions(self, approaches: List[str], use_rag_query_rewriting: Optional[bool] = None) -> bool:
        """
        Process all questions with specified approaches in dependency order.
        
        Args:
            approaches: List of approaches to use ("rag", "memory", "hyde")
            use_rag_query_rewriting: Whether to use query rewriting for RAG (overrides the instance setting if provided)
            
        Returns:
            True if all processing was successful, False otherwise
        """
        if not self.question_tree:
            print("Error: Questions not loaded")
            return False
        
        # Process questions in dependency order to ensure dependencies are processed first
        ordered_questions = self.question_tree.get_questions_in_order()
        question_ids = [q.id for q in ordered_questions]
        
        print(f"Processing {len(question_ids)} questions in dependency order")
        return self.process_question_batch(question_ids, approaches, use_rag_query_rewriting=use_rag_query_rewriting)
    
    def save_results(self, output_csv: str) -> bool:
        """
        Save the current state of questions to a CSV file.
        
        Args:
            output_csv: Path to the output CSV file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.question_tree:
            print("Error: Questions not loaded")
            return False
        
        try:
            self.question_tree.to_csv(output_csv)
            print(f"Results saved to {output_csv}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False


def main():
    """Command-line interface for AuditProcessor."""
    
    parser = argparse.ArgumentParser(description="Process audit questions with various approaches")
    
    parser.add_argument("--questions", type=str, required=True,
                        help="Path to CSV file with questions")
    parser.add_argument("--output", type=str, default="audit_results.csv",
                        help="Path to output CSV file")
    parser.add_argument("--bucket", type=str, required=True,
                        help="S3 bucket containing the document")
    parser.add_argument("--key", type=str, required=True,
                        help="S3 key for the document")
    parser.add_argument("--approaches", type=str, default="rag,memory,hyde",
                        help="Comma-separated list of approaches to use (rag, memory, hyde)")
    parser.add_argument("--question-ids", type=str, default=None,
                        help="Comma-separated list of question IDs to process (default: all)")
    parser.add_argument("--region", type=str, default="us-gov-west-1",
                        help="AWS region for Bedrock")
    parser.add_argument("--profile", type=str, default=None,
                        help="AWS profile to use (overrides environment variables)")
    parser.add_argument("--force-reprocess", action="store_true",
                        help="Force reprocessing of the document even if a processed version exists")
    parser.add_argument("--use-rag-query-rewriting", action="store_true",
                        help="Use query rewriting for the RAG approach to optimize vector search")
    parser.add_argument("--show-accuracy", action="store_true",
                        help="Calculate and display accuracy scores (requires 'truth' column in CSV)")
    
    args = parser.parse_args()
    
    # Get AWS credentials - prioritize profile if specified
    if args.profile:
        try:
            import boto3
            session = boto3.Session(profile_name=args.profile)
            credentials = session.get_credentials()
            if credentials:
                aws_access_key_id = credentials.access_key
                aws_secret_access_key = credentials.secret_key
                aws_session_token = credentials.token
                if not args.region:
                    args.region = session.region_name or args.region
            else:
                print(f"Error: Could not get credentials for profile: {args.profile}")
                return 1
        except Exception as e:
            print(f"Error using AWS profile {args.profile}: {e}")
            return 1
    else:
        # Get AWS credentials from environment
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
    
    # Parse approaches
    approaches = [a.strip() for a in args.approaches.split(",")]
    
    # Initialize processor
    processor = AuditProcessor(
        aws_region=args.region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        use_rag_query_rewriting=args.use_rag_query_rewriting
    )
    
    # Load document
    if not processor.load_document(args.bucket, args.key, force_reprocess=args.force_reprocess):
        print("Failed to load document")
        return 1
    
    # Load questions
    if not processor.load_questions(args.questions):
        print("Failed to load questions")
        return 1
    
    # Process questions
    if args.question_ids:
        question_ids = [q.strip() for q in args.question_ids.split(",")]
        success = processor.process_question_batch(
            question_ids, 
            approaches, 
            use_rag_query_rewriting=args.use_rag_query_rewriting
        )
    else:
        success = processor.process_all_questions(
            approaches,
            use_rag_query_rewriting=args.use_rag_query_rewriting
        )
    
    if not success:
        print("Some questions could not be processed")
    
    # Save results
    if not processor.save_results(args.output):
        print("Failed to save results")
        return 1
    
    # Calculate and display accuracy scores if requested
    if args.show_accuracy:
        print("\n" + "="*50)
        print("ACCURACY SCORES")
        print("="*50)
        
        approach_names = {'rag': 'RAG', 'memory': 'Memory', 'hyde': 'HYDE'}
        accuracy_scores = {}
        
        for approach in approaches:
            correct = 0
            total = 0
            
            for question in processor.question_tree.get_all_questions():
                # Skip questions without truth values
                if question.truth.value == "Unknown":
                    continue
                
                # Get the answer for this approach
                if approach == 'rag':
                    answer = question.rag_answer
                elif approach == 'memory':
                    answer = question.context_answer
                elif approach == 'hyde':
                    answer = question.hyde_answer
                else:
                    continue
                
                # Skip questions that weren't processed with this approach
                if answer is None:
                    continue
                
                total += 1
                if answer == question.truth:
                    correct += 1
            
            if total > 0:
                accuracy = correct / total
                accuracy_scores[approach] = accuracy
                display_name = approach_names.get(approach, approach.upper())
                print(f"{display_name:8s}: {correct:3d}/{total:3d} = {accuracy:6.1%}")
            else:
                print(f"{approach_names.get(approach, approach.upper()):8s}: No questions with truth values processed")
        
        if accuracy_scores:
            best_approach = max(accuracy_scores.items(), key=lambda x: x[1])
            print("\n" + "-"*50)
            print(f"Best performing approach: {approach_names.get(best_approach[0], best_approach[0].upper())} ({best_approach[1]:.1%})")
        
        print("="*50)
    
    return 0


if __name__ == "__main__":
    exit(main()) 