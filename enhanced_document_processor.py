#!/usr/bin/env python3
"""
Enhanced Document Processor that preserves page number metadata for LLM rationale.
This shows how to modify existing RAG systems to include page number information.
"""

import boto3
import io
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.spatial.distance import cosine

from textract_processor import TextractProcessor, TextractResult, TextChunk

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


class EnhancedDocumentProcessor:
    """Enhanced Document Processor that preserves page number metadata for LLM rationale."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", aws_region: str = "us-gov-west-1"):
        """
        Initialize the Enhanced DocumentProcessor.
        
        Args:
            embedding_model_name: Name of the SentenceTransformer model to use for embeddings
            aws_region: AWS region to use
        """
        self.chunks_with_metadata: List[TextChunk] = []
        self.embeddings = []
        self.embedding_model_name = embedding_model_name
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.textract_processor = TextractProcessor(aws_region=aws_region)
        self.textract_result = None
        
        # Initialize the embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        else:
            self.embedding_model = None
            print("Warning: sentence-transformers not available. Install with 'pip install sentence-transformers'")

    def download_pdf(self, bucket_name: str, key: str) -> io.BytesIO:
        """
        Download a PDF file from S3 and return the content as a BytesIO object.

        Args:
            bucket_name: Name of the S3 bucket
            key: Key of the PDF file in the S3 bucket
            
        Returns:
            Content of the PDF file as a BytesIO object
        """
        response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
        pdf_content = response['Body'].read()
        return io.BytesIO(pdf_content)
            
    def load_document(self, bucket_name: str, key: str, chunk_size: int = 1000, overlap: int = 200) -> bool:
        """
        Load a document from S3, process it into chunks with metadata, and create embeddings.
        
        Args:
            bucket_name: Name of the S3 bucket
            key: Key of the PDF file in the S3 bucket
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First try to process with Textract
            try:
                self.textract_result = self.textract_processor.process_document(
                    bucket_name=bucket_name,
                    key=key,
                    extract_tables=True
                )
                
                # Get chunked text with metadata preserved
                self.chunks_with_metadata = self.textract_result.get_chunked_text_with_metadata(
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                
                print(f"Created {len(self.chunks_with_metadata)} chunks with page metadata")
                
            except Exception as e:
                print(f"Error using Textract: {e}. Falling back to PyPDF2.")
                
                if not PYPDF2_AVAILABLE:
                    print("PyPDF2 not available. Install with 'pip install PyPDF2'")
                    return False
                
                # Fall back to PyPDF2 if Textract fails
                pdf_bytes = self.download_pdf(bucket_name, key)
                pdf_reader = PdfReader(pdf_bytes)
                
                # Create chunks without detailed metadata (fallback)
                content = ""
                page_content = {}
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    page_content[page_num] = page_text
                    content += page_text
                
                # Simple chunking with basic page tracking
                self.chunks_with_metadata = self._create_fallback_chunks(
                    page_content, chunk_size, overlap
                )
            
            # Create embeddings if model is available
            if self.embedding_model is not None:
                chunk_texts = [chunk.text for chunk in self.chunks_with_metadata]
                self.embeddings = self.embedding_model.encode(chunk_texts)
            
            return True
        except Exception as e:
            print(f"Error loading document: {e}")
            return False
    
    def load_local_document(self, file_path: str, chunk_size: int = 1000, overlap: int = 200) -> bool:
        """
        Load a local document, process it into chunks with metadata, and create embeddings.
        
        Args:
            file_path: Path to the local PDF file
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process with Textract
            self.textract_result = self.textract_processor.process_local_document(
                file_path=file_path,
                extract_tables=True
            )
            
            # Get chunked text with metadata preserved
            self.chunks_with_metadata = self.textract_result.get_chunked_text_with_metadata(
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            print(f"Created {len(self.chunks_with_metadata)} chunks with page metadata")
            
            # Create embeddings if model is available
            if self.embedding_model is not None:
                chunk_texts = [chunk.text for chunk in self.chunks_with_metadata]
                self.embeddings = self.embedding_model.encode(chunk_texts)
            
            return True
        except Exception as e:
            print(f"Error loading local document: {e}")
            return False
    
    def _create_fallback_chunks(self, page_content: Dict[int, str], chunk_size: int, overlap: int) -> List[TextChunk]:
        """
        Create chunks with basic page metadata when Textract is not available.
        
        Args:
            page_content: Dictionary mapping page numbers to text content
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of TextChunk objects with basic page metadata
        """
        chunks = []
        chunk_index = 0
        
        for page_num, text in page_content.items():
            if len(text) <= chunk_size:
                # Entire page fits in one chunk
                chunks.append(TextChunk(
                    text=text,
                    page_numbers=[page_num],
                    start_page=page_num,
                    end_page=page_num,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            else:
                # Split page into multiple chunks
                start = 0
                while start < len(text):
                    end = min(start + chunk_size, len(text))
                    
                    # Try to find a good breaking point
                    if end < len(text):
                        sentence_break = text.rfind('. ', start, end)
                        if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                            end = sentence_break + 1
                    
                    chunk_text = text[start:end]
                    chunks.append(TextChunk(
                        text=chunk_text,
                        page_numbers=[page_num],
                        start_page=page_num,
                        end_page=page_num,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                    start = end - overlap if end < len(text) else end
        
        return chunks
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        """
        Get the most relevant document chunks for a query using semantic search.
        
        Args:
            query: The query to find context for
            top_k: Number of most relevant chunks to return
            
        Returns:
            List of tuples with (TextChunk, similarity_score)
        """
        if not self.chunks_with_metadata:
            return []
        
        if self.embedding_model is None:
            # Fallback to simple keyword matching if no embedding model
            return self._keyword_search(query, top_k)
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarity scores
        similarities = []
        for doc_embedding in self.embeddings:
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # Get top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return the chunks and their scores
        results = [(self.chunks_with_metadata[i], similarities[i]) for i in top_indices]
        return results
    
    def _keyword_search(self, query: str, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        """
        Fallback method for finding relevant context using simple keyword matching.
        
        Args:
            query: The query to find context for
            top_k: Number of most relevant chunks to return
            
        Returns:
            List of tuples with (TextChunk, similarity_score)
        """
        # Simple keyword matching
        query_words = set(query.lower().split())
        scores = []
        
        for chunk in self.chunks_with_metadata:
            chunk_words = set(chunk.text.lower().split())
            # Score based on word overlap
            overlap = len(query_words.intersection(chunk_words))
            score = overlap / len(query_words) if query_words else 0
            scores.append(score)
        
        # Get top_k chunks
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return the chunks and their scores
        results = [(self.chunks_with_metadata[i], scores[i]) for i in top_indices]
        return results
    
    def generate_llm_context_with_citations(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Generate context for LLM with proper page citations.
        
        Args:
            query: The query to find context for
            top_k: Number of most relevant chunks to return
            
        Returns:
            Dictionary with context and citation information
        """
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k)
        
        if not relevant_chunks:
            return {
                "context": "No relevant information found.",
                "citations": [],
                "page_references": []
            }
        
        context_parts = []
        citations = []
        page_references = set()
        
        for i, (chunk, score) in enumerate(relevant_chunks):
            # Add context with citation marker
            context_parts.append(f"[{i+1}] {chunk.text}")
            
            # Add citation information
            citations.append({
                "citation_id": i+1,
                "page_range": chunk.get_page_range_str(),
                "pages": chunk.page_numbers,
                "confidence": score,
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            })
            
            # Track all referenced pages
            page_references.update(chunk.page_numbers)
        
        return {
            "context": "\n\n".join(context_parts),
            "citations": citations,
            "page_references": sorted(list(page_references)),
            "instruction_for_llm": f"Please provide your response based on the above context. When referencing specific information, cite the source using the format 'Based on information from [citation_id] ({page_range})' where citation_id corresponds to the numbered sections above."
        }
    
    def get_document_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the processed document.
        
        Returns:
            Dictionary with document statistics
        """
        if not self.chunks_with_metadata:
            return {"error": "No document loaded"}
        
        all_pages = set()
        total_text_length = 0
        
        for chunk in self.chunks_with_metadata:
            all_pages.update(chunk.page_numbers)
            total_text_length += len(chunk.text)
        
        return {
            "total_chunks": len(self.chunks_with_metadata),
            "total_pages": len(all_pages),
            "page_range": f"{min(all_pages)}-{max(all_pages)}" if all_pages else "N/A",
            "total_text_length": total_text_length,
            "average_chunk_size": total_text_length // len(self.chunks_with_metadata) if self.chunks_with_metadata else 0,
            "has_embeddings": len(self.embeddings) > 0
        }


def main():
    """Example usage of the Enhanced Document Processor."""
    
    # Initialize processor
    processor = EnhancedDocumentProcessor()
    
    # Load a local document
    if processor.load_local_document("fy_2024.pdf", chunk_size=800, overlap=100):
        print("Document loaded successfully!")
        
        # Get document summary
        summary = processor.get_document_summary()
        print(f"\nDocument Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Example query
        query = "financial performance and revenue"
        print(f"\nQuery: '{query}'")
        
        # Get context with citations
        result = processor.generate_llm_context_with_citations(query, top_k=3)
        
        print(f"\nGenerated Context with Citations:")
        print(f"Pages referenced: {result['page_references']}")
        print(f"\nContext:\n{result['context'][:500]}...")
        
        print(f"\nCitations:")
        for citation in result['citations']:
            print(f"  [{citation['citation_id']}] {citation['page_range']} (confidence: {citation['confidence']:.3f})")
            print(f"      Preview: {citation['text_preview'][:100]}...")
        
        print(f"\nInstruction for LLM:")
        print(result['instruction_for_llm'])
        
        # Example of how LLM would respond with citations
        print(f"\n" + "="*50)
        print("EXAMPLE LLM RESPONSE WITH CITATIONS:")
        print("="*50)
        print(f"Based on information from [1] ({result['citations'][0]['page_range']}), the company's financial performance shows...")
        print(f"Additionally, as noted in [2] ({result['citations'][1]['page_range']}), the revenue trends indicate...")
    else:
        print("Failed to load document")


if __name__ == "__main__":
    main() 