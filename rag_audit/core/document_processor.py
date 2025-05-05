import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
from PyPDF2 import PdfReader
import boto3
import io
import numpy as np
from scipy.spatial.distance import cosine

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class DocumentProcessor:
    """Class for processing financial documents and performing RAG operations."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the DocumentProcessor.
        
        Args:
            embedding_model_name: Name of the SentenceTransformer model to use for embeddings
        """
        self.documents = []
        self.embeddings = []
        self.embedding_model_name = embedding_model_name
        self.s3_client = boto3.client('s3', region_name='us-gov-west-1')
        
        # Initialize the embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        else:
            self.embedding_model = None
            print("Warning: sentence-transformers not available. Install with 'pip install sentence-transformers'")

    def download_pdf(self, bucket_name, key):
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
            
    def load_document(self, bucket_name: str, key: str) -> bool:
        """
        Load a document from S3, process it into chunks, and create embeddings.
        
        Args:
            bucket_name: Name of the S3 bucket
            key: Key of the PDF file in the S3 bucket
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_bytes = self.download_pdf(bucket_name, key)
            pdf_reader = PdfReader(pdf_bytes)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
            
            # Split into chunks
            chunks = self._chunk_text(content)
            self.documents.extend(chunks)
            
            # Create embeddings if model is available
            if self.embedding_model is not None:
                new_embeddings = self.embedding_model.encode(chunks)
                self.embeddings.extend(new_embeddings)
            
            return True
        except Exception as e:
            print(f"Error loading document: {e}")
            return False
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to find a good breaking point
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break
                else:
                    # Look for sentence break
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 1  # Include the period
            
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get the most relevant document chunks for a query using semantic search.
        
        Args:
            query: The query to find context for
            top_k: Number of most relevant chunks to return
            
        Returns:
            List of tuples with (chunk_text, similarity_score)
        """
        if not self.documents:
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
        results = [(self.documents[i], similarities[i]) for i in top_indices]
        return results
    
    def _keyword_search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Fallback method for finding relevant context using simple keyword matching.
        
        Args:
            query: The query to find context for
            top_k: Number of most relevant chunks to return
            
        Returns:
            List of tuples with (chunk_text, similarity_score)
        """
        # Simple keyword matching
        query_words = set(query.lower().split())
        scores = []
        
        for chunk in self.documents:
            chunk_words = set(chunk.lower().split())
            # Score based on word overlap
            overlap = len(query_words.intersection(chunk_words))
            score = overlap / len(query_words) if query_words else 0
            scores.append(score)
        
        # Get top_k chunks
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return the chunks and their scores
        results = [(self.documents[i], scores[i]) for i in top_indices]
        return results 