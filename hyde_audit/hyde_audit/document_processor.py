import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class DocumentProcessor:
    """Class for processing financial documents and creating embeddings."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the DocumentProcessor.
        
        Args:
            embedding_model_name: Name of the SentenceTransformer model to use for embeddings
        """
        self.documents = []
        self.embeddings = []
        self.embedding_model_name = embedding_model_name
        
        # Initialize the embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        else:
            self.embedding_model = None
            print("Warning: sentence-transformers not available. Install with 'pip install sentence-transformers'")
    
    def load_document(self, file_path: Union[str, Path]) -> bool:
        """
        Load a document from a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist")
            return False
        
        # Currently support simple text files
        # TODO: Add support for PDF, Excel, etc. using appropriate libraries
        if file_path.suffix.lower() in ['.txt', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks - simple approach, can be improved
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
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return False
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
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
    
    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        """
        Get the most relevant context for a query using embeddings.
        
        Args:
            query: The query to find context for
            top_k: Number of most relevant chunks to return
            
        Returns:
            Combined relevant context as a string
        """
        if not self.documents:
            return ""
        
        if self.embedding_model is None:
            # Fallback to simple keyword matching if no embedding model
            return self._keyword_search(query, top_k)
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarity scores
        import numpy as np
        from scipy.spatial.distance import cosine
        
        similarities = []
        for doc_embedding in self.embeddings:
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # Get top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Combine the relevant context
        relevant_context = "\n\n".join([self.documents[i] for i in top_indices])
        return relevant_context
    
    def _keyword_search(self, query: str, top_k: int = 3) -> str:
        """
        Fallback method for finding relevant context using simple keyword matching.
        
        Args:
            query: The query to find context for
            top_k: Number of most relevant chunks to return
            
        Returns:
            Combined relevant context as a string
        """
        # Simple keyword matching
        query_words = set(query.lower().split())
        scores = []
        
        for chunk in self.documents:
            chunk_words = set(chunk.lower().split())
            # Score based on word overlap
            overlap = len(query_words.intersection(chunk_words))
            scores.append(overlap / len(query_words) if query_words else 0)
        
        # Get top_k chunks
        import numpy as np
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Combine the relevant context
        relevant_context = "\n\n".join([self.documents[i] for i in top_indices])
        return relevant_context 