"""
Vector store implementation using Langchain Chroma for document storage and retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from textract_processor import TextChunk


class ChromaVectorStore:
    """
    A vector store implementation using Langchain Chroma for storing and retrieving document chunks.
    """
    
    def __init__(
        self,
        collection_name: str = "audit_documents",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the Chroma vector store.
        
        Args:
            collection_name: Name of the collection to store documents
            embedding_model_name: Name of the embedding model to use
            persist_directory: Directory to persist the vector store (optional)
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=embedding_model_name
        )
        
        # Set up persist directory
        if persist_directory is None:
            persist_directory = os.path.join(os.getcwd(), "chroma_db")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
    
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of TextChunk objects to add
        """
        documents = []
        
        for i, chunk in enumerate(chunks):
            # Create metadata for the chunk
            metadata = {
                "chunk_id": i,
                "page_numbers": chunk.page_numbers,
                "page_range": chunk.get_page_range_str(),
                "chunk_type": getattr(chunk, 'chunk_type', 'text'),
                "source": getattr(chunk, 'source', 'unknown')
            }
            
            # Create Langchain Document
            doc = Document(
                page_content=chunk.text,
                metadata=metadata
            )
            documents.append(doc)
        
        # Add documents to vector store
        self.vectorstore.add_documents(documents)
    
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[TextChunk, float]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of tuples (TextChunk, similarity_score)
        """
        # Perform similarity search with scores
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        # Convert results back to TextChunk format
        chunk_results = []
        for doc, score in results:
            # Reconstruct TextChunk from Document
            chunk = TextChunk(
                text=doc.page_content,
                page_numbers=doc.metadata.get("page_numbers", []),
                chunk_type=doc.metadata.get("chunk_type", "text")
            )
            
            # Add source if available
            if "source" in doc.metadata:
                chunk.source = doc.metadata["source"]
            
            chunk_results.append((chunk, float(score)))
        
        return chunk_results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 3,
        alpha: float = 0.7,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[TextChunk, float]]:
        """
        Perform hybrid search combining semantic similarity and keyword matching.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            alpha: Weight for semantic search (1-alpha for keyword search)
            filter_dict: Optional metadata filter
            
        Returns:
            List of tuples (TextChunk, similarity_score)
        """
        # For now, fallback to regular similarity search
        # In a full implementation, this would combine vector search with BM25 or similar
        return self.similarity_search(query, k, filter_dict)
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection.
        """
        self.vectorstore.delete_collection()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get the collection
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "embedding_model": self.embedding_model_name,
                "persist_directory": str(self.persist_directory)
            }
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection without deleting it.
        """
        try:
            # Delete and recreate the collection
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
        except Exception as e:
            print(f"Warning: Could not clear collection: {e}") 