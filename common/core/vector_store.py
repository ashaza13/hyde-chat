"""
Vector store implementation using Langchain Chroma for document storage and retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain.schema import Document
from textract_processor import TextChunk


class ChromaVectorStore:
    """
    A vector store implementation using Langchain Chroma for storing and retrieving document chunks.
    """
    
    def __init__(
        self,
        collection_name: str = "audit_documents",
        embedding_model_name: str = "amazon.titan-embed-text-v2:0",
        persist_directory: Optional[str] = None,
        aws_region: str = "us-gov-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None
    ):
        """
        Initialize the Chroma vector store.
        
        Args:
            collection_name: Name of the collection to store documents
            embedding_model_name: Name of the embedding model to use
            persist_directory: Directory to persist the vector store (optional)
            aws_region: AWS region for Bedrock
            aws_access_key_id: AWS access key ID (optional if using IAM roles)
            aws_secret_access_key: AWS secret access key (optional if using IAM roles)
            aws_session_token: AWS session token (optional, used for temporary credentials)
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Initialize BedrockEmbeddings with credentials
        embedding_kwargs = {
            "model_id": embedding_model_name,
            "region_name": aws_region
        }
        
        # Add credentials if provided
        if aws_access_key_id and aws_secret_access_key:
            embedding_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
            if aws_session_token:
                embedding_kwargs["aws_session_token"] = aws_session_token
        
        self.embeddings = BedrockEmbeddings(**embedding_kwargs)
        
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
            # Create metadata for the chunk - convert lists to strings for ChromaDB compatibility
            metadata = {
                "chunk_id": i,
                "page_numbers_str": ",".join(map(str, chunk.page_numbers)),  # Convert list to comma-separated string
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "chunk_index": chunk.chunk_index,
                "page_range": chunk.get_page_range_str(),
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
            # Reconstruct page_numbers list from comma-separated string
            page_numbers_str = doc.metadata.get("page_numbers_str", "")
            page_numbers = [int(p) for p in page_numbers_str.split(",") if p.strip()] if page_numbers_str else []
            
            # Reconstruct TextChunk from Document with all required parameters
            chunk = TextChunk(
                text=doc.page_content,
                page_numbers=page_numbers,
                start_page=doc.metadata.get("start_page", page_numbers[0] if page_numbers else 1),
                end_page=doc.metadata.get("end_page", page_numbers[-1] if page_numbers else 1),
                chunk_index=doc.metadata.get("chunk_index", 0)
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
    
    def clear(self) -> None:
        """
        Clear all documents from the collection without deleting it.
        Alias for clear_collection for backward compatibility.
        """
        self.clear_collection()
    
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