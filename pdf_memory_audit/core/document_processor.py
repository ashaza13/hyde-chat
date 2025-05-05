import io
from typing import Optional
import boto3
from PyPDF2 import PdfReader

class DocumentProcessor:
    """Class for processing financial documents and storing full text in memory."""
    
    def __init__(self):
        """Initialize the DocumentProcessor."""
        self.document_text = ""
        self.s3_client = boto3.client('s3', region_name='us-gov-west-1')

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
        Load a document from S3 and store its full text in memory.
        
        Args:
            bucket_name: Name of the S3 bucket
            key: Key of the PDF file in the S3 bucket
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_bytes = self.download_pdf(bucket_name, key)
            pdf_reader = PdfReader(pdf_bytes)
            
            # Extract and concatenate text from all pages
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
            
            # Store the full document text in memory
            self.document_text = content
            
            return True
        except Exception as e:
            print(f"Error loading document: {e}")
            return False
    
    def get_document_text(self) -> str:
        """
        Get the full text of the loaded document.
        
        Returns:
            The full text of the loaded document
        """
        return self.document_text 