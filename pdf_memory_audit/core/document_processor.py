import io
from typing import Optional, Dict, Any
import boto3
from PyPDF2 import PdfReader
from textract_processor import TextractProcessor, TextractResult

class DocumentProcessor:
    """Class for processing financial documents and storing full text in memory."""
    
    def __init__(self, aws_region: str = "us-gov-west-1"):
        """
        Initialize the DocumentProcessor.
        
        Args:
            aws_region: AWS region to use
        """
        self.document_text = ""
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.textract_processor = TextractProcessor(aws_region=aws_region)
        self.textract_result = None

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
            # First try to process with Textract
            try:
                self.textract_result = self.textract_processor.process_document(
                    bucket_name=bucket_name,
                    key=key,
                    extract_tables=True
                )
                
                # Get text with tables converted to markdown
                self.document_text = self.textract_result.get_text_with_tables()
                return True
                
            except Exception as e:
                print(f"Error using Textract: {e}. Falling back to PyPDF2.")
                
                # Fall back to PyPDF2 if Textract fails
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