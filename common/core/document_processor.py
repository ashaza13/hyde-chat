import io
import json
import os
from typing import Optional, Dict, Any, Tuple
import boto3
from PyPDF2 import PdfReader
from textract_processor import TextractProcessor, TextractResult


class DocumentProcessor:
    """Centralized class for processing financial documents and storing results in S3."""
    
    def __init__(
        self, 
        aws_region: str = "us-gov-west-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None
    ):
        """
        Initialize the DocumentProcessor.
        
        Args:
            aws_region: AWS region to use
            aws_access_key_id: AWS access key ID (optional if using IAM roles)
            aws_secret_access_key: AWS secret access key (optional if using IAM roles)
            aws_session_token: AWS session token (optional, used for temporary credentials)
        """
        # Create session with optional credentials
        session_kwargs = {"region_name": aws_region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
            # Add session token if provided
            if aws_session_token:
                session_kwargs["aws_session_token"] = aws_session_token
        
        self.aws_region = aws_region
        session = boto3.Session(**session_kwargs)
        self.s3_client = session.client('s3')
        
        # Create TextractProcessor with the same credentials
        self.textract_processor = TextractProcessor(
            aws_region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        
        self.document_text = ""
        self.textract_result = None

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
            
    def get_processed_document_key(self, original_key: str) -> str:
        """
        Generate the S3 key for the processed document.
        
        Args:
            original_key: Original document key
            
        Returns:
            Key for the processed document
        """
        filename = os.path.basename(original_key)
        base_name, _ = os.path.splitext(filename)
        return f"processed_statements/{base_name}/textract_result.json"
        
    def check_processed_document_exists(self, bucket_name: str, original_key: str) -> bool:
        """
        Check if a processed document already exists in S3.
        
        Args:
            bucket_name: Name of the S3 bucket
            original_key: Original document key
            
        Returns:
            True if the processed document exists, False otherwise
        """
        processed_key = self.get_processed_document_key(original_key)
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=processed_key)
            return True
        except Exception:
            return False
            
    def save_textract_result_to_s3(self, textract_result: TextractResult, bucket_name: str, original_key: str) -> bool:
        """
        Save the Textract result to S3.
        
        Args:
            textract_result: The Textract result to save
            bucket_name: Name of the S3 bucket
            original_key: Original document key
            
        Returns:
            True if successful, False otherwise
        """
        processed_key = self.get_processed_document_key(original_key)
        
        try:
            # Serialize the TextractResult to a dictionary
            result_dict = self._textract_result_to_dict(textract_result)
            
            # Upload the result to S3
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=processed_key,
                Body=json.dumps(result_dict),
                ContentType='application/json'
            )
            
            return True
        except Exception as e:
            print(f"Error saving Textract result to S3: {e}")
            return False
            
    def load_textract_result_from_s3(self, bucket_name: str, original_key: str) -> Optional[TextractResult]:
        """
        Load a Textract result from S3.
        
        Args:
            bucket_name: Name of the S3 bucket
            original_key: Original document key
            
        Returns:
            The Textract result if successful, None otherwise
        """
        processed_key = self.get_processed_document_key(original_key)
        
        try:
            # Download the result from S3
            response = self.s3_client.get_object(Bucket=bucket_name, Key=processed_key)
            result_dict = json.loads(response['Body'].read().decode('utf-8'))
            
            # Create a TextractResult from the dictionary
            return self._dict_to_textract_result(result_dict)
        except Exception as e:
            print(f"Error loading Textract result from S3: {e}")
            return None
            
    def process_document(
        self, 
        bucket_name: str, 
        key: str, 
        force_reprocess: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Process a document and store the result in S3.
        
        Args:
            bucket_name: Name of the S3 bucket
            key: Key of the PDF file in the S3 bucket
            force_reprocess: Whether to force reprocessing even if a processed document exists
            
        Returns:
            A tuple containing:
            - True if successful, False otherwise
            - The full text of the document if successful, None otherwise
        """
        try:
            # Check if a processed document already exists
            if not force_reprocess and self.check_processed_document_exists(bucket_name, key):
                print(f"Loading processed document from S3")
                self.textract_result = self.load_textract_result_from_s3(bucket_name, key)
                
                if self.textract_result:
                    self.document_text = self.textract_result.get_text_with_tables()
                    return True, self.document_text
            
            # Process with Textract if needed
            print(f"Processing document with Textract")
            try:
                self.textract_result = self.textract_processor.process_document(
                    bucket_name=bucket_name,
                    key=key,
                    extract_tables=True
                )
                
                # Save the result to S3
                if not self.save_textract_result_to_s3(self.textract_result, bucket_name, key):
                    print("Warning: Failed to save Textract result to S3")
                
                # Get text with tables converted to markdown
                self.document_text = self.textract_result.get_text_with_tables()
                return True, self.document_text
                
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
                
                return True, self.document_text
        except Exception as e:
            print(f"Error processing document: {e}")
            return False, None
    
    def get_document_text(self) -> str:
        """
        Get the full text of the loaded document.
        
        Returns:
            The full text of the loaded document
        """
        return self.document_text
        
    def _textract_result_to_dict(self, result: TextractResult) -> Dict[str, Any]:
        """
        Convert a TextractResult object to a dictionary for JSON serialization.
        
        Args:
            result: The TextractResult to convert
            
        Returns:
            A dictionary representation of the TextractResult
        """
        text_blocks = []
        for block in result.text_blocks:
            bounding_box = None
            if block.bounding_box:
                bounding_box = {
                    "width": block.bounding_box.width,
                    "height": block.bounding_box.height,
                    "left": block.bounding_box.left,
                    "top": block.bounding_box.top
                }
                
            text_blocks.append({
                "id": block.id,
                "type": block.type,
                "text": block.text,
                "confidence": block.confidence,
                "page": block.page,
                "bounding_box": bounding_box,
                "parent_id": block.parent_id
            })
            
        tables = []
        for table in result.tables:
            table_bounding_box = None
            if table.bounding_box:
                table_bounding_box = {
                    "width": table.bounding_box.width,
                    "height": table.bounding_box.height,
                    "left": table.bounding_box.left,
                    "top": table.bounding_box.top
                }
                
            cells = []
            for cell in table.cells:
                cell_bounding_box = None
                if cell.bounding_box:
                    cell_bounding_box = {
                        "width": cell.bounding_box.width,
                        "height": cell.bounding_box.height,
                        "left": cell.bounding_box.left,
                        "top": cell.bounding_box.top
                    }
                    
                cells.append({
                    "id": cell.id,
                    "row_index": cell.row_index,
                    "column_index": cell.column_index,
                    "row_span": cell.row_span,
                    "column_span": cell.column_span,
                    "text": cell.text,
                    "confidence": cell.confidence,
                    "page": cell.page,
                    "bounding_box": cell_bounding_box
                })
                
            tables.append({
                "id": table.id,
                "page": table.page,
                "cells": cells,
                "row_count": table.row_count,
                "column_count": table.column_count,
                "bounding_box": table_bounding_box
            })
            
        return {
            "text_blocks": text_blocks,
            "tables": tables
        }
        
    def _dict_to_textract_result(self, data: Dict[str, Any]) -> TextractResult:
        """
        Create a TextractResult object from a dictionary.
        
        Args:
            data: Dictionary representation of the TextractResult
            
        Returns:
            A TextractResult object
        """
        from textract_processor import TextBlock, TableBlock, CellBlock, BoundingBox, BlockType
        
        text_blocks = []
        for block_data in data.get("text_blocks", []):
            bounding_box = None
            if block_data.get("bounding_box"):
                box_data = block_data["bounding_box"]
                bounding_box = BoundingBox(
                    width=box_data["width"],
                    height=box_data["height"],
                    left=box_data["left"],
                    top=box_data["top"]
                )
                
            text_blocks.append(TextBlock(
                id=block_data["id"],
                type=block_data["type"],
                text=block_data["text"],
                confidence=block_data["confidence"],
                page=block_data["page"],
                bounding_box=bounding_box,
                parent_id=block_data.get("parent_id")
            ))
            
        tables = []
        for table_data in data.get("tables", []):
            table_bounding_box = None
            if table_data.get("bounding_box"):
                box_data = table_data["bounding_box"]
                table_bounding_box = BoundingBox(
                    width=box_data["width"],
                    height=box_data["height"],
                    left=box_data["left"],
                    top=box_data["top"]
                )
                
            cells = []
            for cell_data in table_data.get("cells", []):
                cell_bounding_box = None
                if cell_data.get("bounding_box"):
                    box_data = cell_data["bounding_box"]
                    cell_bounding_box = BoundingBox(
                        width=box_data["width"],
                        height=box_data["height"],
                        left=box_data["left"],
                        top=box_data["top"]
                    )
                    
                cells.append(CellBlock(
                    id=cell_data["id"],
                    row_index=cell_data["row_index"],
                    column_index=cell_data["column_index"],
                    row_span=cell_data["row_span"],
                    column_span=cell_data["column_span"],
                    text=cell_data["text"],
                    confidence=cell_data["confidence"],
                    page=cell_data["page"],
                    bounding_box=cell_bounding_box
                ))
                
            tables.append(TableBlock(
                id=table_data["id"],
                page=table_data["page"],
                cells=cells,
                row_count=table_data["row_count"],
                column_count=table_data["column_count"],
                bounding_box=table_bounding_box
            ))
            
        return TextractResult(text_blocks=text_blocks, tables=tables) 