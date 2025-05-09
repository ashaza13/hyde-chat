import io
import os
import time
import boto3
from typing import List, Dict, Optional, Any, Tuple, Union

from .models import (
    TextractResult, 
    TextBlock, 
    TableBlock, 
    CellBlock, 
    BoundingBox, 
    BlockType
)

class TextractProcessor:
    """
    Process PDF documents using AWS Textract to extract text and tables.
    """
    
    def __init__(
        self,
        aws_region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """
        Initialize the TextractProcessor.
        
        Args:
            aws_region: AWS region to use
            aws_access_key_id: AWS access key ID (optional if using IAM roles)
            aws_secret_access_key: AWS secret access key (optional if using IAM roles)
        """
        # Initialize AWS clients
        self.s3_client = boto3.client(
            's3',
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        self.textract_client = boto3.client(
            'textract',
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    
    def process_document(
        self, 
        bucket_name: str, 
        key: str,
        extract_tables: bool = True
    ) -> TextractResult:
        """
        Process a document using AWS Textract.
        
        Args:
            bucket_name: S3 bucket name
            key: S3 object key
            extract_tables: Whether to extract tables from the document
            
        Returns:
            TextractResult with extracted text and tables
        """
        # Start asynchronous text and table extraction job
        job_id = self._start_extraction_job(bucket_name, key, extract_tables)
        
        # Wait for job to complete
        response = self._wait_for_job_completion(job_id)
        
        # Process results
        if response['JobStatus'] == 'SUCCEEDED':
            return self._process_extraction_results(job_id)
        else:
            raise Exception(f"Textract job failed with status: {response['JobStatus']}")
    
    def process_local_document(
        self, 
        file_path: str,
        extract_tables: bool = True,
        upload_bucket: Optional[str] = None
    ) -> TextractResult:
        """
        Process a local document using AWS Textract.
        
        Args:
            file_path: Path to the local PDF file
            extract_tables: Whether to extract tables from the document
            upload_bucket: S3 bucket to upload the document to (required for documents > 5MB)
            
        Returns:
            TextractResult with extracted text and tables
        """
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        # For files less than 5MB, we can use synchronous API
        if file_size < 5 * 1024 * 1024:
            with open(file_path, 'rb') as file:
                file_bytes = file.read()
            
            return self._process_document_bytes(file_bytes, extract_tables)
        
        # For larger files, we need to upload to S3 and use asynchronous API
        elif upload_bucket:
            # Upload file to S3
            s3_key = f"textract-processor-uploads/{file_name}"
            self.s3_client.upload_file(file_path, upload_bucket, s3_key)
            
            # Process via S3
            result = self.process_document(upload_bucket, s3_key, extract_tables)
            
            # Clean up
            self.s3_client.delete_object(Bucket=upload_bucket, Key=s3_key)
            
            return result
        else:
            raise ValueError("Files larger than 5MB require an S3 bucket for processing. Please provide upload_bucket.")
    
    def _process_document_bytes(
        self, 
        document_bytes: bytes,
        extract_tables: bool = True
    ) -> TextractResult:
        """
        Process document bytes using synchronous Textract API.
        
        Args:
            document_bytes: PDF document as bytes
            extract_tables: Whether to extract tables
            
        Returns:
            TextractResult with extracted text and tables
        """
        # List of all detected blocks
        all_blocks = []
        
        # Call Textract API for document analysis if tables needed
        if extract_tables:
            response = self.textract_client.analyze_document(
                Document={'Bytes': document_bytes},
                FeatureTypes=['TABLES']
            )
            all_blocks.extend(response['Blocks'])
            
            # Handle pagination if needed
            next_token = response.get('NextToken')
            while next_token:
                response = self.textract_client.analyze_document(
                    Document={'Bytes': document_bytes},
                    FeatureTypes=['TABLES'],
                    NextToken=next_token
                )
                all_blocks.extend(response['Blocks'])
                next_token = response.get('NextToken')
        
        # Otherwise just detect text
        else:
            response = self.textract_client.detect_document_text(
                Document={'Bytes': document_bytes}
            )
            all_blocks.extend(response['Blocks'])
            
            # Handle pagination if needed
            next_token = response.get('NextToken')
            while next_token:
                response = self.textract_client.detect_document_text(
                    Document={'Bytes': document_bytes},
                    NextToken=next_token
                )
                all_blocks.extend(response['Blocks'])
                next_token = response.get('NextToken')
        
        # Process blocks into structured result
        return self._parse_blocks(all_blocks)
    
    def _start_extraction_job(
        self, 
        bucket_name: str, 
        key: str,
        extract_tables: bool = True
    ) -> str:
        """
        Start an asynchronous Textract job.
        
        Args:
            bucket_name: S3 bucket name
            key: S3 object key
            extract_tables: Whether to extract tables
            
        Returns:
            Job ID of the Textract job
        """
        if extract_tables:
            response = self.textract_client.start_document_analysis(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': key
                    }
                },
                FeatureTypes=['TABLES']
            )
        else:
            response = self.textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': key
                    }
                }
            )
        
        return response['JobId']
    
    def _wait_for_job_completion(self, job_id: str) -> Dict[str, Any]:
        """
        Wait for a Textract job to complete.
        
        Args:
            job_id: Job ID of the Textract job
            
        Returns:
            Response from Textract get_document_analysis/get_document_text_detection
        """
        # First call to determine which API to use
        response = self.textract_client.get_document_analysis(
            JobId=job_id
        )
        
        # If this doesn't fail, we're doing document analysis
        api_type = 'analysis'
        
        # If the above call fails, we're doing text detection
        try:
            if response['JobStatus'] == 'FAILED':
                raise Exception(f"Textract job failed with status: {response['JobStatus']}")
        except Exception:
            response = self.textract_client.get_document_text_detection(
                JobId=job_id
            )
            api_type = 'text_detection'
        
        # Wait for job to complete
        status = response['JobStatus']
        while status == 'IN_PROGRESS':
            time.sleep(5)
            
            if api_type == 'analysis':
                response = self.textract_client.get_document_analysis(
                    JobId=job_id
                )
            else:
                response = self.textract_client.get_document_text_detection(
                    JobId=job_id
                )
            
            status = response['JobStatus']
        
        return response
    
    def _process_extraction_results(self, job_id: str) -> TextractResult:
        """
        Process the results from a Textract job.
        
        Args:
            job_id: Job ID of the Textract job
            
        Returns:
            TextractResult with extracted text and tables
        """
        # List of all detected blocks
        all_blocks = []
        
        # Try document analysis first (for tables)
        try:
            response = self.textract_client.get_document_analysis(
                JobId=job_id
            )
            all_blocks.extend(response['Blocks'])
            
            # Get all pages
            while 'NextToken' in response:
                response = self.textract_client.get_document_analysis(
                    JobId=job_id,
                    NextToken=response['NextToken']
                )
                all_blocks.extend(response['Blocks'])
                
        except Exception:
            # Fall back to text detection
            response = self.textract_client.get_document_text_detection(
                JobId=job_id
            )
            all_blocks.extend(response['Blocks'])
            
            # Get all pages
            while 'NextToken' in response:
                response = self.textract_client.get_document_text_detection(
                    JobId=job_id,
                    NextToken=response['NextToken']
                )
                all_blocks.extend(response['Blocks'])
        
        # Process blocks into structured result
        return self._parse_blocks(all_blocks)
    
    def _parse_blocks(self, blocks: List[Dict[str, Any]]) -> TextractResult:
        """
        Parse Textract blocks into structured TextractResult.
        
        Args:
            blocks: List of Textract blocks
            
        Returns:
            TextractResult with extracted text and tables
        """
        text_blocks = []
        tables = []
        table_cells = {}  # Map table ID to list of cells
        
        # Process page blocks to get total pages
        page_blocks = [b for b in blocks if b.get('BlockType') == 'PAGE']
        total_pages = len(page_blocks)
        
        # First pass - collect text blocks and table/cell blocks
        for block in blocks:
            block_id = block.get('Id')
            block_type = block.get('BlockType')
            
            # Process text block
            if block_type in ['LINE', 'WORD']:
                text = block.get('Text', '')
                
                # Get page number
                page = 1
                if 'Page' in block:
                    page = block.get('Page', 1)
                
                # Get bounding box if available
                bounding_box = None
                if 'Geometry' in block and 'BoundingBox' in block['Geometry']:
                    bb = block['Geometry']['BoundingBox']
                    bounding_box = BoundingBox(
                        width=bb.get('Width', 0),
                        height=bb.get('Height', 0),
                        left=bb.get('Left', 0),
                        top=bb.get('Top', 0)
                    )
                
                # Create text block
                text_block = TextBlock(
                    id=block_id,
                    type=BlockType(block_type),
                    text=text,
                    confidence=block.get('Confidence', 0),
                    page=page,
                    bounding_box=bounding_box,
                    parent_id=block.get('ParentId')
                )
                
                text_blocks.append(text_block)
            
            # Process table block
            elif block_type == 'TABLE':
                # Get page number
                page = 1
                if 'Page' in block:
                    page = block.get('Page', 1)
                
                # Get bounding box if available
                bounding_box = None
                if 'Geometry' in block and 'BoundingBox' in block['Geometry']:
                    bb = block['Geometry']['BoundingBox']
                    bounding_box = BoundingBox(
                        width=bb.get('Width', 0),
                        height=bb.get('Height', 0),
                        left=bb.get('Left', 0),
                        top=bb.get('Top', 0)
                    )
                
                # Initialize table with empty cell list
                table = TableBlock(
                    id=block_id,
                    page=page,
                    cells=[],
                    row_count=0,
                    column_count=0,
                    bounding_box=bounding_box
                )
                
                tables.append(table)
                table_cells[block_id] = []
            
            # Process cell block
            elif block_type == 'CELL':
                parent_id = block.get('ParentId')
                
                # Get page number
                page = 1
                if 'Page' in block:
                    page = block.get('Page', 1)
                
                # Get cell position and span
                row_index = block.get('RowIndex', 1) - 1  # 0-indexed
                column_index = block.get('ColumnIndex', 1) - 1  # 0-indexed
                row_span = block.get('RowSpan', 1)
                column_span = block.get('ColumnSpan', 1)
                
                # Get cell text
                text = ''
                if 'Relationships' in block:
                    for rel in block['Relationships']:
                        if rel['Type'] == 'CHILD':
                            # Look up text blocks with these IDs
                            for child_id in rel['Ids']:
                                child_block = next((b for b in blocks if b.get('Id') == child_id), None)
                                if child_block and 'Text' in child_block:
                                    text += child_block['Text'] + ' '
                text = text.strip()
                
                # Get bounding box if available
                bounding_box = None
                if 'Geometry' in block and 'BoundingBox' in block['Geometry']:
                    bb = block['Geometry']['BoundingBox']
                    bounding_box = BoundingBox(
                        width=bb.get('Width', 0),
                        height=bb.get('Height', 0),
                        left=bb.get('Left', 0),
                        top=bb.get('Top', 0)
                    )
                
                # Create cell block
                cell = CellBlock(
                    id=block_id,
                    row_index=row_index,
                    column_index=column_index,
                    row_span=row_span,
                    column_span=column_span,
                    text=text,
                    confidence=block.get('Confidence', 0),
                    page=page,
                    bounding_box=bounding_box
                )
                
                # Add to table cells
                if parent_id in table_cells:
                    table_cells[parent_id].append(cell)
        
        # Second pass - associate cells with tables and compute table dimensions
        for table in tables:
            if table.id in table_cells:
                cells = table_cells[table.id]
                table.cells = cells
                
                # Compute table dimensions (row and column counts)
                if cells:
                    max_row = max(c.row_index + c.row_span for c in cells)
                    max_col = max(c.column_index + c.column_span for c in cells)
                    table.row_count = max_row
                    table.column_count = max_col
        
        return TextractResult(text_blocks=text_blocks, tables=tables) 