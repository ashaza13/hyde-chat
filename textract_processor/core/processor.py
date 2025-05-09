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
        # Start asynchronous document analysis job
        job_id = self._start_analysis_job(bucket_name, key, extract_tables)
        
        # Wait for job to complete
        response = self._wait_for_job_completion(job_id)
        
        # Process results
        if response['JobStatus'] == 'SUCCEEDED':
            return self._process_analysis_results(job_id)
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
            print(f"Using synchronous AnalyzeDocument API for file under 5MB")
            with open(file_path, 'rb') as file:
                file_bytes = file.read()
            
            return self._analyze_document_bytes(file_bytes, extract_tables)
        
        # For larger files, we need to upload to S3 and use asynchronous API
        elif upload_bucket:
            print(f"Using asynchronous StartDocumentAnalysis API for file over 5MB")
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
    
    def _analyze_document_bytes(
        self, 
        document_bytes: bytes,
        extract_tables: bool = True
    ) -> TextractResult:
        """
        Process document bytes using synchronous Textract AnalyzeDocument API.
        
        Args:
            document_bytes: PDF document as bytes
            extract_tables: Whether to extract tables
            
        Returns:
            TextractResult with extracted text and tables
        """
        # List of all detected blocks
        all_blocks = []
        
        # Define the features to analyze
        feature_types = ['TABLES']
        
        # Always use AnalyzeDocument API since it can handle both text and tables
        print("Calling AnalyzeDocument API...")
        response = self.textract_client.analyze_document(
            Document={'Bytes': document_bytes},
            FeatureTypes=feature_types
        )
        all_blocks.extend(response['Blocks'])
        
        # Handle pagination if needed
        next_token = response.get('NextToken')
        while next_token:
            print(f"Getting next page of results with token: {next_token[:10]}...")
            response = self.textract_client.analyze_document(
                Document={'Bytes': document_bytes},
                FeatureTypes=feature_types,
                NextToken=next_token
            )
            all_blocks.extend(response['Blocks'])
            next_token = response.get('NextToken')
        
        print(f"Total blocks from AnalyzeDocument: {len(all_blocks)}")
        
        # Process blocks into structured result
        return self._parse_blocks(all_blocks)
    
    def _start_analysis_job(
        self, 
        bucket_name: str, 
        key: str,
        extract_tables: bool = True
    ) -> str:
        """
        Start an asynchronous Textract document analysis job.
        
        Args:
            bucket_name: S3 bucket name
            key: S3 object key
            extract_tables: Whether to extract tables
            
        Returns:
            Job ID of the Textract job
        """
        # Define the features to analyze
        feature_types = ['TABLES']
        
        # Always use StartDocumentAnalysis since it can handle both text and tables
        print("Starting document analysis job...")
        response = self.textract_client.start_document_analysis(
            DocumentLocation={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': key
                }
            },
            FeatureTypes=feature_types
        )
        
        return response['JobId']
    
    def _wait_for_job_completion(self, job_id: str) -> Dict[str, Any]:
        """
        Wait for a Textract job to complete.
        
        Args:
            job_id: Job ID of the Textract job
            
        Returns:
            Response from Textract get_document_analysis
        """
        # Get initial job status
        print(f"Checking status of job {job_id}...")
        response = self.textract_client.get_document_analysis(
            JobId=job_id
        )
        
        # Wait for job to complete
        status = response['JobStatus']
        while status == 'IN_PROGRESS':
            print(f"Job status: {status}. Waiting...")
            time.sleep(5)
            
            response = self.textract_client.get_document_analysis(
                JobId=job_id
            )
            status = response['JobStatus']
        
        print(f"Job completed with status: {status}")
        return response
    
    def _process_analysis_results(self, job_id: str) -> TextractResult:
        """
        Process the results from a Textract document analysis job.
        
        Args:
            job_id: Job ID of the Textract job
            
        Returns:
            TextractResult with extracted text and tables
        """
        # List of all detected blocks
        all_blocks = []
        
        # Get analysis results
        print(f"Getting document analysis results for job {job_id}...")
        response = self.textract_client.get_document_analysis(
            JobId=job_id
        )
        all_blocks.extend(response['Blocks'])
        
        # Get all pages
        next_token = response.get('NextToken')
        while next_token:
            print(f"Getting next page of results with token: {next_token[:10]}...")
            response = self.textract_client.get_document_analysis(
                JobId=job_id,
                NextToken=next_token
            )
            all_blocks.extend(response['Blocks'])
            next_token = response.get('NextToken')
        
        print(f"Total blocks from document analysis: {len(all_blocks)}")
        
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
        # Create dictionaries to store blocks by ID for easy lookup
        block_map = {block.get('Id'): block for block in blocks}
        
        # Create a reverse mapping from child blocks to parent blocks
        # This helps us find the table that a cell belongs to, even without ParentId
        child_to_parent = {}
        for block in blocks:
            if 'Relationships' in block:
                for rel in block['Relationships']:
                    if rel['Type'] == 'CHILD':
                        for child_id in rel['Ids']:
                            child_to_parent[child_id] = block['Id']
        
        text_blocks = []
        tables = []
        table_cells = {}  # Map table ID to list of cells
        
        # Process page blocks to get total pages
        page_blocks = [b for b in blocks if b.get('BlockType') == 'PAGE']
        total_pages = len(page_blocks)
        print(f"Document has {total_pages} pages")
        
        # Count different block types for debugging
        block_types = {}
        for block in blocks:
            block_type = block.get('BlockType')
            if block_type in block_types:
                block_types[block_type] += 1
            else:
                block_types[block_type] = 1
        
        print("Block types found:")
        for block_type, count in block_types.items():
            print(f"  {block_type}: {count}")
        
        # First pass - collect text blocks and table blocks
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
                
                # Find parent ID through relationships map if not directly available
                parent_id = None
                if block_id in child_to_parent:
                    parent_id = child_to_parent[block_id]
                
                # Create text block
                text_block = TextBlock(
                    id=block_id,
                    type=BlockType(block_type),
                    text=text,
                    confidence=block.get('Confidence', 0),
                    page=page,
                    bounding_box=bounding_box,
                    parent_id=parent_id
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
        
        # Second pass - find all cells and associate them with tables
        for block in blocks:
            if block.get('BlockType') == 'CELL':
                cell_id = block.get('Id')
                
                # Find the parent table ID - either from ParentId, Relationships or our child_to_parent map
                parent_table_id = None
                
                # Try to find table ID from the child_to_parent map
                if cell_id in child_to_parent:
                    parent_block_id = child_to_parent[cell_id]
                    parent_block = block_map.get(parent_block_id)
                    if parent_block and parent_block.get('BlockType') == 'TABLE':
                        parent_table_id = parent_block_id
                
                # If we still don't have a parent table ID, try to find it by looking at table relationships
                if not parent_table_id:
                    for table_block in [b for b in blocks if b.get('BlockType') == 'TABLE']:
                        if 'Relationships' in table_block:
                            for rel in table_block['Relationships']:
                                if rel['Type'] == 'CHILD' and cell_id in rel['Ids']:
                                    parent_table_id = table_block['Id']
                                    break
                
                # Skip if we can't associate with a table
                if not parent_table_id or parent_table_id not in table_cells:
                    print(f"Warning: Cell {cell_id} has no parent table or parent table not found")
                    continue
                
                # Get page number
                page = 1
                if 'Page' in block:
                    page = block.get('Page', 1)
                
                # Get cell position and span information
                row_index = block.get('RowIndex', 1) - 1  # 0-indexed
                column_index = block.get('ColumnIndex', 1) - 1  # 0-indexed
                row_span = block.get('RowSpan', 1)
                column_span = block.get('ColumnSpan', 1)
                
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
                
                # Extract cell text by gathering all child words/lines
                cell_text = ""
                # First check if there's a direct relationship to child elements
                if 'Relationships' in block:
                    for rel in block['Relationships']:
                        if rel['Type'] == 'CHILD':
                            child_ids = rel['Ids']
                            # Process each child ID
                            for child_id in child_ids:
                                if child_id in block_map:
                                    child_block = block_map[child_id]
                                    if 'Text' in child_block:
                                        cell_text += child_block['Text'] + " "
                
                # If no text found from direct children, try to find text within the cell bounds
                if not cell_text and bounding_box:
                    # Find all WORD blocks that might be inside this cell
                    for word_block in [b for b in blocks if b.get('BlockType') == 'WORD' and 'Geometry' in b]:
                        if 'BoundingBox' not in word_block['Geometry']:
                            continue
                        
                        word_bb = word_block['Geometry']['BoundingBox']
                        
                        # Convert word_bb to an object with properties like bounding_box for easier comparison
                        word_bb_obj = {
                            'left': word_bb.get('Left', 0),
                            'top': word_bb.get('Top', 0),
                            'width': word_bb.get('Width', 0),
                            'height': word_bb.get('Height', 0)
                        }
                        
                        # Check if the word is inside the cell bounds
                        if (word_bb_obj['left'] >= bounding_box.left and 
                            word_bb_obj['top'] >= bounding_box.top and
                            word_bb_obj['left'] + word_bb_obj['width'] <= bounding_box.left + bounding_box.width and
                            word_bb_obj['top'] + word_bb_obj['height'] <= bounding_box.top + bounding_box.height):
                            
                            # Add the word text
                            if 'Text' in word_block:
                                cell_text += word_block['Text'] + " "
                
                # If STILL no text, check if this cell is referenced in a MERGED_CELL
                if not cell_text:
                    for merged_cell_block in [b for b in blocks if b.get('BlockType') == 'MERGED_CELL']:
                        if 'Relationships' in merged_cell_block and merged_cell_block.get('RowIndex') == row_index + 1 and merged_cell_block.get('ColumnIndex') == column_index + 1:
                            for rel in merged_cell_block['Relationships']:
                                if rel['Type'] == 'CHILD':
                                    for child_id in rel['Ids']:
                                        if child_id in block_map:
                                            child_block = block_map[child_id]
                                            if child_block.get('BlockType') == 'WORD' and 'Text' in child_block:
                                                cell_text += child_block['Text'] + " "
                
                # Create cell block
                cell = CellBlock(
                    id=cell_id,
                    row_index=row_index,
                    column_index=column_index,
                    row_span=row_span,
                    column_span=column_span,
                    text=cell_text.strip(),
                    confidence=block.get('Confidence', 0),
                    page=page,
                    bounding_box=bounding_box
                )
                
                # Add to table cells
                table_cells[parent_table_id].append(cell)
        
        # Third pass - associate cells with tables and compute table dimensions
        for table in tables:
            if table.id in table_cells:
                cells = table_cells[table.id]
                print(f"Table {table.id}: Found {len(cells)} cells")
                
                # Only keep the table if it has cells
                if cells:
                    # Find max row and column indices to determine table dimensions
                    max_row_index = max((c.row_index + c.row_span) for c in cells) if cells else 0
                    max_col_index = max((c.column_index + c.column_span) for c in cells) if cells else 0
                    
                    print(f"Table dimensions: {max_row_index} rows x {max_col_index} columns")
                    
                    # Set table dimensions
                    table.row_count = max_row_index
                    table.column_count = max_col_index
                    
                    # Sort cells by row then column for easier processing
                    cells.sort(key=lambda c: (c.row_index, c.column_index))
                    
                    # Add to table
                    table.cells = cells
                    
                    # Debug first few cells
                    for i, cell in enumerate(cells[:5]):
                        print(f"  Cell {i}: Row {cell.row_index}, Col {cell.column_index}, Text: '{cell.text[:20]}...' if len(cell.text) > 20 else cell.text")
        
        # Only keep tables that have cells and dimensions
        original_tables_count = len(tables)
        tables = [t for t in tables if t.cells and t.row_count > 0 and t.column_count > 0]
        print(f"Keeping {len(tables)} out of {original_tables_count} tables that have valid cells and dimensions")
        
        # Dump sample of first table for debugging
        if tables:
            first_table = tables[0]
            print(f"First table: {first_table.row_count} rows x {first_table.column_count} columns")
            print(f"First table has {len(first_table.cells)} cells")
            
            # Create a grid visualization of the first table
            grid = [['' for _ in range(first_table.column_count)] for _ in range(first_table.row_count)]
            for cell in first_table.cells:
                if (cell.row_index < first_table.row_count and 
                    cell.column_index < first_table.column_count):
                    grid[cell.row_index][cell.column_index] = cell.text[:10] + '...' if len(cell.text) > 10 else cell.text
            
            print("Table grid preview:")
            for row in grid:
                print(" | ".join([f"'{cell}'" for cell in row]))
        
        return TextractResult(text_blocks=text_blocks, tables=tables) 