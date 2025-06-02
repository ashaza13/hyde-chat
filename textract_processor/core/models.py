from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

class BlockType(str, Enum):
    """Types of blocks returned by AWS Textract"""
    PAGE = "PAGE"
    LINE = "LINE"
    WORD = "WORD"
    TABLE = "TABLE"
    CELL = "CELL"
    KEY_VALUE_SET = "KEY_VALUE_SET"
    SELECTION_ELEMENT = "SELECTION_ELEMENT"

@dataclass
class BoundingBox:
    """Bounding box for a block element"""
    width: float
    height: float
    left: float
    top: float

@dataclass
class TextBlock:
    """Represents a text block from Textract (line or word)"""
    id: str
    type: BlockType
    text: str
    confidence: float
    page: int
    bounding_box: Optional[BoundingBox] = None
    parent_id: Optional[str] = None

@dataclass
class CellBlock:
    """Represents a cell in a table"""
    id: str
    row_index: int
    column_index: int
    row_span: int
    column_span: int
    text: str
    confidence: float
    page: int
    bounding_box: Optional[BoundingBox] = None

@dataclass
class TableBlock:
    """Represents a table extracted from Textract"""
    id: str
    page: int
    cells: List[CellBlock]
    row_count: int
    column_count: int
    bounding_box: Optional[BoundingBox] = None
    
    def to_markdown(self) -> str:
        """
        Convert table to markdown format
        
        Returns:
            String representation of the table in markdown format
        """
        # Check if we have any cells
        if not self.cells or self.row_count <= 0 or self.column_count <= 0:
            return ""  # Return empty string if no valid cells
            
        # Create empty grid based on row_count and column_count
        grid = [['' for _ in range(self.column_count)] for _ in range(self.row_count)]
        
        # Fill grid with cell values
        for cell in self.cells:
            # Skip cells with invalid indices
            if (cell.row_index < 0 or cell.row_index >= self.row_count or 
                cell.column_index < 0 or cell.column_index >= self.column_count):
                continue
                
            # Account for spans by filling all spanned cells with the same value
            for r in range(cell.row_span):
                for c in range(cell.column_span):
                    if (cell.row_index + r < self.row_count and 
                        cell.column_index + c < self.column_count):
                        grid[cell.row_index + r][cell.column_index + c] = cell.text.strip()
        
        # Convert grid to markdown
        markdown_rows = []
        
        # Check if grid is empty
        if not grid or len(grid) == 0:
            return ""
            
        # Check if any of the rows has content
        has_content = any(any(cell.strip() for cell in row) for row in grid)
        if not has_content:
            return ""
            
        # Header row
        header = '| ' + ' | '.join(grid[0]) + ' |'
        markdown_rows.append(header)
        
        # Separator row
        separator = '| ' + ' | '.join(['---' for _ in range(self.column_count)]) + ' |'
        markdown_rows.append(separator)
        
        # Data rows
        for row in grid[1:]:
            data_row = '| ' + ' | '.join(row) + ' |'
            markdown_rows.append(data_row)
        
        return '\n'.join(markdown_rows)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    page_numbers: List[int]
    start_page: int
    end_page: int
    chunk_index: int
    source_blocks: List[str] = None  # Optional: IDs of source blocks
    
    def get_page_range_str(self) -> str:
        """Get a string representation of the page range"""
        if self.start_page == self.end_page:
            return f"Page {self.start_page}"
        else:
            return f"Pages {self.start_page}-{self.end_page}"

@dataclass
class TextractResult:
    """Complete result from Textract processing"""
    text_blocks: List[TextBlock]
    tables: List[TableBlock]
    
    def get_full_text(self) -> str:
        """
        Get all text content excluding tables
        
        Returns:
            String containing all text content
        """
        # Sort text blocks by page and position
        sorted_blocks = sorted(
            self.text_blocks, 
            key=lambda b: (b.page, b.bounding_box.top if b.bounding_box else 0)
        )
        
        # Join text with newlines
        return '\n'.join(block.text for block in sorted_blocks if block.type == BlockType.LINE)
    
    def get_text_with_tables(self) -> str:
        """
        Get text content with tables converted to markdown
        
        Returns:
            String containing all text with tables in markdown format
        """
        result = []
        current_page = 1
        text_pointer = 0
        
        # Sort text blocks by page and position
        text_blocks = sorted(
            [b for b in self.text_blocks if b.type == BlockType.LINE],
            key=lambda b: (b.page, b.bounding_box.top if b.bounding_box else 0)
        )
        
        # Sort tables by page and position
        tables = sorted(
            self.tables,
            key=lambda t: (t.page, t.bounding_box.top if t.bounding_box else 0)
        )
        
        # Filter out empty tables
        tables = [t for t in tables if t.cells and t.row_count > 0 and t.column_count > 0]
        
        # Debug info
        if not tables:
            result.append("Note: No tables were detected or all tables were empty.")
        
        # Create a set to track text blocks that should be excluded (those within table bounds)
        excluded_text_blocks = set()
        
        # Mark text blocks that overlap with tables as excluded
        for table in tables:
            if table.bounding_box:
                for i, text_block in enumerate(text_blocks):
                    if (text_block.bounding_box and 
                        text_block.page == table.page and
                        self._is_block_within_table(text_block, table)):
                        excluded_text_blocks.add(i)
        
        # Merge text and tables in order of appearance
        while text_pointer < len(text_blocks) or tables:
            # If we've processed all text blocks but still have tables
            if text_pointer >= len(text_blocks):
                for table in tables:
                    markdown_table = table.to_markdown()
                    if markdown_table:
                        result.append(f"\n{markdown_table}\n")
                tables = []
                continue
                
            current_text_block = text_blocks[text_pointer]
            
            # Skip this text block if it's been marked as part of a table
            if text_pointer in excluded_text_blocks:
                text_pointer += 1
                continue
            
            # If we have tables on the current page
            if tables and tables[0].page == current_text_block.page:
                table = tables[0]
                # If the table appears before the text block
                if (table.bounding_box and current_text_block.bounding_box and 
                    table.bounding_box.top < current_text_block.bounding_box.top):
                    markdown_table = table.to_markdown()
                    if markdown_table:
                        result.append(f"\n{markdown_table}\n")
                    tables.pop(0)
                else:
                    result.append(current_text_block.text)
                    text_pointer += 1
            else:
                result.append(current_text_block.text)
                text_pointer += 1
        
        return '\n'.join(result)
    
    def get_text_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Get text content with page number metadata preserved
        
        Returns:
            List of dictionaries with 'text' and 'page' keys
        """
        result = []
        
        # Sort text blocks by page and position
        text_blocks = sorted(
            [b for b in self.text_blocks if b.type == BlockType.LINE],
            key=lambda b: (b.page, b.bounding_box.top if b.bounding_box else 0)
        )
        
        # Sort tables by page and position
        tables = sorted(
            self.tables,
            key=lambda t: (t.page, t.bounding_box.top if t.bounding_box else 0)
        )
        
        # Filter out empty tables
        tables = [t for t in tables if t.cells and t.row_count > 0 and t.column_count > 0]
        
        # Create a set to track text blocks that should be excluded (those within table bounds)
        excluded_text_blocks = set()
        
        # Mark text blocks that overlap with tables as excluded
        for table in tables:
            if table.bounding_box:
                for i, text_block in enumerate(text_blocks):
                    if (text_block.bounding_box and 
                        text_block.page == table.page and
                        self._is_block_within_table(text_block, table)):
                        excluded_text_blocks.add(i)
        
        text_pointer = 0
        
        # Merge text and tables in order of appearance
        while text_pointer < len(text_blocks) or tables:
            # If we've processed all text blocks but still have tables
            if text_pointer >= len(text_blocks):
                for table in tables:
                    markdown_table = table.to_markdown()
                    if markdown_table:
                        result.append({
                            'text': markdown_table,
                            'page': table.page,
                            'type': 'table',
                            'block_id': table.id
                        })
                tables = []
                continue
                
            current_text_block = text_blocks[text_pointer]
            
            # Skip this text block if it's been marked as part of a table
            if text_pointer in excluded_text_blocks:
                text_pointer += 1
                continue
            
            # If we have tables on the current page
            if tables and tables[0].page == current_text_block.page:
                table = tables[0]
                # If the table appears before the text block
                if (table.bounding_box and current_text_block.bounding_box and 
                    table.bounding_box.top < current_text_block.bounding_box.top):
                    markdown_table = table.to_markdown()
                    if markdown_table:
                        result.append({
                            'text': markdown_table,
                            'page': table.page,
                            'type': 'table',
                            'block_id': table.id
                        })
                    tables.pop(0)
                else:
                    result.append({
                        'text': current_text_block.text,
                        'page': current_text_block.page,
                        'type': 'text',
                        'block_id': current_text_block.id
                    })
                    text_pointer += 1
            else:
                result.append({
                    'text': current_text_block.text,
                    'page': current_text_block.page,
                    'type': 'text',
                    'block_id': current_text_block.id
                })
                text_pointer += 1
        
        return result
    
    def get_chunked_text_with_metadata(self, chunk_size: int = 1000, overlap: int = 200) -> List[TextChunk]:
        """
        Get text content chunked with page number metadata preserved
        
        Args:
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of TextChunk objects with page metadata
        """
        # Get text with metadata
        text_with_metadata = self.get_text_with_metadata()
        
        if not text_with_metadata:
            return []
        
        chunks = []
        current_chunk_text = ""
        current_chunk_pages = set()
        current_chunk_blocks = []
        chunk_index = 0
        
        for item in text_with_metadata:
            text = item['text']
            page = item['page']
            block_id = item['block_id']
            
            # If adding this text would exceed chunk size, finalize current chunk
            if current_chunk_text and len(current_chunk_text) + len(text) + 1 > chunk_size:
                # Create chunk from current content
                if current_chunk_text.strip():
                    page_list = sorted(list(current_chunk_pages))
                    chunks.append(TextChunk(
                        text=current_chunk_text.strip(),
                        page_numbers=page_list,
                        start_page=min(page_list) if page_list else 1,
                        end_page=max(page_list) if page_list else 1,
                        chunk_index=chunk_index,
                        source_blocks=current_chunk_blocks.copy()
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk_text) > overlap:
                    # Keep the last 'overlap' characters
                    overlap_text = current_chunk_text[-overlap:]
                    current_chunk_text = overlap_text + "\n" + text
                else:
                    current_chunk_text = text
                
                current_chunk_pages = {page}
                current_chunk_blocks = [block_id]
            else:
                # Add to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n" + text
                else:
                    current_chunk_text = text
                current_chunk_pages.add(page)
                current_chunk_blocks.append(block_id)
        
        # Add final chunk if there's remaining content
        if current_chunk_text.strip():
            page_list = sorted(list(current_chunk_pages))
            chunks.append(TextChunk(
                text=current_chunk_text.strip(),
                page_numbers=page_list,
                start_page=min(page_list) if page_list else 1,
                end_page=max(page_list) if page_list else 1,
                chunk_index=chunk_index,
                source_blocks=current_chunk_blocks.copy()
            ))
        
        return chunks

    def _is_block_within_table(self, text_block: TextBlock, table: TableBlock) -> bool:
        """
        Check if a text block is within the bounds of a table
        
        Args:
            text_block: The text block to check
            table: The table to check against
            
        Returns:
            True if the text block is within the bounds of the table
        """
        if not text_block.bounding_box or not table.bounding_box:
            return False
            
        tb_box = text_block.bounding_box
        table_box = table.bounding_box
        
        # Check if the text block is completely within the table bounds
        return (tb_box.left >= table_box.left and
                tb_box.top >= table_box.top and
                tb_box.left + tb_box.width <= table_box.left + table_box.width and
                tb_box.top + tb_box.height <= table_box.top + table_box.height) 