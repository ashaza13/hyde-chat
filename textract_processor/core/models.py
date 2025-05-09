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
        # Create empty grid based on row_count and column_count
        grid = [['' for _ in range(self.column_count)] for _ in range(self.row_count)]
        
        # Fill grid with cell values
        for cell in self.cells:
            # Account for spans by filling all spanned cells with the same value
            for r in range(cell.row_span):
                for c in range(cell.column_span):
                    if (cell.row_index + r < self.row_count and 
                        cell.column_index + c < self.column_count):
                        grid[cell.row_index + r][cell.column_index + c] = cell.text
        
        # Convert grid to markdown
        markdown_rows = []
        
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
        
        # Merge text and tables in order of appearance
        while text_pointer < len(text_blocks) or tables:
            # If we've processed all text blocks but still have tables
            if text_pointer >= len(text_blocks):
                for table in tables:
                    result.append(f"\n{table.to_markdown()}\n")
                tables = []
                continue
                
            current_text_block = text_blocks[text_pointer]
            
            # If we have tables on the current page
            if tables and tables[0].page == current_text_block.page:
                table = tables[0]
                # If the table appears before the text block
                if (table.bounding_box and current_text_block.bounding_box and 
                    table.bounding_box.top < current_text_block.bounding_box.top):
                    result.append(f"\n{table.to_markdown()}\n")
                    tables.pop(0)
                else:
                    result.append(current_text_block.text)
                    text_pointer += 1
            else:
                result.append(current_text_block.text)
                text_pointer += 1
        
        return '\n'.join(result) 