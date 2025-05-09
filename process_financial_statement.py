#!/usr/bin/env python3
"""
Example script to process a financial statement PDF with Textract and extract tables.
This script works with the sample financial statement PDF showing tables with financial data.
"""

import os
from textract_processor import TextractProcessor

def main():
    """Process the financial statement PDF and extract tables."""
    
    # Path to the financial statement PDF
    pdf_file = "fy_2024.pdf"
    
    # Check if the file exists
    if not os.path.exists(pdf_file):
        print(f"Error: File {pdf_file} does not exist")
        return
    
    print(f"Processing financial statement: {pdf_file}")
    
    # Initialize the Textract processor
    processor = TextractProcessor()
    
    try:
        # Process the document
        result = processor.process_local_document(
            file_path=pdf_file,
            extract_tables=True
        )
        
        # Print information about extracted content
        print(f"Extracted {len(result.text_blocks)} text blocks")
        print(f"Extracted {len(result.tables)} tables")
        
        # Output directory for results
        output_dir = "textract_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the complete text with tables as markdown
        combined_output = os.path.join(output_dir, "financial_statement_with_tables.md")
        with open(combined_output, "w", encoding="utf-8") as f:
            f.write(result.get_text_with_tables())
        print(f"Saved combined output to {combined_output}")
        
        # Save each table separately
        if result.tables:
            tables_dir = os.path.join(output_dir, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            
            for i, table in enumerate(result.tables):
                table_file = os.path.join(tables_dir, f"table_{i+1}.md")
                with open(table_file, "w", encoding="utf-8") as f:
                    # Write table metadata
                    f.write(f"# Table {i+1}\n\n")
                    f.write(f"- Page: {table.page}\n")
                    f.write(f"- Rows: {table.row_count}\n")
                    f.write(f"- Columns: {table.column_count}\n")
                    f.write(f"- Cells: {len(table.cells)}\n\n")
                    
                    # Write table as markdown
                    f.write(table.to_markdown())
                    
                    # Also write raw cell data
                    f.write("\n\n## Raw Cell Data\n\n")
                    for j, cell in enumerate(table.cells):
                        f.write(f"- Cell {j+1}: Row {cell.row_index}, Col {cell.column_index}, Text: '{cell.text}'\n")
                        
                print(f"Saved table {i+1} to {table_file}")
        
        # Print the first table as an example
        if result.tables:
            print("\nExample of first extracted table:")
            print("-----------------------------")
            print(result.tables[0].to_markdown())
        else:
            print("\nNo tables were extracted from the document.")
            
    except Exception as e:
        import traceback
        print(f"Error processing document: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 