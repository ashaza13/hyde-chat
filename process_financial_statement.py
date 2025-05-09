#!/usr/bin/env python3
"""
Example script to process a financial statement PDF with AWS Textract AnalyzeDocument API.
This script works with financial statement PDFs containing tables such as balance sheets,
income statements, and cash flow statements.
"""

import os
import argparse
import json
from textract_processor import TextractProcessor

def main():
    """Process the financial statement PDF and extract tables using AWS Textract AnalyzeDocument API."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process financial statement PDFs with AWS Textract')
    parser.add_argument('--file', default="fy_2024.pdf", help='Path to the financial statement PDF')
    parser.add_argument('--region', default="us-east-1", help='AWS region')
    parser.add_argument('--bucket', help='S3 bucket for large files (>5MB)')
    parser.add_argument('--output-dir', default="textract_output", help='Output directory for results')
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        return
    
    print(f"Processing financial statement: {args.file}")
    
    # Initialize the Textract processor with specified region
    processor = TextractProcessor(aws_region=args.region)
    
    try:
        # Determine if S3 is needed based on file size
        file_size = os.path.getsize(args.file)
        use_s3 = file_size >= 5 * 1024 * 1024
        
        if use_s3 and not args.bucket:
            print(f"Warning: File is {file_size/1024/1024:.2f}MB (over 5MB)")
            print("For files larger than 5MB, AWS Textract requires S3 bucket access.")
            print("Please provide a bucket with --bucket argument.")
            return
        
        # Process the document using the appropriate method
        print(f"File size: {file_size/1024/1024:.2f}MB")
        if use_s3:
            print(f"Using asynchronous StartDocumentAnalysis API with S3 bucket: {args.bucket}")
            # Upload and process via S3
            result = processor.process_local_document(
                file_path=args.file,
                extract_tables=True,
                upload_bucket=args.bucket
            )
        else:
            print("Using synchronous AnalyzeDocument API")
            # Process locally
            result = processor.process_local_document(
                file_path=args.file,
                extract_tables=True
            )
        
        # Print information about extracted content
        print(f"\nExtraction Results:")
        print(f"-------------------")
        print(f"Text Blocks: {len(result.text_blocks)}")
        print(f"Lines: {len([b for b in result.text_blocks if b.type == 'LINE'])}")
        print(f"Words: {len([b for b in result.text_blocks if b.type == 'WORD'])}")
        print(f"Tables: {len(result.tables)}")
        
        # Create output directory
        output_dir = args.output_dir
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
            
            print(f"\nTable Details:")
            print(f"-------------")
            
            for i, table in enumerate(result.tables):
                print(f"Table {i+1}:")
                print(f"  Page: {table.page}")
                print(f"  Dimensions: {table.row_count} rows × {table.column_count} columns")
                print(f"  Total cells: {len(table.cells)}")
                
                # Check if cells have content
                cells_with_text = sum(1 for cell in table.cells if cell.text.strip())
                print(f"  Cells with text: {cells_with_text} ({cells_with_text/len(table.cells)*100:.1f}%)")
                
                # Save table to file
                table_file = os.path.join(tables_dir, f"table_{i+1}.md")
                with open(table_file, "w", encoding="utf-8") as f:
                    # Write table metadata
                    f.write(f"# Table {i+1}\n\n")
                    f.write(f"- Page: {table.page}\n")
                    f.write(f"- Rows: {table.row_count}\n")
                    f.write(f"- Columns: {table.column_count}\n")
                    f.write(f"- Cells: {len(table.cells)}\n")
                    f.write(f"- Cells with text: {cells_with_text}\n\n")
                    
                    # Write table as markdown
                    f.write(table.to_markdown())
                    
                    # Also write raw cell data
                    f.write("\n\n## Raw Cell Data\n\n")
                    for j, cell in enumerate(table.cells):
                        f.write(f"- Cell {j+1}: Row {cell.row_index}, Col {cell.column_index}, Text: '{cell.text}'\n")
                
                # Also save CSV version
                csv_file = os.path.join(tables_dir, f"table_{i+1}.csv")
                with open(csv_file, "w", encoding="utf-8") as f:
                    # Create a grid of cells
                    grid = [['' for _ in range(table.column_count)] for _ in range(table.row_count)]
                    
                    # Fill the grid with cell values
                    for cell in table.cells:
                        if cell.row_index < table.row_count and cell.column_index < table.column_count:
                            grid[cell.row_index][cell.column_index] = cell.text.replace('"', '""')
                    
                    # Write as CSV
                    for row in grid:
                        f.write(','.join([f'"{cell}"' for cell in row]) + '\n')
                
                print(f"  Saved table {i+1} to {table_file} and {csv_file}")
        
        # Print the first table as an example
        if result.tables:
            table = result.tables[0]
            print(f"\nExample of first extracted table ({table.row_count}x{table.column_count}):")
            print("-----------------------------")
            markdown = table.to_markdown()
            print(markdown)
            
            # Print sample of raw cell data
            print("\nSample of cell data:")
            for i, cell in enumerate(table.cells[:5]):  # Print first 5 cells
                print(f"  Cell {i+1}: Row {cell.row_index}, Col {cell.column_index}, Text: '{cell.text}'")
            if len(table.cells) > 5:
                print(f"  ... and {len(table.cells) - 5} more cells")
        else:
            print("\nNo tables were extracted from the document.")
            
    except Exception as e:
        import traceback
        print(f"Error processing document: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 