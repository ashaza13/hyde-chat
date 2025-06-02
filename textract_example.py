#!/usr/bin/env python3
"""
Example showing how to use the TextractProcessor to extract text and tables from a PDF file.
"""

import os
import argparse
import json
import boto3
from textract_processor import TextractProcessor

def main():
    """Main function to demonstrate TextractProcessor with local PDF files."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PDF files with AWS Textract')
    parser.add_argument('--file', required=True, help='Path to the local PDF file')
    parser.add_argument('--aws-region', default='us-east-1', help='AWS region to use')
    parser.add_argument('--bucket', help='S3 bucket for large files (>5MB)')
    parser.add_argument('--output', help='Output file (markdown format)')
    parser.add_argument('--debug', action='store_true', help='Show detailed debug information')
    parser.add_argument('--no-tables', action='store_true', help='Skip table extraction')
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        return
    
    # Initialize processor
    print(f"Initializing TextractProcessor with region {args.aws_region}")
    processor = TextractProcessor(aws_region=args.aws_region)
    
    # Process the document
    try:
        print(f"Processing file {args.file}")
        
        # Determine if we need to upload to S3 for processing
        file_size = os.path.getsize(args.file)
        if file_size >= 5 * 1024 * 1024 and not args.bucket:
            print("Warning: File is larger than 5MB and no S3 bucket provided.")
            print("AWS Textract requires files larger than 5MB to be processed via S3.")
            print("Please provide an S3 bucket using the --bucket argument.")
            return
        
        # Process document
        if file_size >= 5 * 1024 * 1024:
            print(f"File is {file_size/1024/1024:.2f}MB, uploading to S3 bucket {args.bucket}")
            result = processor.process_local_document(
                file_path=args.file,
                extract_tables=not args.no_tables,
                upload_bucket=args.bucket
            )
        else:
            print(f"File is {file_size/1024/1024:.2f}MB, processing locally")
            result = processor.process_local_document(
                file_path=args.file,
                extract_tables=not args.no_tables
            )
        
        # Get text content
        text_content = result.get_full_text()
        print(f"Extracted {len(text_content)} characters of text")
        
        # Get tables
        tables = result.tables
        print(f"Extracted {len(tables)} tables")
        
        # Debug table information
        if args.debug and tables:
            print("\nTable Details:")
            for i, table in enumerate(tables):
                print(f"\nTable {i+1}:")
                print(f"  Page: {table.page}")
                print(f"  Rows: {table.row_count}")
                print(f"  Columns: {table.column_count}")
                print(f"  Cells: {len(table.cells)}")
                
                # Print cell details for first few cells
                if table.cells:
                    print("\n  Sample Cells:")
                    for j, cell in enumerate(table.cells[:5]):  # Show first 5 cells
                        print(f"    Cell {j+1}: Row {cell.row_index}, Col {cell.column_index}, Text: '{cell.text}'")
                    
                    if len(table.cells) > 5:
                        print(f"    ... and {len(table.cells) - 5} more cells")
                        
                # Show table as markdown
                print("\n  Markdown Representation:")
                markdown = table.to_markdown()
                print(f"    {markdown.replace(chr(10), chr(10)+'    ')}")
        
        # Get combined content with tables
        combined_content = result.get_text_with_tables()
        print(f"Combined content is {len(combined_content)} characters")
        
        # Write output to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(combined_content)
            print(f"Saved output to {args.output}")
        else:
            # Print a sample of the output
            print("\nSample of extracted content:")
            print("----------------------------")
            sample_length = 1000
            if len(combined_content) > sample_length:
                print(combined_content[:sample_length] + "...")
            else:
                print(combined_content)
            
            if tables:
                print("\nSample of first table as markdown:")
                print("--------------------------------")
                print(tables[0].to_markdown())
                
        print("\nDone!")
        
        # Example 1: Process a local PDF file
        local_file_path = "fy_2024.pdf"
        
        if os.path.exists(local_file_path):
            print(f"Processing local file: {local_file_path}")
            
            try:
                # Process the document
                result = processor.process_local_document(
                    file_path=local_file_path,
                    extract_tables=True
                )
                
                print(f"Successfully processed document!")
                print(f"Found {len(result.text_blocks)} text blocks")
                print(f"Found {len(result.tables)} tables")
                
                # Get pages represented in the document
                pages = set()
                for block in result.text_blocks:
                    pages.add(block.page)
                for table in result.tables:
                    pages.add(table.page)
                print(f"Document spans {len(pages)} pages: {sorted(pages)}")
                
                # Example 2: Get text with metadata preserved
                print("\n" + "="*50)
                print("EXAMPLE: Text with Metadata")
                print("="*50)
                
                text_with_metadata = result.get_text_with_metadata()
                print(f"Found {len(text_with_metadata)} text/table items with metadata")
                
                # Show first few items with their page numbers
                for i, item in enumerate(text_with_metadata[:5]):
                    print(f"\nItem {i+1} (Page {item['page']}, Type: {item['type']}):")
                    print(f"Text preview: {item['text'][:100]}...")
                
                # Example 3: Get chunked text with metadata
                print("\n" + "="*50)
                print("EXAMPLE: Chunked Text with Page Metadata")
                print("="*50)
                
                chunks = result.get_chunked_text_with_metadata(chunk_size=800, overlap=100)
                print(f"Created {len(chunks)} chunks with metadata")
                
                # Show details for each chunk
                for i, chunk in enumerate(chunks):
                    print(f"\nChunk {i+1}:")
                    print(f"  Page Range: {chunk.get_page_range_str()}")
                    print(f"  Pages: {chunk.page_numbers}")
                    print(f"  Text Length: {len(chunk.text)} characters")
                    print(f"  Source Blocks: {len(chunk.source_blocks) if chunk.source_blocks else 0}")
                    print(f"  Text Preview: {chunk.text[:150]}...")
                    
                    # This is the key information for LLM rationale!
                    print(f"  📍 For LLM Response: This information comes from {chunk.get_page_range_str()}")
                
                # Example 4: Traditional text extraction (for comparison)
                print("\n" + "="*50)
                print("COMPARISON: Traditional Text Extraction")
                print("="*50)
                
                # Get traditional text (without metadata)
                text_content = result.get_full_text()
                print(f"Traditional text length: {len(text_content)} characters")
                print(f"Text preview: {text_content[:200]}...")
                
                # Get text with tables
                combined_content = result.get_text_with_tables()
                print(f"Text with tables length: {len(combined_content)} characters")
                
                # Example 5: Demonstrate how this would work in a RAG system
                print("\n" + "="*50)
                print("EXAMPLE: RAG System Integration")
                print("="*50)
                
                # Simulate a query and show how page numbers would be preserved
                sample_query = "financial performance"
                print(f"Sample Query: '{sample_query}'")
                
                # Find relevant chunks (simple keyword matching for demo)
                relevant_chunks = []
                for chunk in chunks:
                    if sample_query.lower() in chunk.text.lower():
                        relevant_chunks.append(chunk)
                
                print(f"Found {len(relevant_chunks)} relevant chunks")
                
                for i, chunk in enumerate(relevant_chunks[:3]):  # Show top 3
                    print(f"\nRelevant Chunk {i+1}:")
                    print(f"  📍 Source: {chunk.get_page_range_str()}")
                    print(f"  Content: {chunk.text[:200]}...")
                    print(f"  🤖 LLM can now cite: 'Based on information from {chunk.get_page_range_str()}...'")
                
                # Save results to files for inspection
                print("\n" + "="*50)
                print("SAVING RESULTS")
                print("="*50)
                
                # Save traditional text
                with open("output_traditional.txt", "w", encoding="utf-8") as f:
                    f.write(result.get_text_with_tables())
                print("Saved traditional text to: output_traditional.txt")
                
                # Save chunked text with metadata
                with open("output_chunks_with_metadata.txt", "w", encoding="utf-8") as f:
                    f.write("CHUNKED TEXT WITH PAGE METADATA\n")
                    f.write("="*50 + "\n\n")
                    
                    for i, chunk in enumerate(chunks):
                        f.write(f"CHUNK {i+1}\n")
                        f.write(f"Page Range: {chunk.get_page_range_str()}\n")
                        f.write(f"Pages: {chunk.page_numbers}\n")
                        f.write(f"Length: {len(chunk.text)} characters\n")
                        f.write("-" * 30 + "\n")
                        f.write(chunk.text)
                        f.write("\n\n" + "="*50 + "\n\n")
                
                print("Saved chunked text with metadata to: output_chunks_with_metadata.txt")
                
            except Exception as e:
                print(f"Error processing document: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"File {local_file_path} not found. Please ensure the file exists.")
            
            # Example with S3 (commented out - uncomment and modify as needed)
            """
            # Example 2: Process a document from S3
            bucket_name = "your-bucket-name"
            key = "path/to/your/document.pdf"
            
            print(f"Processing S3 document: s3://{bucket_name}/{key}")
            
            try:
                result = processor.process_document(
                    bucket_name=bucket_name,
                    key=key,
                    extract_tables=True
                )
                
                print(f"Successfully processed S3 document!")
                print(f"Found {len(result.text_blocks)} text blocks")
                print(f"Found {len(result.tables)} tables")
                
                # Use the new metadata-aware methods
                chunks = result.get_chunked_text_with_metadata()
                for chunk in chunks:
                    print(f"Chunk from {chunk.get_page_range_str()}: {chunk.text[:100]}...")
                    
            except Exception as e:
                print(f"Error processing S3 document: {e}")
            """
        
    except Exception as e:
        import traceback
        print(f"Error processing document: {e}")
        if args.debug:
            traceback.print_exc()

if __name__ == "__main__":
    main() 