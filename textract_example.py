#!/usr/bin/env python3
"""
Example showing how to use the TextractProcessor to extract text and tables from a PDF file.
"""

import os
import argparse
from textract_processor import TextractProcessor

def main():
    """Main function to demonstrate TextractProcessor with local PDF files."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PDF files with AWS Textract')
    parser.add_argument('--file', required=True, help='Path to the local PDF file')
    parser.add_argument('--aws-region', default='us-east-1', help='AWS region to use')
    parser.add_argument('--bucket', help='S3 bucket for large files (>5MB)')
    parser.add_argument('--output', help='Output file (markdown format)')
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
            print(combined_content[:1000] + "..." if len(combined_content) > 1000 else combined_content)
            
            if tables:
                print("\nSample of first table as markdown:")
                print("--------------------------------")
                print(tables[0].to_markdown())
                
        print("\nDone!")
        
    except Exception as e:
        print(f"Error processing document: {e}")

if __name__ == "__main__":
    main() 