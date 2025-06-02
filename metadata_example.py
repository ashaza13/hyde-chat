#!/usr/bin/env python3
"""
Example showing how to use the enhanced TextractProcessor to extract text with page number metadata.
This demonstrates the new functionality for preserving page numbers during text extraction and chunking.
"""

import os
from textract_processor import TextractProcessor

def main():
    """Main function to demonstrate metadata-aware text extraction."""
    
    # Initialize the processor
    processor = TextractProcessor(aws_region="us-gov-west-1")
    
    # Process a local PDF file
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
            
            # Example 1: Get text with metadata preserved
            print("\n" + "="*50)
            print("EXAMPLE: Text with Metadata")
            print("="*50)
            
            text_with_metadata = result.get_text_with_metadata()
            print(f"Found {len(text_with_metadata)} text/table items with metadata")
            
            # Show first few items with their page numbers
            for i, item in enumerate(text_with_metadata[:5]):
                print(f"\nItem {i+1} (Page {item['page']}, Type: {item['type']}):")
                print(f"Text preview: {item['text'][:100]}...")
            
            # Example 2: Get chunked text with metadata
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
            
            # Example 3: Traditional text extraction (for comparison)
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
            
            # Example 4: Demonstrate how this would work in a RAG system
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
            
            # Example 5: Save results to files for inspection
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
            
            # Example 6: Show how to integrate with existing RAG systems
            print("\n" + "="*50)
            print("INTEGRATION EXAMPLE: Enhanced Document Processor")
            print("="*50)
            
            print("Here's how you can modify your existing document processors:")
            print("""
# Instead of:
# chunks = self._chunk_text(content)

# Use:
chunks_with_metadata = self.textract_result.get_chunked_text_with_metadata(
    chunk_size=1000, 
    overlap=200
)

# Now each chunk has page information:
for chunk in chunks_with_metadata:
    # Store both text and metadata
    document_text = chunk.text
    page_info = chunk.get_page_range_str()
    
    # When LLM responds, include page reference:
    # "Based on information from {page_info}, the financial performance..."
            """)
            
        except Exception as e:
            print(f"Error processing document: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"File {local_file_path} not found. Please ensure the file exists.")
        print("You can also modify this script to use S3 documents:")
        print("""
# Example with S3:
result = processor.process_document(
    bucket_name="your-bucket-name",
    key="path/to/your/document.pdf",
    extract_tables=True
)

# Then use the new metadata methods:
chunks = result.get_chunked_text_with_metadata()
        """)

if __name__ == "__main__":
    main() 