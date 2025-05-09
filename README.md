# Financial Audit QA Packages

This repository contains packages for answering financial audit questions using different approaches:

1. **PDF Memory Audit**: Loads the entire PDF document into memory and uses it as context for each question.
2. **RAG Audit**: Uses Retrieval-Augmented Generation to find relevant parts of the document for each question.
3. **Hyde Audit**: Uses Hypothetical Document Embeddings to improve retrieval of relevant context.

All packages return structured responses with "Yes", "No", or "N/A" answers along with explanations.

## New Feature: AWS Textract Integration with AnalyzeDocument API

The packages now include enhanced document processing capabilities using AWS Textract's AnalyzeDocument API:

- Extracts both text and tables from PDF documents with high accuracy
- Preserves table structure by converting tables to markdown format
- Uses the AnalyzeDocument API specifically to better extract table data
- Integrates with all three approaches (PDF Memory, RAG, and Hyde)
- Falls back to PyPDF2 if Textract processing fails

## Installation

```bash
pip install -r requirements.txt
```

## AWS Textract Processor

The Textract processor can be used directly to extract text and tables from PDF documents using the AnalyzeDocument API:

```python
from textract_processor import TextractProcessor

# Initialize the processor
processor = TextractProcessor(
    aws_region="us-east-1",
    aws_access_key_id="YOUR_ACCESS_KEY",  # Optional if using IAM roles
    aws_secret_access_key="YOUR_SECRET_KEY"  # Optional if using IAM roles
)

# Process a document in S3 (uses StartDocumentAnalysis API for larger files)
result = processor.process_document(
    bucket_name="your-bucket",
    key="path/to/document.pdf",
    extract_tables=True
)

# Process a local document (uses AnalyzeDocument API for small files)
result = processor.process_local_document(
    file_path="path/to/local/document.pdf",
    extract_tables=True
)

# Get text content
text_content = result.get_full_text()

# Get tables
tables = result.tables
for table in tables:
    # Convert table to markdown
    markdown_table = table.to_markdown()
    print(markdown_table)

# Get combined content with tables in markdown format
combined_content = result.get_text_with_tables()
```

You can also process local PDF files:

```python
# For files less than 5MB (uses synchronous AnalyzeDocument API)
result = processor.process_local_document(
    file_path="path/to/local/document.pdf",
    extract_tables=True
)

# For files larger than 5MB (requires S3 upload, uses StartDocumentAnalysis API)
result = processor.process_local_document(
    file_path="path/to/large/document.pdf",
    extract_tables=True,
    upload_bucket="your-temp-bucket"
)
```

## PDF Memory Audit

This approach loads the entire PDF document into memory and uses it as context for answering each question.

```python
from pdf_memory_audit import AuditQA

# Initialize the client
audit_qa = AuditQA(
    aws_region="us-gov-west-1",
    aws_access_key_id="YOUR_ACCESS_KEY",  # Optional if using IAM roles
    aws_secret_access_key="YOUR_SECRET_KEY"  # Optional if using IAM roles
)

# Load a document from S3
audit_qa.load_document(bucket_name="your-bucket", key="path/to/document.pdf")

# Ask a question
response = audit_qa.answer_question("Does the financial statement comply with GASB standards?")

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
print(f"Explanation: {response.explanation}")
```

## RAG Audit

This approach uses Retrieval-Augmented Generation to find the most relevant parts of the document for each question.

```python
from rag_audit import AuditQA

# Initialize the client
audit_qa = AuditQA(
    aws_region="us-gov-west-1",
    aws_access_key_id="YOUR_ACCESS_KEY",  # Optional if using IAM roles
    aws_secret_access_key="YOUR_SECRET_KEY"  # Optional if using IAM roles
)

# Load a document from S3
audit_qa.load_document(bucket_name="your-bucket", key="path/to/document.pdf")

# Ask a question
response = audit_qa.answer_question("Does the financial statement comply with GASB standards?")

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
print(f"Explanation: {response.explanation}")

# Print sources if available
if response.sources:
    print("Sources:")
    for source in response.sources:
        print(f"- {source}")
```

## Key Differences

- **PDF Memory Audit**: Simpler approach but may not handle large documents well due to context length limitations.
- **RAG Audit**: More efficient for large documents as it only retrieves the relevant parts, potentially providing more focused answers.
- **Hyde Audit**: Generates hypothetical answers to improve retrieval of relevant content, often resulting in more accurate context selection.

## Requirements

- Python 3.8+
- AWS account with Bedrock and Textract access
- Required Python packages listed in requirements.txt