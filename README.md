# Financial Audit QA Packages

This repository contains two packages for answering financial audit questions:

1. **PDF Memory Audit**: Loads the entire PDF document into memory and uses it as context for each question.
2. **RAG Audit**: Uses Retrieval-Augmented Generation to find relevant parts of the document for each question.

Both packages return structured responses with "Yes", "No", or "N/A" answers along with explanations.

## Installation

```bash
pip install -r requirements.txt
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

## Requirements

- Python 3.8+
- AWS account with Bedrock access
- Required Python packages listed in requirements.txt