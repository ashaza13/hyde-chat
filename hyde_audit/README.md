# HyDE Audit

A Python package that uses Hypothetical Document Embeddings (HyDE) to answer questions on higher education financial statement audits.

## Features

- Uses AWS Bedrock for LLM integration
- Provides clear "Yes", "No", or "N/A" answers to audit questions
- Enforces response format using Pydantic models
- Easily integrates with existing audit workflows

## Installation

```bash
pip install hyde-audit
```

## Usage

```python
from hyde_audit import AuditQA

# Initialize the service
audit_qa = AuditQA(
    aws_region="us-east-1",
    aws_access_key_id="your-access-key",  # Optional if using IAM roles
    aws_secret_access_key="your-secret-key"  # Optional if using IAM roles
)

# Load your financial statement
audit_qa.load_document("path/to/financial_statement.pdf")

# Ask audit questions
result = audit_qa.answer_question("Does the statement comply with GASB standards?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Explanation: {result.explanation}")
```

## License

MIT 