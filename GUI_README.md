# Audit Question Processing GUI

A Streamlit-based graphical user interface for processing audit questions using RAG, Memory, and HYDE approaches.

## Features

### 🔐 AWS Credentials Management
- Input fields for AWS Access Key ID, Secret Access Key, and Session Token
- Support for temporary credentials via session tokens
- Configurable AWS region selection

### 🤖 Model Configuration
- Adjustable model parameters:
  - **Temperature**: Controls randomness in responses (0.0 - 1.0)
  - **Top P**: Controls diversity of responses (0.0 - 1.0)
  - **Max Tokens**: Maximum number of tokens to generate (100 - 4000)
- Support for multiple Bedrock models:
  - Claude 3 Sonnet, Haiku, and Opus
  - Amazon Titan Text Express and Lite

### 📋 Question Management
- Upload CSV files with audit questions
- Automatic loading of `sample_questions.csv` if available
- Preview of loaded questions
- Individual question selection for processing

### 🚀 Processing Options
- **Single Question Processing**: Process individual questions with selected approaches
- **Batch Processing**: Process all questions at once
- **Approach Selection**: Choose from RAG, Memory, and HYDE approaches
- **RAG Query Rewriting**: Optional query optimization for better vector search

### 📊 Results Display
- Real-time results for single question processing
- Detailed answers, confidence scores, and explanations
- Batch processing status monitoring
- Download results as CSV files

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_streamlit.txt
```

2. Ensure all audit processing modules are available in your Python path.

## Usage

### Method 1: Using the run script
```bash
python run_gui.py
```

### Method 2: Direct Streamlit command
```bash
streamlit run streamlit_app.py
```

The GUI will be available at `http://localhost:8501`

## Configuration Steps

1. **AWS Credentials**: Enter your AWS credentials in the sidebar
   - AWS Access Key ID (required)
   - AWS Secret Access Key (required)
   - AWS Session Token (optional, for temporary credentials)
   - AWS Region (default: us-gov-west-1)

2. **Model Configuration**: Adjust model parameters as needed
   - Select the desired Bedrock model
   - Tune temperature, top_p, and max_tokens
   - Enable RAG query rewriting if desired

3. **Document Setup**: Specify the S3 location of your document
   - S3 Bucket name
   - S3 Key (file path)
   - Force reprocess option

4. **Questions**: Upload or load questions
   - Upload a CSV file with questions
   - Or use the "Load Sample Questions" button if `sample_questions.csv` exists

5. **Initialize**: Click "Initialize Processor" to set up the system

## Processing Workflows

### Single Question Processing
1. Select a question from the dropdown
2. Choose which approaches to use (RAG, Memory, HYDE)
3. Click "Process Question"
4. View results in expandable sections

### Batch Processing
1. Select approaches for batch processing
2. Click "Start Batch Processing"
3. Monitor progress and status
4. Download results when complete

## CSV Format

The questions CSV should have the following columns:
- `id`: Unique identifier for the question
- `question`: The audit question text
- `truth`: Expected answer (optional)
- Additional columns for storing results

Example:
```csv
id,question,truth
1,Does the financial statement comply with all relevant GASB standards?,Yes
1a,Does the financial statement comply with GASB Statement No. 34?,Yes
2,Are there any significant deficiencies in internal controls?,No
```

## Features in Detail

### AWS Integration
- Secure credential handling with password-masked input fields
- Support for both permanent and temporary AWS credentials
- Automatic session management for AWS services

### Model Flexibility
- Real-time model parameter adjustment
- Support for multiple Bedrock model families
- Configuration persistence during session

### Processing Status
- Real-time status updates for batch processing
- Error handling and reporting
- Progress monitoring with automatic refresh

### Results Management
- Structured display of answers, confidence, and explanations
- CSV download functionality for batch results
- Session state management for result persistence

## Troubleshooting

### Common Issues

1. **AWS Credentials Error**: Ensure your credentials are valid and have the necessary permissions for Bedrock and S3
2. **Document Loading Failed**: Check S3 bucket and key are correct and accessible
3. **Model Invocation Error**: Verify the selected model is available in your AWS region
4. **Questions Not Loading**: Ensure CSV format matches expected structure

### Performance Notes

- Batch processing may take several minutes depending on the number of questions
- The GUI automatically refreshes during batch processing
- Large documents may take longer to process initially but are cached for subsequent use

## Security Considerations

- AWS credentials are handled securely and not stored persistently
- Session tokens are supported for enhanced security
- All processing occurs within your AWS environment
- No data is sent to external services beyond AWS Bedrock

## Support

For issues or questions about the GUI, please refer to the main project documentation or contact the development team. 