import streamlit as st
import pandas as pd
import os
import tempfile
import json
from typing import List, Optional, Dict, Any
import asyncio
import threading
import time
from pathlib import Path

from audit_processor import AuditProcessor
from common.core import BedrockModelConfig
from question_structure import QuestionTree, AnswerType


# Page configuration
st.set_page_config(
    page_title="Audit Question Processing GUI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'questions_df' not in st.session_state:
    st.session_state.questions_df = None
if 'question_tree' not in st.session_state:
    st.session_state.question_tree = None
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}


def load_questions_from_csv(csv_file) -> Optional[pd.DataFrame]:
    """Load questions from uploaded CSV file."""
    try:
        df = pd.read_csv(csv_file)
        # Ensure ID column is treated as string
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None


def create_processor(aws_credentials: Dict[str, str], model_config: BedrockModelConfig, use_rag_query_rewriting: bool = False) -> Optional[AuditProcessor]:
    """Create an AuditProcessor with the given credentials and configuration."""
    try:
        processor = AuditProcessor(
            aws_region=aws_credentials.get('region', 'us-gov-west-1'),
            aws_access_key_id=aws_credentials.get('access_key_id'),
            aws_secret_access_key=aws_credentials.get('secret_access_key'),
            aws_session_token=aws_credentials.get('session_token'),
            use_rag_query_rewriting=use_rag_query_rewriting
        )
        
        # Update model configurations for all approaches
        processor.memory_qa.bedrock_client.model_config = model_config
        processor.rag_qa.bedrock_client.model_config = model_config
        processor.hyde_qa.bedrock_client.model_config = model_config
        
        return processor
    except Exception as e:
        st.error(f"Error creating processor: {e}")
        return None


def process_single_question(processor: AuditProcessor, question_id: str, approaches: List[str], use_rag_query_rewriting: bool = False) -> Dict[str, Any]:
    """Process a single question with selected approaches."""
    results = {}
    
    # Ensure question_id is a string
    question_id = str(question_id).strip()
    
    # Debug: Check if question exists
    question = processor.question_tree.get_question(question_id)
    if not question:
        available_ids = list(processor.question_tree.questions.keys())
        st.error(f"Question '{question_id}' not found. Available question IDs: {available_ids}")
        return {'error': f"Question '{question_id}' not found"}
    
    for approach in approaches:
        try:
            if approach.lower() == "rag":
                success = processor.process_with_rag(question_id, use_query_rewriting=use_rag_query_rewriting)
                if success:
                    question = processor.question_tree.get_question(question_id)
                    results['rag'] = {
                        'answer': question.rag_answer.value if question.rag_answer else 'N/A',
                        'confidence': question.rag_confidence,
                        'explanation': question.rag_explanation
                    }
            elif approach.lower() == "memory":
                success = processor.process_with_memory(question_id)
                if success:
                    question = processor.question_tree.get_question(question_id)
                    results['memory'] = {
                        'answer': question.context_answer.value if question.context_answer else 'N/A',
                        'confidence': question.context_confidence,
                        'explanation': question.context_explanation
                    }
            elif approach.lower() == "hyde":
                success = processor.process_with_hyde(question_id)
                if success:
                    question = processor.question_tree.get_question(question_id)
                    results['hyde'] = {
                        'answer': question.hyde_answer.value if question.hyde_answer else 'N/A',
                        'confidence': question.hyde_confidence,
                        'explanation': question.hyde_explanation
                    }
        except Exception as e:
            results[approach] = {'error': str(e)}
    
    return results


def run_batch_processing(processor: AuditProcessor, approaches: List[str], use_rag_query_rewriting: bool = False):
    """Run batch processing in a separate thread."""
    try:
        st.session_state.processing_status['status'] = 'running'
        st.session_state.processing_status['progress'] = 0
        
        success = processor.process_all_questions(approaches, use_rag_query_rewriting=use_rag_query_rewriting)
        
        if success:
            # Save results to a temporary file and then to S3 (if configured)
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            processor.save_results(temp_file.name)
            
            st.session_state.processing_status['status'] = 'completed'
            st.session_state.processing_status['results_file'] = temp_file.name
        else:
            st.session_state.processing_status['status'] = 'error'
            st.session_state.processing_status['error'] = 'Some questions could not be processed'
            
    except Exception as e:
        st.session_state.processing_status['status'] = 'error'
        st.session_state.processing_status['error'] = str(e)


def main():
    st.title("📊 Audit Question Processing GUI")
    st.markdown("Process audit questions using RAG, Memory, and HYDE approaches")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # AWS Credentials Section
        st.subheader("🔐 AWS Credentials")
        aws_access_key_id = st.text_input("AWS Access Key ID", type="password", help="Your AWS access key ID")
        aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password", help="Your AWS secret access key")
        aws_session_token = st.text_input("AWS Session Token (Optional)", type="password", help="AWS session token for temporary credentials")
        aws_region = st.selectbox("AWS Region", ["us-gov-west-1", "us-east-1", "us-west-2", "eu-west-1"], index=0)
        
        # Model Configuration Section
        st.subheader("🤖 Model Configuration")
        model_id = st.selectbox(
            "Model ID",
            [
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "anthropic.claude-3-opus-20240229-v1:0",
                "amazon.titan-text-express-v1",
                "amazon.titan-text-lite-v1"
            ],
            index=0
        )
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, help="Controls randomness in responses")
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1, help="Controls diversity of responses")
        max_tokens = st.number_input("Max Tokens", min_value=100, max_value=4000, value=1024, help="Maximum number of tokens to generate")
        
        # RAG Configuration
        st.subheader("🔍 RAG Configuration")
        use_rag_query_rewriting = st.checkbox("Use RAG Query Rewriting", value=False, help="Enable query rewriting for better vector search")
        
        # Document Configuration
        st.subheader("📄 Document Configuration")
        s3_bucket = st.text_input("S3 Bucket", help="S3 bucket containing the document")
        s3_key = st.text_input("S3 Key", help="S3 key for the document")
        force_reprocess = st.checkbox("Force Reprocess", value=False, help="Force reprocessing even if processed document exists")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📋 Questions")
        
        # File upload for questions
        uploaded_file = st.file_uploader("Upload Questions CSV", type=['csv'], help="Upload a CSV file with audit questions")
        
        if uploaded_file is not None:
            questions_df = load_questions_from_csv(uploaded_file)
            if questions_df is not None:
                st.session_state.questions_df = questions_df
                st.success(f"Loaded {len(questions_df)} questions")
                
                # Display questions preview
                st.subheader("Questions Preview")
                st.dataframe(questions_df[['id', 'question']].head(10), use_container_width=True)
        
        # Load default sample questions if available
        if st.session_state.questions_df is None and Path("sample_questions.csv").exists():
            if st.button("Load Sample Questions"):
                try:
                    questions_df = pd.read_csv("sample_questions.csv")
                    # Ensure ID column is treated as string
                    if 'id' in questions_df.columns:
                        questions_df['id'] = questions_df['id'].astype(str).str.strip()
                    st.session_state.questions_df = questions_df
                    st.success(f"Loaded {len(st.session_state.questions_df)} sample questions")
                except Exception as e:
                    st.error(f"Error loading sample questions: {e}")
    
    with col2:
        st.header("🚀 Processing")
        
        # Check if we have all required inputs
        has_credentials = aws_access_key_id and aws_secret_access_key
        has_document = s3_bucket and s3_key
        has_questions = st.session_state.questions_df is not None
        
        if not has_credentials:
            st.warning("⚠️ Please provide AWS credentials in the sidebar")
        if not has_document:
            st.warning("⚠️ Please provide S3 bucket and key for the document")
        if not has_questions:
            st.warning("⚠️ Please upload a questions CSV file")
        
        if has_credentials and has_document and has_questions:
            # Create processor button
            if st.button("🔧 Initialize Processor", type="primary"):
                with st.spinner("Initializing processor and loading document..."):
                    # Create model config
                    model_config = BedrockModelConfig(
                        model_id=model_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                    
                    # Create AWS credentials dict
                    aws_credentials = {
                        'access_key_id': aws_access_key_id,
                        'secret_access_key': aws_secret_access_key,
                        'session_token': aws_session_token if aws_session_token else None,
                        'region': aws_region
                    }
                    
                    # Create processor
                    processor = create_processor(aws_credentials, model_config, use_rag_query_rewriting)
                    
                    if processor:
                        # Load document
                        success = processor.load_document(s3_bucket, s3_key, force_reprocess=force_reprocess)
                        if success:
                            # Create temporary CSV file for questions
                            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                            st.session_state.questions_df.to_csv(temp_csv.name, index=False)
                            temp_csv.close()
                            
                            # Load questions
                            if processor.load_questions(temp_csv.name):
                                st.session_state.processor = processor
                                st.session_state.document_loaded = True
                                st.success("✅ Processor initialized and document loaded successfully!")
                                
                                # Debug: Show loaded question IDs
                                loaded_ids = list(processor.question_tree.questions.keys())
                                st.info(f"Loaded question IDs: {loaded_ids}")
                            else:
                                st.error("❌ Failed to load questions")
                        else:
                            st.error("❌ Failed to load document")
            
            # Single question processing
            if st.session_state.processor and st.session_state.document_loaded:
                st.subheader("🎯 Single Question Processing")
                
                # Question selection
                question_options = [(str(row['id']).strip(), f"{str(row['id']).strip()}: {row['question'][:100]}...") for _, row in st.session_state.questions_df.iterrows()]
                selected_question = st.selectbox(
                    "Select Question",
                    options=[opt[0] for opt in question_options],
                    format_func=lambda x: next(opt[1] for opt in question_options if opt[0] == x)
                )
                
                # Approach selection
                st.write("Select Approaches:")
                col_rag, col_memory, col_hyde = st.columns(3)
                with col_rag:
                    use_rag = st.checkbox("RAG", value=True)
                with col_memory:
                    use_memory = st.checkbox("Memory", value=True)
                with col_hyde:
                    use_hyde = st.checkbox("HYDE", value=True)
                
                selected_approaches = []
                if use_rag:
                    selected_approaches.append("rag")
                if use_memory:
                    selected_approaches.append("memory")
                if use_hyde:
                    selected_approaches.append("hyde")
                
                if selected_approaches and st.button("🔍 Process Question"):
                    with st.spinner("Processing question..."):
                        results = process_single_question(
                            st.session_state.processor, 
                            selected_question, 
                            selected_approaches,
                            use_rag_query_rewriting
                        )
                        
                        # Display results
                        if 'error' in results:
                            st.error(f"Error: {results['error']}")
                        else:
                            st.subheader("📊 Results")
                            for approach, result in results.items():
                                with st.expander(f"{approach.upper()} Results"):
                                    if 'error' in result:
                                        st.error(f"Error: {result['error']}")
                                    else:
                                        st.write(f"**Answer:** {result.get('answer', 'N/A')}")
                                        st.write(f"**Confidence:** {result.get('confidence', 'N/A')}")
                                        st.write(f"**Explanation:** {result.get('explanation', 'N/A')}")
                
                # Batch processing
                st.subheader("📦 Batch Processing")
                st.write("Process all questions with selected approaches")
                
                col_batch_rag, col_batch_memory, col_batch_hyde = st.columns(3)
                with col_batch_rag:
                    batch_use_rag = st.checkbox("RAG", value=True, key="batch_rag")
                with col_batch_memory:
                    batch_use_memory = st.checkbox("Memory", value=True, key="batch_memory")
                with col_batch_hyde:
                    batch_use_hyde = st.checkbox("HYDE", value=True, key="batch_hyde")
                
                batch_approaches = []
                if batch_use_rag:
                    batch_approaches.append("rag")
                if batch_use_memory:
                    batch_approaches.append("memory")
                if batch_use_hyde:
                    batch_approaches.append("hyde")
                
                if batch_approaches:
                    if st.button("🚀 Start Batch Processing", type="primary"):
                        # Initialize processing status
                        st.session_state.processing_status = {'status': 'starting'}
                        
                        # Start processing in a thread
                        thread = threading.Thread(
                            target=run_batch_processing,
                            args=(st.session_state.processor, batch_approaches, use_rag_query_rewriting)
                        )
                        thread.start()
                        st.success("🚀 Batch processing started!")
                        st.rerun()
                
                # Display processing status
                if 'status' in st.session_state.processing_status:
                    status = st.session_state.processing_status['status']
                    
                    if status == 'starting':
                        st.info("🔄 Starting batch processing...")
                    elif status == 'running':
                        st.info("🔄 Processing questions... This may take several minutes.")
                        # Auto-refresh every 5 seconds
                        time.sleep(5)
                        st.rerun()
                    elif status == 'completed':
                        st.success("✅ Batch processing completed successfully!")
                        st.info("📁 Results have been saved and stored in S3 as configured.")
                        
                        # Offer to download results
                        if 'results_file' in st.session_state.processing_status:
                            try:
                                with open(st.session_state.processing_status['results_file'], 'r') as f:
                                    results_csv = f.read()
                                st.download_button(
                                    label="📥 Download Results CSV",
                                    data=results_csv,
                                    file_name="audit_results.csv",
                                    mime="text/csv"
                                )
                            except Exception as e:
                                st.error(f"Error reading results file: {e}")
                        
                        # Clear status after showing
                        if st.button("🔄 Reset Status"):
                            st.session_state.processing_status = {}
                            st.rerun()
                            
                    elif status == 'error':
                        st.error(f"❌ Error during batch processing: {st.session_state.processing_status.get('error', 'Unknown error')}")
                        if st.button("🔄 Reset Status"):
                            st.session_state.processing_status = {}
                            st.rerun()


if __name__ == "__main__":
    main() 