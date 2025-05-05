import json
import boto3
from typing import Dict, Any, Optional
from .models import BedrockModelConfig


class BedrockClient:
    """Client for interacting with AWS Bedrock models."""
    
    def __init__(
        self,
        aws_region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        model_config: Optional[BedrockModelConfig] = None
    ):
        """
        Initialize the Bedrock client.
        
        Args:
            aws_region: AWS region to use
            aws_access_key_id: AWS access key ID (optional if using IAM roles)
            aws_secret_access_key: AWS secret access key (optional if using IAM roles)
            model_config: Configuration for the Bedrock model
        """
        self.aws_region = aws_region
        
        # Create session with optional credentials
        session_kwargs = {"region_name": aws_region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
        
        session = boto3.Session(**session_kwargs)
        self.bedrock_runtime = session.client(
            service_name="bedrock-runtime"
        )
        
        # Set default model config if not provided
        self.model_config = model_config or BedrockModelConfig(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            temperature=0.0,
            max_tokens=1024,
            top_p=0.9
        )
    
    def invoke_model(self, prompt: str) -> str:
        """
        Invoke the Bedrock model with a prompt.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response text
        """
        model_id = self.model_config.model_id
        
        # Handle different model providers
        if model_id.startswith("anthropic.claude"):
            return self._invoke_claude(prompt)
        elif model_id.startswith("amazon.titan"):
            return self._invoke_titan(prompt)
        else:
            raise ValueError(f"Unsupported model: {model_id}")
    
    def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude model with appropriate formatting."""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.model_config.max_tokens,
            "temperature": self.model_config.temperature,
            "top_p": self.model_config.top_p,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_config.model_id,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response.get("body").read())
        return response_body.get("content")[0].get("text")
    
    def _invoke_titan(self, prompt: str) -> str:
        """Invoke Titan model with appropriate formatting."""
        request_body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": self.model_config.max_tokens,
                "temperature": self.model_config.temperature,
                "topP": self.model_config.top_p
            }
        }
        
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_config.model_id,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response.get("body").read())
        return response_body.get("results")[0].get("outputText") 