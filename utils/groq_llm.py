import os
import json
from typing import Optional, Dict, Any, List, Union
from groq import Groq, GroqError

class GroqLLM:
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192"):
        """
        Initialize Groq LLM with API key and model name.
        
        Args:
            api_key: Your Groq API key (starts with 'gsk_')
            model_name: Name of the model to use (default: llama3-8b-8192)
                      Supported models: llama3-8b-8192, llama3-70b-8192, gemma-7b-it
            
        Raises:
            ValueError: If API key is invalid or empty
            GroqError: If there's an issue with the Groq API
        """
        if not api_key or not isinstance(api_key, str) or not api_key.startswith('gsk_'):
            raise ValueError("Invalid Groq API key. Please provide a valid API key that starts with 'gsk_'.")
        
        # List of supported models and their context lengths
        supported_models = {
            "llama3-8b-8192": 8192,     # Llama 3 8B (recommended for most use cases)
            "llama3-70b-8192": 8192,    # Llama 3 70B (larger, more capable)
            "gemma-7b-it": 8192,        # Gemma 7B (good balance of speed and quality)
            "mixtral-8x7b-32768": 32768 # Deprecated, kept for backward compatibility
        }
        
        if model_name not in supported_models:
            raise ValueError(f"Unsupported model: {model_name}. Supported models are: {', '.join(supported_models.keys())}")
            
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = min(8192, supported_models[model_name])
        
        # Initialize Groq client with timeout
        self.client = Groq(
            api_key=api_key,
            timeout=30.0  # 30 second timeout
        )
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024, 
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text using Groq API.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system message to set the assistant's behavior
            max_tokens: Maximum number of tokens to generate (1-8192)
            temperature: Controls randomness (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            stop: Up to 4 sequences where the API will stop generating further tokens
            stream: Whether to stream the response
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If input parameters are invalid
            GroqError: If there's an error with the Groq API
        """
        # Validate inputs
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not (0 <= temperature <= 1.0):  # Groq's temperature range is 0-1
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if not (1 <= max_tokens <= 8192):
            raise ValueError("max_tokens must be between 1 and 8192")
        if not (0 < top_p <= 1.0):
            raise ValueError("top_p must be between 0 and 1")
            
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt.strip()})
            messages.append({"role": "user", "content": prompt.strip()})
            
            # Prepare request parameters
            request_params = {
                "messages": messages,
                "model": self.model_name,
                "temperature": max(0.0, min(float(temperature), 1.0)),  # Ensure within 0-1 range
                "max_tokens": min(int(max_tokens), 8192),  # Ensure within max limit
                "top_p": max(0.0, min(float(top_p), 1.0)),  # Ensure within 0-1 range
            }
            
            # Add stop sequences if provided
            if stop:
                if isinstance(stop, str):
                    request_params["stop"] = [stop]
                elif isinstance(stop, list) and len(stop) > 0:
                    request_params["stop"] = stop[:4]  # Max 4 stop sequences
            
            # Add streaming flag if needed
            if stream:
                request_params["stream"] = True
            
            # Make API call
            response = self.client.chat.completions.create(**request_params)
            
            # Handle response
            if stream:
                # For streaming responses, collect chunks
                collected_chunks = []
                for chunk in response:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                        collected_chunks.append(chunk.choices[0].delta.content)
                return "".join(collected_chunks)
            else:
                # For non-streaming, return the content directly
                if hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
                return ""  # Return empty string if no content
                
        except GroqError as e:
            # Handle specific Groq API errors
            error_msg = str(e)
            error_details = str(e.response.text) if hasattr(e, 'response') and hasattr(e.response, 'text') else str(e)
            
            # Log the full error details for debugging
            import json
            try:
                error_data = json.loads(error_details) if error_details.startswith('{') else {}
                error_type = error_data.get('error', {}).get('type', 'unknown_error')
                error_message = error_data.get('error', {}).get('message', error_msg)
                
                # Log detailed error information
                print(f"Groq API Error - Type: {error_type}")
                print(f"Message: {error_message}")
                print(f"Full Error: {error_details}")
                
                # Handle specific error types
                if "invalid_api_key" in error_type.lower() or "401" in error_msg:
                    raise GroqError("Invalid API key. Please check your Groq API key and try again.")
                elif "rate_limit" in error_type.lower() or "429" in error_msg:
                    raise GroqError("Rate limit exceeded. Please try again later.")
                elif "invalid_request" in error_type.lower() or "400" in error_msg:
                    # Provide more detailed error message for 400 errors
                    if "model_not_found" in error_type.lower():
                        raise GroqError(f"Model not found. Please check the model name. Error: {error_message}")
                    elif "messages" in error_message.lower():
                        raise GroqError(f"Invalid message format: {error_message}")
                    else:
                        raise GroqError(f"Invalid request: {error_message}")
                else:
                    raise GroqError(f"Groq API error: {error_message}")
                    
            except json.JSONDecodeError:
                # If we can't parse the error as JSON, fall back to the original message
                if "401" in error_msg:
                    raise GroqError("Invalid API key. Please check your Groq API key and try again.")
                elif "429" in error_msg:
                    raise GroqError("Rate limit exceeded. Please try again later.")
                elif "400" in error_msg:
                    raise GroqError(f"Invalid request: {error_msg}")
                else:
                    raise GroqError(f"Error from Groq API: {error_msg}")
        except Exception as e:
            raise Exception(f"Unexpected error while generating response: {str(e)}")

# Example usage:
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("Please set GROQ_API_KEY in your environment variables or .env file")
    else:
        llm = GroqLLM(api_key=api_key)
        response = llm.generate("Tell me a short joke about AI")
        print(response)
