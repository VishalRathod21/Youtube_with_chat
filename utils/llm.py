from typing import Optional
from groq_llm import GroqLLM
import os

def get_llm(api_key: Optional[str] = None, model_name: str = "mixtral-8x7b-32768", temperature: float = 0.7):
    """
    Get a language model instance.
    
    Args:
        api_key: Optional API key. If not provided, will try to get from environment.
        model_name: Name of the model to use.
        temperature: Temperature for generation.
        
    Returns:
        An instance of the language model.
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key is required. Please set GROQ_API_KEY in your environment variables.")
        
    return GroqLLM(api_key=api_key, model_name=model_name)
