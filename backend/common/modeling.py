"""Unified LLM interface using LiteLLM for 100+ providers."""

import os
import litellm
from typing import Optional, Dict, Tuple, Any
from common import utils

# Configure LiteLLM
litellm.suppress_debug_info = True  # Reduce verbose logging
litellm.drop_params = True  # Auto-drop unsupported params for each provider

SYS_PROMPT = 'You are a fact-checking agent responsible for verifying the accuracy of claims.'


class Model:
    """Unified LLM interface using LiteLLM.
    
    Supports 100+ providers: OpenAI, Anthropic, Azure, Groq, Gemini, Cohere, etc.
    Model format: provider/model-name or organization:model_id (legacy)
    
    Examples:
        - openai/gpt-4o-mini
        - anthropic/claude-3-5-sonnet-20241022
        - groq/llama-3.1-70b-versatile
        - gemini/gemini-1.5-pro
        - azure/gpt-4
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        show_responses: bool = False,
        show_prompts: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize model with LiteLLM.
        
        Args:
            model_name: Model identifier in format 'provider/model' or 'org:model' (legacy)
                       If None, uses DEFAULT_MODEL_NAME from .env
            temperature: Sampling temperature (0-1). If None, uses DEFAULT_TEMPERATURE from .env
            max_tokens: Maximum tokens in response. If None, uses DEFAULT_MAX_TOKENS from .env
            show_responses: Print responses after generation
            show_prompts: Print prompts before generation
            api_key: Optional API key override
            base_url: Optional base URL override
        """
        # Load defaults from environment variables
        self.model_name = self._parse_model_name(
            model_name or os.getenv('DEFAULT_MODEL_NAME', 'openai/gpt-4o-mini')
        )
        self.temperature = temperature if temperature is not None else float(
            os.getenv('DEFAULT_TEMPERATURE', '0.5')
        )
        self.max_tokens = max_tokens if max_tokens is not None else int(
            os.getenv('DEFAULT_MAX_TOKENS', '2048')
        )
        self.show_responses = show_responses
        self.show_prompts = show_prompts
        self.api_key = api_key
        self.base_url = base_url
        
        # Setup API keys from environment if not provided
        self._setup_api_keys()
        
        print(f"🚀 Initialized LiteLLM model: {self.model_name}")
    
    def _parse_model_name(self, model_name: str) -> str:
        """Parse model name to LiteLLM format.
        
        Converts legacy 'organization:model' to 'provider/model' format.
        """
        if ':' in model_name:
            # Legacy format: organization:model_id
            org, model_id = model_name.split(':', 1)
            provider_map = {
                'openai': 'openai',
                'anthropic': 'anthropic',
                'hf': 'huggingface',
            }
            provider = provider_map.get(org, org)
            return f"{provider}/{model_id}"
        elif '/' in model_name:
            # Modern format: provider/model
            return model_name
        else:
            # Assume OpenAI if no prefix
            return f"openai/{model_name}"
    
    def _setup_api_keys(self) -> None:
        """Setup API keys from environment variables."""
        if not self.api_key:
            provider = self.model_name.split('/')[0].lower()
            
            key_map = {
                'openai': 'OPENAI_API_KEY',
                'anthropic': 'ANTHROPIC_API_KEY',
                'groq': 'GROQ_API_KEY',
                'gemini': 'GEMINI_API_KEY',
                'google': 'GEMINI_API_KEY',
                'azure': 'AZURE_API_KEY',
                'cohere': 'COHERE_API_KEY',
                'together': 'TOGETHER_API_KEY',
            }
            
            env_key = key_map.get(provider)
            if env_key:
                self.api_key = os.getenv(env_key, '')



    def generate(self, context: str, system_prompt: Optional[str] = None) -> Tuple[str, Optional[Dict]]:
        """Generate response using LiteLLM.
        
        Args:
            context: User prompt/context
            system_prompt: Optional system prompt override
            
        Returns:
            Tuple of (response_text, usage_dict)
        """
        if self.show_prompts:
            print(f"\n📝 Prompt:\n{context}\n")
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt or SYS_PROMPT},
                {"role": "user", "content": context}
            ]
            
            # Prepare kwargs
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            # Add API key if provided
            if self.api_key:
                kwargs["api_key"] = self.api_key
            
            # Add base URL if provided
            if self.base_url:
                kwargs["api_base"] = self.base_url
            
            # Call LiteLLM
            response = litellm.completion(**kwargs)
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Extract usage metadata
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                }
            
            if self.show_responses:
                print(f"\n💬 Response:\n{content}\n")
                if usage:
                    print(f"📊 Usage: {usage}")
            
            return content, usage
            
        except Exception as e:
            print(f"❌ LiteLLM generation error: {e}")
            raise


    def print_config(self) -> None:
        """Print model configuration."""
        provider = self.model_name.split('/')[0]
        model_id = self.model_name.split('/', 1)[1] if '/' in self.model_name else self.model_name
        
        settings = {
            'provider': provider,
            'model': model_id,
            'full_model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'show_responses': self.show_responses,
            'show_prompts': self.show_prompts,
            'api_key_set': bool(self.api_key),
            'base_url': self.base_url or 'default',
        }
        print(utils.to_readable_json(settings))
    
    @staticmethod
    def list_supported_providers() -> None:
        """List all supported LiteLLM providers."""
        providers = [
            'openai', 'anthropic', 'azure', 'groq', 'gemini', 'cohere',
            'together_ai', 'huggingface', 'replicate', 'bedrock', 'vertex_ai',
            'palm', 'ollama', 'deepseek', 'mistral', 'perplexity', 'fireworks_ai'
        ]
        print("\n🌐 Supported LLM Providers (100+):")
        print("  " + ", ".join(providers))
        print("\n📚 Full list: https://docs.litellm.ai/docs/providers")





