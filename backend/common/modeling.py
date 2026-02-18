import os
import litellm
from typing import Optional, Dict, Tuple, Any
from common import utils
from common.prompts import FACT_CHECK_SYSTEM_PROMPT

litellm.suppress_debug_info = True
litellm.drop_params = True


# Default local Ollama endpoint
LOCAL_OLLAMA_URL = os.getenv('LOCAL_OLLAMA_URL', 'http://localhost:8001')


class Model:
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None, 
                 max_tokens: Optional[int] = None, show_responses: bool = False, 
                 show_prompts: bool = False, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.model_name = self._parse_model_name(model_name or os.getenv('DEFAULT_MODEL_NAME', 'openai/gpt-4o-mini'))
        self.temperature = temperature if temperature is not None else float(os.getenv('DEFAULT_TEMPERATURE', '0.5'))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv('DEFAULT_MAX_TOKENS', '2048'))
        self.show_responses = show_responses
        self.show_prompts = show_prompts
        self.api_key = api_key
        self.base_url = base_url
        self._setup_api_keys()
        self._setup_local_ollama()
        
    
    def _parse_model_name(self, model_name: str) -> str:
        if ':' in model_name:
            org, model_id = model_name.split(':', 1)
            provider_map = {
                'openai': 'openai', 
                'anthropic': 'anthropic', 
                'hf': 'huggingface',
                'local': 'ollama',
                'ollama': 'ollama',
            }
            provider = provider_map.get(org, org)
            return f"{provider}/{model_id}"
        elif '/' in model_name:
            return model_name
        else:
            return f"openai/{model_name}"
    
    def _setup_api_keys(self) -> None:
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
    
    def _setup_local_ollama(self) -> None:
        """Configure base URL for local Ollama if using ollama provider."""
        provider = self.model_name.split('/')[0].lower()
        if provider == 'ollama' and not self.base_url:
            self.base_url = LOCAL_OLLAMA_URL
            # Ollama doesn't need API key, but LiteLLM may require a placeholder
            if not self.api_key:
                self.api_key = 'ollama'  # Placeholder for local Ollama

    def generate(self, context: str, system_prompt: Optional[str] = None) -> Tuple[str, Optional[Dict]]:
        if self.show_prompts:
            print(f"\nPrompt:\n{context}\n")
        
        try:
            messages = [
                {"role": "system", "content": system_prompt or FACT_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ]
            
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["api_base"] = self.base_url
            
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                }
            
            if self.show_responses:
                print(f"\nResponse:\n{content}\n")
                if usage:
                    print(f"Usage: {usage}")
            
            return content, usage
        except Exception as e:
            print(f"LiteLLM generation error: {e}")
            raise


    def print_config(self) -> None:
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
        providers = [
            'openai', 'anthropic', 'azure', 'groq', 'gemini', 'cohere',
            'together_ai', 'huggingface', 'replicate', 'bedrock', 'vertex_ai',
            'palm', 'ollama', 'deepseek', 'mistral', 'perplexity', 'fireworks_ai'
        ]
        print("\nSupported LLM Providers (100+):")
        print("  " + ", ".join(providers))
        print("\nFull list: https://docs.litellm.ai/docs/providers")