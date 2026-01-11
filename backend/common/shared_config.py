################################################################################
#                         Configuration Settings                               #
################################################################################

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys - loaded from environment variables for security
openai_api_key = os.getenv('OPENAI_API_KEY', '')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
serper_api_key = os.getenv('SERPER_API_KEY', '')
groq_api_key = os.getenv('GROQ_API_KEY', '')
gemini_api_key = os.getenv('GEMINI_API_KEY', '')
azure_api_key = os.getenv('AZURE_API_KEY', '')

# LiteLLM Default Model Configuration
# Format: provider/model-name (e.g., openai/gpt-4o-mini, anthropic/claude-3-5-sonnet-20241022, groq/llama-3.1-70b-versatile)
default_model_name = os.getenv('DEFAULT_MODEL_NAME', 'openai/gpt-4o-mini')
default_temperature = float(os.getenv('DEFAULT_TEMPERATURE', '0.5'))
default_max_tokens = int(os.getenv('DEFAULT_MAX_TOKENS', '2048'))

# Legacy settings (deprecated - use default_model_name instead)
model_name = default_model_name  # For backward compatibility
api_key = openai_api_key
base_url = os.getenv('BASE_URL', '')

# Other settings
random_seed = int(os.getenv('RANDOM_SEED', '1'))

# Debug settings
litellm_log = os.getenv('LITELLM_LOG', '')  # Set to 'DEBUG' for detailed logs
