
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY', '')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
serper_api_key = os.getenv('SERPER_API_KEY', '')
groq_api_key = os.getenv('GROQ_API_KEY', '')
gemini_api_key = os.getenv('GEMINI_API_KEY', '')
azure_api_key = os.getenv('AZURE_API_KEY', '')
cohere_api_key = os.getenv('COHERE_API_KEY', '')
together_api_key = os.getenv('TOGETHER_API_KEY', '')

default_model_name = os.getenv('DEFAULT_MODEL_NAME', 'ollama/qwen3:1.7b')
default_temperature = float(os.getenv('DEFAULT_TEMPERATURE', '0.3'))
default_max_tokens = int(os.getenv('DEFAULT_MAX_TOKENS', '1500'))

model_name = default_model_name
api_key = openai_api_key
base_url = os.getenv('BASE_URL', '')

random_seed = int(os.getenv('RANDOM_SEED', '1'))

litellm_log = os.getenv('LITELLM_LOG', '')

# Local Ollama
LOCAL_OLLAMA_URL = os.getenv('LOCAL_OLLAMA_URL', 'http://localhost:11434')

# Database
factcheck_db_path = os.getenv('FACTCHECK_DB_PATH', 'dev.db')