
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY', '')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
serper_api_key = os.getenv('SERPER_API_KEY', '')
groq_api_key = os.getenv('GROQ_API_KEY', '')
gemini_api_key = os.getenv('GEMINI_API_KEY', '')
azure_api_key = os.getenv('AZURE_API_KEY', '')

default_model_name = os.getenv('DEFAULT_MODEL_NAME', 'openai/gpt-4o-mini')
default_temperature = float(os.getenv('DEFAULT_TEMPERATURE', '0.5'))
default_max_tokens = int(os.getenv('DEFAULT_MAX_TOKENS', '2048'))

model_name = default_model_name
api_key = openai_api_key
base_url = os.getenv('BASE_URL', '')

random_seed = int(os.getenv('RANDOM_SEED', '1'))

litellm_log = os.getenv('LITELLM_LOG', '')
# Semantic model and optimization settings
SEMANTIC_MODEL_CACHE = True
SEMANTIC_MODEL_NAME = os.getenv('SEMANTIC_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')
QUERY_DEDUP_THRESHOLD = float(os.getenv('QUERY_DEDUP_THRESHOLD', '0.85'))