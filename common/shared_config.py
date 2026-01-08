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

# Other settings
random_seed = int(os.getenv('RANDOM_SEED', '1'))
