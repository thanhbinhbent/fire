
"""Specific configuration for the FIRE framewrok."""

from common import shared_config

# search_type: str = Google Search API used. Choose from ['serper'].
search_type = 'serper'
num_searches = 3

max_steps = 2  # Reduced for performance (target: 1-5s)
max_retries = 10
max_tolerance = 2
diverse_prompt = False