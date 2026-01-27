"""Class for querying the Google Serper API with Vietnamese enhancements."""

import random
import time
from typing import Any, Optional, Literal, List, Dict

import requests

_SERPER_URL = 'https://google.serper.dev'
NO_RESULT_MSG = 'No good Google Search result was found'

try:
    from common.vietnamese_utils import preprocessor
    VIETNAMESE_SUPPORT = True
except ImportError:
    VIETNAMESE_SUPPORT = False
    print("Vietnamese utilities not available")


class SerperAPI:
  """Class for querying the Google Serper API."""

  def __init__(
      self,
      serper_api_key: str,
      gl: str = 'us',
      hl: str = 'en',
      k: int = 1,
      tbs: Optional[str] = None,
      search_type: Literal['news', 'search', 'places', 'images'] = 'search',
  ):
    self.serper_api_key = serper_api_key
    self.gl = gl
    self.hl = hl
    self.k = k
    self.tbs = tbs
    self.search_type = search_type
    self.result_key_for_type = {
        'news': 'news',
        'places': 'places',
        'images': 'images',
        'search': 'organic',
    }

  def run(self, query: str, **kwargs: Any) -> str:
    """Run query through GoogleSearch and parse result."""
    assert self.serper_api_key, 'Missing serper_api_key.'
    results = self._google_serper_api_results(
        query,
        gl=self.gl,
        hl=self.hl,
        num=self.k,
        tbs=self.tbs,
        search_type=self.search_type,
        **kwargs,
    )

    return self._parse_results(results)

  def _google_serper_api_results(
      self,
      search_term: str,
      search_type: str = 'search',
      max_retries: int = 20,
      **kwargs: Any,
  ) -> dict[Any, Any]:
    """Run query through Google Serper."""
    headers = {
        'X-API-KEY': self.serper_api_key or '',
        'Content-Type': 'application/json',
    }
    params = {
        'q': search_term,
        **{key: value for key, value in kwargs.items() if value is not None},
    }
    response, num_fails, sleep_time = None, 0, 0

    while not response and num_fails < max_retries:
      try:
        response = requests.post(
            f'{_SERPER_URL}/{search_type}', headers=headers, params=params
        )
      except AssertionError as e:
        raise e
      except Exception:  # pylint: disable=broad-exception-caught
        response = None
        num_fails += 1
        sleep_time = min(sleep_time * 2, 600)
        sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
        time.sleep(sleep_time)

    if not response:
      raise ValueError('Failed to get result from Google Serper API')

    response.raise_for_status()
    search_results = response.json()
    return search_results

  def _parse_snippets(self, results: dict[Any, Any]) -> list[str]:
    """Parse results."""
    snippets = []

    if results.get('answerBox'):
      answer_box = results.get('answerBox', {})
      answer = answer_box.get('answer')
      snippet = answer_box.get('snippet')
      snippet_highlighted = answer_box.get('snippetHighlighted')

      if answer and isinstance(answer, str):
        snippets.append(answer)
      if snippet and isinstance(snippet, str):
        snippets.append(snippet.replace('\n', ' '))
      if snippet_highlighted:
        snippets.append(snippet_highlighted)

    if results.get('knowledgeGraph'):
      kg = results.get('knowledgeGraph', {})
      title = kg.get('title')
      entity_type = kg.get('type')
      description = kg.get('description')

      if entity_type:
        snippets.append(f'{title}: {entity_type}.')

      if description:
        snippets.append(description)

      for attribute, value in kg.get('attributes', {}).items():
        snippets.append(f'{title} {attribute}: {value}.')

    result_key = self.result_key_for_type[self.search_type]

    if result_key in results:
      for result in results[result_key][:self.k]:
        if 'snippet' in result:
          snippets.append(result['snippet'])

        for attribute, value in result.get('attributes', {}).items():
          snippets.append(f'{attribute}: {value}.')

    if not snippets:
      return [NO_RESULT_MSG]

    return snippets

  def _parse_results(self, results: dict[Any, Any]) -> str:
    return ' '.join(self._parse_snippets(results))


def enhance_vietnamese_query(
    query: str,
    claim: Optional[str] = None,
    prefer_vietnamese: bool = True
) -> str:
    """
    Enhance search query for Vietnamese content with entity extraction.
    
    VIETNAMESE-AWARE ENHANCEMENTS:
    1. Add Vietnamese site filters (trusted news sources)
    2. Extract and emphasize key entities from claim
    3. Add language-specific search operators
    4. Add current year for time-sensitive queries
    
    Args:
        query: Original search query
        claim: Original claim (optional, for entity extraction)
        prefer_vietnamese: If True, add Vietnamese site filters
    
    Returns:
        Enhanced query string
    """
    enhanced = query
    
    # Add current year for time-sensitive queries
    from datetime import datetime
    current_year = datetime.now().year
    time_keywords = ['hiện nay', 'hiện tại', 'bây giờ', 'current', 'now', 'present']
    if any(keyword in query.lower() for keyword in time_keywords):
        if str(current_year) not in query:
            enhanced = f"{enhanced} {current_year}"
    
    if claim and prefer_vietnamese and VIETNAMESE_SUPPORT:
        try:
            processed = preprocessor.preprocess_claim(claim)
            entities = [e['text'] for e in processed['entities']]
            
            for entity in entities[:3]:
                if entity.lower() not in query.lower():
                    enhanced = f"{enhanced} {entity}"
        except Exception as e:
            print(f"Entity extraction failed: {e}")
    
    # Use semantic analysis instead of hardcoded keywords
    # Only add Vietnamese site filter for Vietnamese-related queries
    if prefer_vietnamese:
        try:
            from common.claim_analyzer import get_analyzer
            analyzer = get_analyzer()
            
            # Let LLM decide if should use Vietnamese sources
            should_use_vn_sources = analyzer.should_use_vietnamese_sources(claim if claim else query)
            
            if should_use_vn_sources:
                tier1_sources = ["vnexpress.net", "tuoitre.vn", "thanhnien.vn"]
                tier2_sources = ["vietnamnet.vn", "dantri.com.vn", "baochinhphu.vn"]
                
                site_filter = " OR ".join([f"site:{source}" for source in tier1_sources + tier2_sources])
                enhanced = f"({enhanced}) ({site_filter})"
        except Exception as e:
            # Fallback to simple check if analyzer fails
            vietnamese_keywords = ['việt nam', 'viet nam', 'việt', 'chính phủ', 'quốc hội', 'tổng bí thư', 'chủ tịch']
            is_vietnamese_query = any(kw in query.lower() for kw in vietnamese_keywords) or (claim and any(kw in claim.lower() for kw in vietnamese_keywords))
            
            if is_vietnamese_query:
                tier1_sources = ["vnexpress.net", "tuoitre.vn", "thanhnien.vn"]
                tier2_sources = ["vietnamnet.vn", "dantri.com.vn", "baochinhphu.vn"]
                
                site_filter = " OR ".join([f"site:{source}" for source in tier1_sources + tier2_sources])
                enhanced = f"({enhanced}) ({site_filter})"
    
    return enhanced


class VietnameseSerperAPI:
    """
    Extended SerperAPI with Vietnamese language support.
    Wraps Serper API with Vietnamese-specific enhancements.
    """
    
    def __init__(self, serper_api_key: str, k: int = 3, prefer_vietnamese: bool = True):
        """
        Initialize Vietnamese-aware search API.
        
        Args:
            serper_api_key: Serper API key
            k: Number of results to return
            prefer_vietnamese: Enable Vietnamese site filtering
        """
        self.serper_api_key = serper_api_key
        self.k = k
        self.prefer_vietnamese = prefer_vietnamese
        
        # Use tbs='qdr:y' for results from past year (more recent)
        self.base_api = SerperAPI(
            serper_api_key=serper_api_key,
            gl='vn',
            hl='vi',
            k=k,
            tbs='qdr:y'  # Filter to past year for recent information
        )
    
    def run(self, query: str, claim: Optional[str] = None, k: Optional[int] = None) -> str:
        """
        Run search with Vietnamese enhancements.
        
        Args:
            query: Search query
            claim: Original claim (for entity extraction)
            k: Number of results
        
        Returns:
            str: Search results as formatted string
        """
        if self.prefer_vietnamese:
            query = enhance_vietnamese_query(
                query,
                claim=claim,
                prefer_vietnamese=True
            )
        
        return self.base_api.run(query, k=k or self.k)

