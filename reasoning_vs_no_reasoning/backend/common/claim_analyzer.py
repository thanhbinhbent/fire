"""
Intelligent claim analyzer using LLM and embeddings.
Replaces hardcoded keyword matching with dynamic semantic analysis.
"""

from typing import Dict, Optional
import json


class ClaimAnalyzer:
    """
    Analyze claims using LLM to determine:
    1. Claim type (current events, historical, basic fact, etc.)
    2. Temporal sensitivity
    3. Key entities and search terms
    
    This replaces hardcoded keyword checking.
    """
    
    def __init__(self, model=None):
        """
        Initialize with LLM model.
        
        Args:
            model: LLM model instance (optional, can use on-demand)
        """
        self.model = model
        self._cache = {}  # Cache for repeated claims
    
    def analyze_claim(self, claim: str, model=None) -> Dict:
        """
        Analyze claim using LLM to get metadata.
        
        Returns:
            {
                'claim_type': 'current_event' | 'historical_fact' | 'basic_fact' | 'temporal_specific',
                'temporal_sensitivity': 'high' | 'medium' | 'low',
                'requires_recent_info': bool,
                'has_specific_date': bool,
                'key_entities': [str],
                'suggested_query': str,
                'confidence': float
            }
        """
        # Check cache first
        if claim in self._cache:
            return self._cache[claim]
        
        # Use provided model or instance model
        llm = model or self.model
        if not llm:
            # Fallback to simple heuristics
            return self._fallback_analysis(claim)
        
        # LLM-based analysis
        analysis_prompt = f"""Analyze this claim and return a JSON object with metadata.

Claim: "{claim}"

Return JSON with these fields:
{{
    "claim_type": "current_event" | "historical_fact" | "basic_fact" | "temporal_specific",
    "temporal_sensitivity": "high" | "medium" | "low",
    "requires_recent_info": true/false,
    "has_specific_date": true/false,
    "key_entities": ["entity1", "entity2"],
    "suggested_query": "optimized search query",
    "language": "vi" | "en" | "mixed"
}}

Rules:
- claim_type:
  * "current_event": About present situations (requires latest data)
  * "historical_fact": Past events with known dates
  * "basic_fact": Geography, definitions, well-known facts
  * "temporal_specific": Claims with specific dates to verify
  
- temporal_sensitivity:
  * "high": Information changes frequently (current positions, prices)
  * "medium": Changes occasionally (laws, policies)
  * "low": Rarely changes (history, geography)

- suggested_query: Extract 3-5 key terms for effective search (no full sentences)

Return ONLY valid JSON, no explanations."""

        try:
            # Note: Model.generate() doesn't accept max_tokens/temperature parameters
            # These are set in Model.__init__() instead
            response, _ = llm.generate(analysis_prompt)
            
            # Extract JSON from response
            from common.utils import extract_json_from_output
            analysis = extract_json_from_output(response)
            
            if analysis and self._validate_analysis(analysis):
                # Cache result
                self._cache[claim] = analysis
                return analysis
            else:
                return self._fallback_analysis(claim)
                
        except Exception as e:
            print(f"LLM analysis failed: {e}, using fallback")
            return self._fallback_analysis(claim)
    
    def _validate_analysis(self, analysis: Dict) -> bool:
        """Validate LLM analysis output."""
        required_fields = ['claim_type', 'temporal_sensitivity', 'requires_recent_info', 
                          'has_specific_date', 'suggested_query']
        return all(field in analysis for field in required_fields)
    
    def _fallback_analysis(self, claim: str) -> Dict:
        """
        Simple fallback when LLM is unavailable.
        Still better than scattered keyword checks.
        """
        import re
        from common.vietnamese_utils import preprocessor
        
        # Extract entities
        try:
            processed = preprocessor.preprocess_claim(claim)
            entities = [e['text'] for e in processed.get('entities', [])[:3]]
        except:
            entities = []
        
        # Simple semantic patterns (better than keyword lists)
        claim_lower = claim.lower()
        
        # Detect temporal specificity
        has_specific_date = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b', claim))
        
        # Detect if requires recent info (semantic approach)
        temporal_indicators = ['hiện', 'nay', 'tại', 'current', 'now', 'present', 'today', 'bây giờ']
        requires_recent = any(indicator in claim_lower for indicator in temporal_indicators)
        
        # Basic heuristics for claim type
        if requires_recent:
            claim_type = 'current_event'
            temporal_sensitivity = 'high'
        elif has_specific_date:
            claim_type = 'temporal_specific'
            temporal_sensitivity = 'medium'
        elif any(kw in claim_lower for kw in ['là', 'định nghĩa', 'thủ đô', 'capital', 'definition']):
            claim_type = 'basic_fact'
            temporal_sensitivity = 'low'
        else:
            claim_type = 'historical_fact'
            temporal_sensitivity = 'medium'
        
        # Generate simple query
        if entities:
            suggested_query = ' '.join(entities[:3])
        else:
            words = claim.split()[:6]
            suggested_query = ' '.join(words)
        
        return {
            'claim_type': claim_type,
            'temporal_sensitivity': temporal_sensitivity,
            'requires_recent_info': requires_recent,
            'has_specific_date': has_specific_date,
            'key_entities': entities,
            'suggested_query': suggested_query,
            'language': 'vi' if any(c in claim for c in 'àáảãạâầấẩẫậăằắẳẵặèéẻẽẹ') else 'en',
            'confidence': 0.6  # Lower confidence for fallback
        }
    
    def is_vietnamese_related(self, claim: str, analysis: Optional[Dict] = None) -> bool:
        """
        Determine if claim is Vietnamese-related using semantic approach.
        Replaces hardcoded keyword list.
        """
        if analysis is None:
            analysis = self.analyze_claim(claim)
        
        # Check language
        if analysis.get('language') == 'vi':
            return True
        
        # Check entities
        vietnamese_entities = ['việt nam', 'viet nam', 'vietnamese', 'hà nội', 'hanoi', 
                              'sài gòn', 'saigon', 'hồ chí minh']
        claim_lower = claim.lower()
        return any(entity in claim_lower for entity in vietnamese_entities)
    
    def should_use_vietnamese_sources(self, claim: str, analysis: Optional[Dict] = None) -> bool:
        """
        Determine if Vietnamese news sources should be prioritized.
        More intelligent than simple keyword matching.
        """
        if analysis is None:
            analysis = self.analyze_claim(claim)
        
        # Vietnamese language claims -> use Vietnamese sources
        if analysis.get('language') == 'vi':
            return True
        
        # Current events about Vietnam -> use Vietnamese sources
        if (analysis.get('claim_type') == 'current_event' and 
            self.is_vietnamese_related(claim, analysis)):
            return True
        
        # Check if entities are Vietnamese
        entities = analysis.get('key_entities', [])
        vietnamese_entities = sum(1 for e in entities if self.is_vietnamese_related(e))
        return vietnamese_entities >= len(entities) / 2  # Majority Vietnamese entities


# Global instance (lazy loaded)
_analyzer = None

def get_analyzer(model=None):
    """Get or create global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ClaimAnalyzer(model)
    return _analyzer
