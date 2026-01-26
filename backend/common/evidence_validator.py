from typing import List, Dict, Optional
import re
from urllib.parse import urlparse
from datetime import datetime, timedelta
import calendar


class EvidenceValidator:
    """
    Validate evidence quality for Vietnamese fact-checking.
    Scores based on:
    1. Source credibility (trusted Vietnamese news sites)
    2. Language quality (proper Vietnamese)
    3. Relevance to claim
    """

    def __init__(self):
        # Initialize sentence transformer for semantic similarity (lazy load)
        self._semantic_model = None
        
        self.trusted_sources = {
            "tier1": [
                "baochinhphu.vn",
                "vnexpress.net",
                "vtv.vn",
                "vov.vn",
            ],
            "tier2": [
                "tuoitre.vn",
                "thanhnien.vn",
                "vietnamnet.vn",
                "dantri.com.vn",
                "zing.vn",
            ],
            "tier3": [
                "vietnamplus.vn",
                "tienphong.vn",
                "nhandan.vn",
                "laodong.vn",
                "zingnews.vn",
            ]
        }

        self.all_trusted = []
        for sources in self.trusted_sources.values():
            self.all_trusted.extend(sources)
        
        # For temporal scoring
        self.months_vi = {
            "tháng 1": 1, "tháng 2": 2, "tháng 3": 3, "tháng 4": 4,
            "tháng 5": 5, "tháng 6": 6, "tháng 7": 7, "tháng 8": 8,
            "tháng 9": 9, "tháng 10": 10, "tháng 11": 11, "tháng 12": 12,
        }

    def get_source_credibility(self, url: str) -> float:
        """
        Score source credibility based on domain.

        Args:
            url: Source URL

        Returns:
            float: Credibility score (0-1)
        """
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace("www.", "")

            if domain in self.trusted_sources["tier1"]:
                return 1.0
            elif domain in self.trusted_sources["tier2"]:
                return 0.8
            elif domain in self.trusted_sources["tier3"]:
                return 0.6
            elif any(trusted in domain for trusted in self.all_trusted):
                return 0.5
            else:
                if domain.endswith('.vn'):
                    return 0.4
                return 0.3

        except Exception:
            return 0.2

    def check_language_quality(self, text: str) -> float:
        """
        Check if text is proper Vietnamese.

        Args:
            text: Evidence text

        Returns:
            float: Language quality score (0-1)
        """
        if not text:
            return 0.0

        score = 0.5
        vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        has_vietnamese = any(char in text.lower() for char in vietnamese_chars)
        if has_vietnamese:
            score += 0.2

        has_punctuation = any(p in text for p in ".!?")
        if has_punctuation:
            score += 0.15

        if len(text) > 50:
            score += 0.15

        return min(score, 1.0)

    def calculate_relevance(
        self,
        claim: str,
        evidence: str,
        entities: Optional[List[str]] = None
    ) -> float:
        """
        Calculate relevance score between claim and evidence.

        Args:
            claim: Original claim
            evidence: Evidence text
            entities: Optional list of key entities to check

        Returns:
            float: Relevance score (0-1)
        """
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()

        claim_words = set(claim_lower.split())
        evidence_words = set(evidence_lower.split())
        
        stopwords = {'là', 'của', 'và', 'có', 'được', 'trong', 'để', 'cho', 'với', 'từ', 'theo', 'trên'}
        claim_words = claim_words - stopwords
        evidence_words = evidence_words - stopwords
        
        overlap = len(claim_words & evidence_words)
        overlap_ratio = overlap / len(claim_words) if claim_words else 0

        entity_score = 0.0
        if entities:
            entity_matches = sum(1 for entity in entities if entity.lower() in evidence_lower)
            entity_score = entity_matches / len(entities) if entities else 0

        if entities:
            relevance = (overlap_ratio * 0.6) + (entity_score * 0.4)
        else:
            relevance = overlap_ratio

        return min(relevance, 1.0)

    def _load_semantic_model(self):
        """Lazy load sentence transformer for semantic similarity."""
        if self._semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use lightweight multilingual model for Vietnamese
                self._semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except Exception as e:
                print(f"Could not load semantic model: {e}")
                self._semantic_model = False
        return self._semantic_model

    def calculate_semantic_similarity(self, claim: str, evidence: str) -> float:
        """
        Calculate semantic similarity between claim and evidence using embeddings.
        More accurate than keyword matching for Vietnamese.
        
        Args:
            claim: Original claim
            evidence: Evidence text
            
        Returns:
            float: Semantic similarity score (0-1)
        """
        model = self._load_semantic_model()
        if not model:
            # Fallback to lexical matching
            return self.calculate_relevance(claim, evidence)
        
        try:
            from sentence_transformers import util
            claim_embedding = model.encode(claim, convert_to_tensor=True)
            evidence_embedding = model.encode(evidence, convert_to_tensor=True)
            similarity = util.cos_sim(claim_embedding, evidence_embedding)
            return float(similarity[0][0])
        except Exception as e:
            print(f"Semantic similarity calculation failed: {e}")
            return self.calculate_relevance(claim, evidence)

    def extract_temporal_info(self, text: str, source_url: str = "") -> Optional[datetime]:
        """
        Extract date/time information from Vietnamese text and URL.
        
        Args:
            text: Text to extract date from
            source_url: Source URL (may contain date info)
            
        Returns:
            datetime object if date found, else None
        """
        current_year = datetime.now().year
        now = datetime.now()
        
        # Try to extract year from URL patterns
        year_from_url = None
        if source_url:
            # Pattern: /2025/01/20/, -20250120-, 102250120 (baochinhphu.vn format)
            url_patterns = [
                r'/(\d{4})/\d{1,2}/\d{1,2}',  # /2025/01/20/
                r'-(\d{4})\d{4}-',  # -20250120-
                r'(\d{4})(\d{2})(\d{2})',  # 20250120
            ]
            for pattern in url_patterns:
                match = re.search(pattern, source_url)
                if match:
                    try:
                        potential_year = int(match.group(1))
                        # Validate year is reasonable (1990-2050)
                        if 1990 <= potential_year <= 2050:
                            year_from_url = potential_year
                            break
                    except (ValueError, IndexError):
                        continue
        
        # Extract year from text - multiple patterns
        year_in_text = None
        year_patterns = [
            r'năm\s+(\d{4})',  # "năm 2025"
            r'(\d{4})',  # Just "2025" (last resort, must validate)
        ]
        for pattern in year_patterns:
            year_match = re.search(pattern, text.lower())
            if year_match:
                try:
                    potential_year = int(year_match.group(1))
                    # Validate year is reasonable (1990-2050)
                    if 1990 <= potential_year <= 2050:
                        year_in_text = potential_year
                        break
                except (ValueError, IndexError):
                    continue
        
        # Try to find year in context around date mentions
        # Look for patterns like "ngày 20/1 ... 2025" or "2025 ... ngày 20/1"
        context_year = None
        date_with_context = re.search(r'(\d{4}).*?(?:ngày|vào|hôm)\s*\d{1,2}[/-]\d{1,2}|(?:ngày|vào|hôm)\s*\d{1,2}[/-]\d{1,2}.*?(\d{4})', text)
        if date_with_context:
            for group in date_with_context.groups():
                if group:
                    try:
                        potential_year = int(group)
                        if 1990 <= potential_year <= 2050:
                            context_year = potential_year
                            break
                    except (ValueError, TypeError):
                        pass
        
        # Prefer context year (closest to date), then explicit mention, then URL
        best_year_guess = context_year or year_in_text or year_from_url
        
        # Debug logging
        if best_year_guess:
            sources = []
            if context_year: sources.append(f"context:{context_year}")
            if year_in_text: sources.append(f"text:{year_in_text}")
            if year_from_url: sources.append(f"url:{year_from_url}")
            # Uncomment for debugging: print(f"Year inference: {best_year_guess} from {', '.join(sources)}")
        
        # Comprehensive date patterns for Vietnamese content
        date_patterns = [
            # Full dates with year
            r'(?:ngày\s+)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # "7/1/2026" or "ngày 7/1/2026"
            r'(?:ngày\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',  # "ngày 20 tháng 1 năm 2025"
            r'(\d{1,2})\s+tháng\s+(\d{1,2}),?\s+(\d{4})',  # "20 tháng 1, 2025"
            
            # Dates with month name and year
            r'(tháng\s+\d{1,2})\s+năm\s+(\d{4})',  # "tháng 10 năm 2023"
            r'(tháng\s+\d{1,2})[/](\d{4})',  # "tháng 10/2023"
            
            # Dates without explicit year (need inference)
            r'(?:vào|hôm|ngày)\s+(\d{1,2})[/-](\d{1,2})(?:\s|,|$|\))',  # "vào 20/1" or "hôm 20/1"
            r'(?:ngày\s+)?(\d{1,2})[/-](\d{1,2})(?:\s|,|$|\))',  # "ngày 7/1" or "7/1"
            r'(?:ngày\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})(?:\s|,|$)',  # "ngày 20 tháng 1"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    groups = match.groups()
                    
                    if len(groups) == 3:  # Full date with year
                        # Check if it's month-year pattern
                        if "tháng" in groups[0]:  # "tháng X năm YYYY"
                            month_str = groups[0]
                            for vi_month, num in self.months_vi.items():
                                if vi_month in month_str:
                                    month = num
                                    break
                            year = int(groups[1])
                            return datetime(year, month, 1)
                        else:  # Regular day/month/year
                            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                            return datetime(year, month, day)
                            
                    elif len(groups) == 2:
                        if "tháng" in pattern:  # Month + year pattern
                            month_str = groups[0]
                            for vi_month, num in self.months_vi.items():
                                if vi_month in month_str:
                                    month = num
                                    break
                            year = int(groups[1])
                            return datetime(year, month, 1)
                        else:  # Day + month without year - need smart inference
                            day, month = int(groups[0]), int(groups[1])
                            
                            # Use explicit year from text or URL if found
                            if best_year_guess:
                                try:
                                    return datetime(best_year_guess, month, day)
                                except ValueError:
                                    pass
                            
                            # Smart year inference based on temporal distance
                            # Try current year first
                            try:
                                candidate_date = datetime(current_year, month, day)
                                
                                # If date is in the future (more than 7 days ahead), 
                                # it's likely from last year
                                if candidate_date > now + timedelta(days=7):
                                    return datetime(current_year - 1, month, day)
                                else:
                                    return candidate_date
                            except ValueError:
                                # Invalid date, try last year
                                try:
                                    return datetime(current_year - 1, month, day)
                                except ValueError:
                                    pass
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def calculate_temporal_score(self, evidence_date: Optional[datetime], claim: str) -> float:
        """
        Calculate temporal score - higher for more recent evidence.
        For "current"/"now" questions, recent sources get much higher scores.
        
        Args:
            evidence_date: Extracted date from evidence
            claim: Original claim to check if it's about current events
            
        Returns:
            float: Temporal score (0-1)
        """
        if not evidence_date:
            return 0.5  # Neutral if no date found
        
        # Check if claim is about current/now
        is_current_claim = any(keyword in claim.lower() for keyword in 
                               ["hiện nay", "hiện tại", "now", "current", "bây giờ", "hôm nay"])
        
        now = datetime.now()
        age_days = (now - evidence_date).days
        
        if is_current_claim:
            # For current claims, heavily penalize old info
            if age_days < 30:  # Less than 1 month
                return 1.0
            elif age_days < 90:  # 1-3 months
                return 0.8
            elif age_days < 180:  # 3-6 months
                return 0.5
            elif age_days < 365:  # 6-12 months
                return 0.3
            else:  # More than 1 year
                return 0.1
        else:
            # For non-current claims, age matters less
            if age_days < 365:  # Less than 1 year
                return 1.0
            elif age_days < 365 * 2:  # 1-2 years
                return 0.9
            elif age_days < 365 * 3:  # 2-3 years
                return 0.8
            else:
                return 0.6

    def validate_evidence(
        self,
        evidence_text: str,
        source_url: str,
        claim: str,
        entities: Optional[List[str]] = None
    ) -> Dict:
        """
        Comprehensive evidence validation.

        Args:
            evidence_text: Evidence content
            source_url: Source URL
            claim: Original claim
            entities: Optional key entities from claim

        Returns:
            dict: Validation results with scores
        """
        credibility = self.get_source_credibility(source_url)
        language_quality = self.check_language_quality(evidence_text)
        relevance = self.calculate_relevance(claim, evidence_text, entities)
        
        # Extract and score temporal information (pass source_url for better date extraction)
        evidence_date = self.extract_temporal_info(evidence_text, source_url)
        temporal_score = self.calculate_temporal_score(evidence_date, claim)

        # Weighted overall score - temporal score gets higher weight for current claims
        is_current_claim = any(keyword in claim.lower() for keyword in 
                               ["hiện nay", "hiện tại", "now", "current", "bây giờ", "hôm nay"])
        
        if is_current_claim:
            # For current claims, temporal recency is critical
            overall_score = (
                credibility * 0.25 +
                language_quality * 0.15 +
                relevance * 0.3 +
                temporal_score * 0.3  # 30% weight for recency
            )
        else:
            overall_score = (
                credibility * 0.4 +
                language_quality * 0.2 +
                relevance * 0.4
            )

        return {
            "overall_score": overall_score,
            "credibility": credibility,
            "language_quality": language_quality,
            "relevance": relevance,
            "temporal_score": temporal_score,
            "evidence_date": evidence_date.strftime("%Y-%m-%d") if evidence_date else None,
            "is_valid": overall_score >= 0.5,
            "source_url": source_url,
        }

    def filter_low_quality(
        self,
        evidence_list: List[Dict],
        min_score: float = 0.5
    ) -> List[Dict]:
        """
        Filter out low-quality evidence.

        Args:
            evidence_list: List of evidence items with validation scores
            min_score: Minimum quality threshold

        Returns:
            List[Dict]: Filtered high-quality evidence
        """
        return [
            evidence for evidence in evidence_list
            if evidence.get("overall_score", 0) >= min_score
        ]
    
    def sort_by_temporal_relevance(
        self,
        evidence_list: List[Dict],
        claim: str
    ) -> List[Dict]:
        """
        Sort evidence by temporal relevance - most recent first for current claims.
        
        Args:
            evidence_list: List of evidence items with validation scores
            claim: Original claim
            
        Returns:
            List[Dict]: Sorted evidence list
        """
        is_current_claim = any(keyword in claim.lower() for keyword in 
                               ["hiện nay", "hiện tại", "now", "current", "bây giờ", "hôm nay"])
        
        if is_current_claim:
            # For current claims, sort by temporal score (most recent first)
            return sorted(
                evidence_list,
                key=lambda x: (
                    x.get("temporal_score", 0),
                    x.get("overall_score", 0)
                ),
                reverse=True
            )
        else:
            # For other claims, sort by overall score
            return sorted(
                evidence_list,
                key=lambda x: x.get("overall_score", 0),
                reverse=True
            )


validator = EvidenceValidator()


if __name__ == "__main__":
    
    cred1 = validator.get_source_credibility("https://vnexpress.net/article")
    print(f"VNExpress credibility: {cred1}")
    
    cred2 = validator.get_source_credibility("https://unknown-site.com/article")
    print(f"Unknown site credibility: {cred2}")
    
    text_vn = "Việt Nam có dân số hơn 100 triệu người vào năm 2024."
    quality = validator.check_language_quality(text_vn)
    print(f"Vietnamese text quality: {quality}")
    
    validation = validator.validate_evidence(
        evidence_text=text_vn,
        source_url="https://vnexpress.net/article",
        claim="Dân số Việt Nam",
        entities=["Việt Nam", "100 triệu"]
    )
    print(f"Validation result: {validation}")
