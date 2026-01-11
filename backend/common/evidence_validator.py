from typing import List, Dict, Optional
import re
from urllib.parse import urlparse


class EvidenceValidator:
    """
    Validate evidence quality for Vietnamese fact-checking.
    Scores based on:
    1. Source credibility (trusted Vietnamese news sites)
    2. Language quality (proper Vietnamese)
    3. Relevance to claim
    """

    def __init__(self):
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
