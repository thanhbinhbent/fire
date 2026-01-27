from typing import List, Dict, Optional
import re
from urllib.parse import urlparse
from datetime import datetime, timedelta
import calendar


class EvidenceValidator:
    def __init__(self):
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
        try:
            domain = urlparse(url).netloc.lower().replace("www.", "")

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
        if not text:
            return 0.0

        score = 0.5
        vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        if any(char in text.lower() for char in vietnamese_chars):
            score += 0.2
        if any(p in text for p in ".!?"):
            score += 0.15
        if len(text) > 50:
            score += 0.15
        return min(score, 1.0)

    def calculate_relevance(self, claim: str, evidence: str, entities: Optional[List[str]] = None) -> float:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()

        claim_words = set(claim_lower.split())
        evidence_words = set(evidence_lower.split())
        
        stopwords = {'là', 'của', 'và', 'có', 'được', 'trong', 'để', 'cho', 'với', 'từ', 'theo', 'trên'}
        claim_words -= stopwords
        evidence_words -= stopwords
        
        overlap = len(claim_words & evidence_words)
        overlap_ratio = overlap / len(claim_words) if claim_words else 0

        entity_score = 0.0
        if entities:
            entity_matches = sum(1 for entity in entities if entity.lower() in evidence_lower)
            entity_score = entity_matches / len(entities) if entities else 0

        relevance = (overlap_ratio * 0.6) + (entity_score * 0.4) if entities else overlap_ratio
        return min(relevance, 1.0)

    def _load_semantic_model(self):
        if self._semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except Exception:
                self._semantic_model = False
        return self._semantic_model

    def calculate_semantic_similarity(self, claim: str, evidence: str) -> float:
        model = self._load_semantic_model()
        if not model:
            return self.calculate_relevance(claim, evidence)
        
        try:
            from sentence_transformers import util
            claim_embedding = model.encode(claim, convert_to_tensor=True)
            evidence_embedding = model.encode(evidence, convert_to_tensor=True)
            similarity = util.cos_sim(claim_embedding, evidence_embedding)
            return float(similarity[0][0])
        except Exception:
            return self.calculate_relevance(claim, evidence)

    def extract_temporal_info(self, text: str, source_url: str = "") -> Optional[datetime]:
        current_year = datetime.now().year
        now = datetime.now()
        
        year_from_url = None
        if source_url:
            url_patterns = [
                r'/(\d{4})/\d{1,2}/\d{1,2}',
                r'-(\d{4})\d{4}-',
                r'(\d{4})(\d{2})(\d{2})',
            ]
            for pattern in url_patterns:
                match = re.search(pattern, source_url)
                if match:
                    try:
                        potential_year = int(match.group(1))
                        if 1990 <= potential_year <= 2050:
                            year_from_url = potential_year
                            break
                    except (ValueError, IndexError):
                        continue
        
        year_in_text = None
        year_patterns = [r'năm\s+(\d{4})', r'(\d{4})']
        for pattern in year_patterns:
            year_match = re.search(pattern, text.lower())
            if year_match:
                try:
                    potential_year = int(year_match.group(1))
                    if 1990 <= potential_year <= 2050:
                        year_in_text = potential_year
                        break
                except (ValueError, IndexError):
                    continue
        
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
        
        best_year_guess = context_year or year_in_text or year_from_url
        date_patterns = [
            r'(?:ngày\s+)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            r'(?:ngày\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
            r'(\d{1,2})\s+tháng\s+(\d{1,2}),?\s+(\d{4})',
            r'(tháng\s+\d{1,2})\s+năm\s+(\d{4})',
            r'(tháng\s+\d{1,2})[/](\d{4})',
            r'(?:vào|hôm|ngày)\s+(\d{1,2})[/-](\d{1,2})(?:\s|,|$|\))',
            r'(?:ngày\s+)?(\d{1,2})[/-](\d{1,2})(?:\s|,|$|\))',
            r'(?:ngày\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})(?:\s|,|$)',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    groups = match.groups()
                    
                    if len(groups) == 3:
                        if "tháng" in groups[0]:
                            month_str = groups[0]
                            for vi_month, num in self.months_vi.items():
                                if vi_month in month_str:
                                    month = num
                                    break
                            year = int(groups[1])
                            return datetime(year, month, 1)
                        else:
                            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                            return datetime(year, month, day)
                            
                    elif len(groups) == 2:
                        if "tháng" in pattern:
                            month_str = groups[0]
                            for vi_month, num in self.months_vi.items():
                                if vi_month in month_str:
                                    month = num
                                    break
                            year = int(groups[1])
                            return datetime(year, month, 1)
                        else:
                            day, month = int(groups[0]), int(groups[1])
                            
                            if best_year_guess:
                                try:
                                    return datetime(best_year_guess, month, day)
                                except ValueError:
                                    pass
                            
                            try:
                                candidate_date = datetime(current_year, month, day)
                                if candidate_date > now + timedelta(days=7):
                                    return datetime(current_year - 1, month, day)
                                else:
                                    return candidate_date
                            except ValueError:
                                try:
                                    return datetime(current_year - 1, month, day)
                                except ValueError:
                                    pass
                except (ValueError, IndexError):
                    continue
        return None
    
    def calculate_temporal_score(self, evidence_date: Optional[datetime], claim: str) -> float:
        if not evidence_date:
            return 0.5
        
        is_current_claim = any(keyword in claim.lower() for keyword in 
                               ["hiện nay", "hiện tại", "now", "current", "bây giờ", "hôm nay"])
        
        now = datetime.now()
        age_days = (now - evidence_date).days
        
        if is_current_claim:
            if age_days < 30:
                return 1.0
            elif age_days < 90:
                return 0.8
            elif age_days < 180:
                return 0.5
            elif age_days < 365:
                return 0.3
            else:
                return 0.1
        else:
            if age_days < 365:
                return 1.0
            elif age_days < 365 * 2:
                return 0.9
            elif age_days < 365 * 3:
                return 0.8
            else:
                return 0.6

    def validate_evidence(self, evidence_text: str, source_url: str, claim: str, entities: Optional[List[str]] = None) -> Dict:
        credibility = self.get_source_credibility(source_url)
        language_quality = self.check_language_quality(evidence_text)
        relevance = self.calculate_relevance(claim, evidence_text, entities)
        
        evidence_date = self.extract_temporal_info(evidence_text, source_url)
        temporal_score = self.calculate_temporal_score(evidence_date, claim)

        is_current_claim = any(keyword in claim.lower() for keyword in 
                               ["hiện nay", "hiện tại", "now", "current", "bây giờ", "hôm nay"])
        
        if is_current_claim:
            overall_score = credibility * 0.25 + language_quality * 0.15 + relevance * 0.3 + temporal_score * 0.3
        else:
            overall_score = credibility * 0.4 + language_quality * 0.2 + relevance * 0.4

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

    def filter_low_quality(self, evidence_list: List[Dict], min_score: float = 0.5) -> List[Dict]:
        return [e for e in evidence_list if e.get("overall_score", 0) >= min_score]
    
    def sort_by_temporal_relevance(self, evidence_list: List[Dict], claim: str) -> List[Dict]:
        is_current_claim = any(keyword in claim.lower() for keyword in 
                               ["hiện nay", "hiện tại", "now", "current", "bây giờ", "hôm nay"])
        
        if is_current_claim:
            return sorted(evidence_list, key=lambda x: (x.get("temporal_score", 0), x.get("overall_score", 0)), reverse=True)
        else:
            return sorted(evidence_list, key=lambda x: x.get("overall_score", 0), reverse=True)


validator = EvidenceValidator()
