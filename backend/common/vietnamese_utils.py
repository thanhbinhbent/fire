"""Vietnamese text preprocessing utilities."""

import re
import unicodedata
from typing import List, Dict, Optional
from underthesea import sent_tokenize, word_tokenize, ner


class VietnamesePreprocessor:
    """Vietnamese text normalization and tokenization."""

    def __init__(self):
        self.abbreviations = {
            "TPHCM": "Thành phố Hồ Chí Minh",
            "TP.HCM": "Thành phố Hồ Chí Minh",
            "HN": "Hà Nội",
            "UBND": "Ủy ban Nhân dân",
            "VN": "Việt Nam",
            "TP": "Thành phố",
            "PGS.TS": "Phó Giáo sư Tiến sĩ",
            "TS": "Tiến sĩ",
            "ThS": "Thạc sĩ",
            "GS": "Giáo sư",
        }

    def normalize_text(self, text: str) -> str:
        """
        Normalize Vietnamese text:
        - Convert to NFC Unicode normalization
        - Expand abbreviations
        - Remove extra whitespace
        """
        text = unicodedata.normalize('NFC', text)

        for abbr, full in self.abbreviations.items():
            text = re.sub(rf'\b{re.escape(abbr)}\b', full, text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """Sentence tokenization for Vietnamese using underthesea."""
        try:
            return sent_tokenize(text)
        except Exception as e:
            print(f"⚠️ Sentence tokenization error: {e}")
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        """Word tokenization for Vietnamese."""
        try:
            return word_tokenize(text, format="text").split()
        except Exception as e:
            print(f"⚠️ Word tokenization error: {e}")
            return text.split()

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from Vietnamese text.
        Uses underthesea NER model.
        """
        try:
            entities_raw = ner(text)
            results = []

            for entity in entities_raw:
                if len(entity) >= 4 and entity[3] != 'O':
                    results.append({
                        "text": entity[0],
                        "type": entity[3],
                        "pos": entity[1] if len(entity) > 1 else "",
                    })

            return results
        except Exception as e:
            print(f"⚠️ Entity extraction error: {e}")
            return []

    def preprocess_claim(self, claim: str) -> Dict:
        """
        Complete preprocessing pipeline for a claim.
        Returns dict with various representations.
        """
        normalized = self.normalize_text(claim)
        sentences = self.tokenize_sentences(normalized)
        tokens = self.tokenize_words(normalized)
        entities = self.extract_entities(normalized)

        return {
            "original": claim,
            "normalized": normalized,
            "sentences": sentences,
            "tokens": tokens,
            "entities": entities,
            "token_count": len(tokens)
        }


preprocessor = VietnamesePreprocessor()


if __name__ == "__main__":
    test_claim = "TP.HCM có hơn 10 triệu dân vào năm 2024"
    result = preprocessor.preprocess_claim(test_claim)
    
    print("Preprocessing Test:")
    print(f"Original: {result['original']}")
    print(f"Normalized: {result['normalized']}")
    print(f"Tokens: {result['tokens'][:10]}")
    print(f"Entities: {result['entities']}")
    print(f"Token count: {result['token_count']}")
