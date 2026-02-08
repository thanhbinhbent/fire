from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch


class QueryDeduplicator:
    """
    Detect duplicate queries using sentence embeddings.
    Prevents wasting API calls on semantically similar searches.
    """

    def __init__(self, model_name: str = "dangvantuan/vietnamese-document-embedding", threshold: float = 0.85):
        """
        Initialize deduplicator with Vietnamese document embedding model (8192 token context).

        Args:
            model_name: HuggingFace model for Vietnamese embeddings (default: vietnamese-document-embedding)
            threshold: Cosine similarity threshold (0-1) for duplicates
        """
        self.threshold = threshold
        self.query_history: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        self._model_loaded = False
        self._use_sentence_transformers = True

    def _load_model(self):
        """Load sentence transformer model lazily (faster than PhoBERT)."""
        if self._model_loaded:
            return
        
        try:
            # Try sentence-transformers first (much faster)
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            self._use_sentence_transformers = True
            self._model_loaded = True
            print(f"Loaded Vietnamese embedding model: {self.model_name}")
            return
        except Exception as e:
            print(f"Could not load sentence-transformers: {e}")
        
        # Fallback to simple string matching
        print("Using simple string matching for query deduplication")
        self._model_loaded = False
        self._use_sentence_transformers = False

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get sentence embedding for text using sentence-transformers (faster than PhoBERT).

        Args:
            text: Input text

        Returns:
            np.ndarray: Embedding vector
        """
        self._load_model()
        
        if not self._model_loaded:
            # Simple hash-based fallback
            return np.array([hash(text) % 10000])
        
        try:
            if self._use_sentence_transformers:
                # Much faster than PhoBERT
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding
            else:
                # Should not reach here
                return np.array([hash(text) % 10000])
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.array([hash(text) % 10000])

    def is_duplicate(self, query: str) -> Tuple[bool, float]:
        """
        Check if query is duplicate of any previous query.

        Args:
            query: New search query

        Returns:
            tuple: (is_duplicate, max_similarity)
        """
        if not self.query_history:
            return False, 0.0

        new_embedding = self._get_embedding(query)

        similarities = cosine_similarity(
            [new_embedding],
            self.embeddings
        )[0]

        max_similarity = float(np.max(similarities))

        is_dup = max_similarity >= self.threshold

        return is_dup, max_similarity

    def add_query(self, query: str):
        """
        Add query to history.

        Args:
            query: Search query to add
        """
        embedding = self._get_embedding(query)
        self.query_history.append(query)
        self.embeddings.append(embedding)

    def check_and_add(self, query: str) -> Tuple[bool, float]:
        """
        Check for duplicates and add if unique.

        Args:
            query: New search query

        Returns:
            tuple: (is_duplicate, similarity_score)
        """
        is_dup, similarity = self.is_duplicate(query)

        if not is_dup:
            self.add_query(query)
        else:
            pass

        return is_dup, similarity

    def reset(self):
        """Clear query history."""
        self.query_history = []
        self.embeddings = []

    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            "total_queries": len(self.query_history),
            "unique_queries": len(set(self.query_history)),
            "threshold": self.threshold,
            "model_loaded": self._model_loaded
        }


deduplicator = QueryDeduplicator()


if __name__ == "__main__":
    
    query1 = "COVID-19 vaccine effectiveness"
    query2 = "vaccine COVID-19 effective"
    query3 = "Vietnam population 2024"
    
    is_dup1, sim1 = deduplicator.check_and_add(query1)
    print(f"Query 1 duplicate: {is_dup1}, similarity: {sim1:.3f}")
    
    is_dup2, sim2 = deduplicator.check_and_add(query2)
    print(f"Query 2 duplicate: {is_dup2}, similarity: {sim2:.3f}")
    
    is_dup3, sim3 = deduplicator.check_and_add(query3)
    print(f"Query 3 duplicate: {is_dup3}, similarity: {sim3:.3f}")
    
    stats = deduplicator.get_stats()
    print(f"Stats: {stats}")
