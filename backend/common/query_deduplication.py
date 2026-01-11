# common/query_deduplication.py
"""
Embedding-based query deduplication to avoid redundant searches.
Uses PhoBERT embeddings and cosine similarity.
"""

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

    def __init__(self, model_name: str = "vinai/phobert-base-v2", threshold: float = 0.85):
        """
        Initialize deduplicator with PhoBERT model.

        Args:
            model_name: HuggingFace model for Vietnamese embeddings
            threshold: Cosine similarity threshold (0-1) for duplicates
        """
        self.threshold = threshold
        self.query_history: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        # Lazy loading - only load model when first used
        self._model_loaded = False

    def _load_model(self):
        """Load PhoBERT model lazily."""
        if self._model_loaded:
            return
        
        try:
            print(f"📥 Loading embedding model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ Model loaded on GPU")
            else:
                print("✅ Model loaded on CPU")
            
            self._model_loaded = True
        except Exception as e:
            print(f"⚠️ Could not load PhoBERT model: {e}")
            print("⚠️ Falling back to simple string matching")
            self._model_loaded = False

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get sentence embedding for text.

        Args:
            text: Input text

        Returns:
            np.ndarray: Embedding vector
        """
        self._load_model()
        
        if not self._model_loaded:
            # Fallback: use simple hash-based embedding
            return np.array([hash(text) % 10000])
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            # Move to same device as model
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            return embedding
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
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

        # Get embedding for new query
        new_embedding = self._get_embedding(query)

        # Calculate similarities with all previous queries
        similarities = cosine_similarity(
            [new_embedding],
            self.embeddings
        )[0]

        max_similarity = float(np.max(similarities))

        # Check if any similarity exceeds threshold
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
            print(f"✅ Added new query: {query[:50]}...")
        else:
            print(f"⚠️ Skipping duplicate query (similarity={similarity:.2f}): {query[:50]}...")

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


# Global instance
deduplicator = QueryDeduplicator()


# Testing
if __name__ == "__main__":
    print("Testing query deduplicator...")
    
    # Test queries
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
