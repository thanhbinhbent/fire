import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
import threading


class FactCheckDB:
    """Simple SQLite database for fact-checking cache and results."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = "dev.db"):
        """Singleton pattern to ensure one database connection."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "dev.db"):
        if hasattr(self, '_initialized'):
            return
        
        self.db_path = db_path
        self._initialized = True
        self._create_tables()

    def _get_connection(self):
        """Get a new database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_text TEXT NOT NULL UNIQUE,
                claim_hash TEXT,
                domain TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                verdict TEXT,
                confidence REAL,
                reasoning TEXT,
                model_used TEXT,
                iterations INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (claim_id) REFERENCES claims(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verification_id INTEGER NOT NULL,
                evidence_text TEXT NOT NULL,
                source_url TEXT,
                source_domain TEXT,
                relevance_score REAL,
                retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (verification_id) REFERENCES verifications(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL UNIQUE,
                results TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_claims_text ON claims(claim_text)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_query ON search_cache(query)")

        conn.commit()
        conn.close()
        print("Database initialized successfully")

    def cache_search(self, query: str, results: str):
        """Cache search results to avoid redundant API calls."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT OR REPLACE INTO search_cache (query, results, cached_at) VALUES (?, ?, ?)",
                (query, results, datetime.now())
            )
            conn.commit()
        except Exception as e:
            print(f"Cache save error: {e}")
        finally:
            conn.close()

    def get_cached_search(self, query: str) -> Optional[str]:
        """Retrieve cached search results."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT results FROM search_cache WHERE query = ?",
                (query,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Cache retrieve error: {e}")
            return None
        finally:
            conn.close()

    def save_verification(
        self,
        claim: str,
        verdict: str,
        confidence: float,
        reasoning: str,
        model: str,
        searches: List[Dict]
    ) -> int:
        """Save complete verification result."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT OR IGNORE INTO claims (claim_text, claim_hash) VALUES (?, ?)",
                (claim, str(hash(claim)))
            )
            cursor.execute("SELECT id FROM claims WHERE claim_text = ?", (claim,))
            claim_id = cursor.fetchone()[0]

            cursor.execute("""
                INSERT INTO verifications
                (claim_id, verdict, confidence, reasoning, model_used, iterations)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (claim_id, verdict, confidence, reasoning, model, len(searches)))

            verification_id = cursor.lastrowid

            for search in searches:
                for result in search.get('results', []):
                    cursor.execute("""
                        INSERT INTO evidence
                        (verification_id, evidence_text, source_url, relevance_score)
                        VALUES (?, ?, ?, ?)
                    """, (
                        verification_id,
                        result.get('snippet', '')[:500],
                        result.get('link', ''),
                        result.get('validation', {}).get('relevance', 0.5)
                    ))

            conn.commit()
            return verification_id
        except Exception as e:
            print(f"Verification save error: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()

    def get_recent_verifications(self, limit: int = 10) -> List[Dict]:
        """Get recent verifications."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT c.claim_text, v.verdict, v.confidence, v.created_at
                FROM verifications v
                JOIN claims c ON v.claim_id = c.id
                ORDER BY v.created_at DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "claim": row[0],
                    "verdict": row[1],
                    "confidence": row[2],
                    "created_at": row[3]
                })
            return results
        except Exception as e:
            print(f"Retrieve error: {e}")
            return []
        finally:
            conn.close()


db = FactCheckDB()


if __name__ == "__main__":
    print("Testing database...")
    
    db.cache_search("test query", "test results")
    cached = db.get_cached_search("test query")
    print(f"Cached result: {cached}")
    
    verification_id = db.save_verification(
        claim="Test claim",
        verdict="SUPPORTS",
        confidence=0.85,
        reasoning="Test reasoning",
        model="gpt-4o-mini",
        searches=[{"results": [{"snippet": "test", "link": "http://test.com"}]}]
    )
    print(f"Verification ID: {verification_id}")
    
    recent = db.get_recent_verifications(5)
    print(f"Recent verifications: {len(recent)}")
