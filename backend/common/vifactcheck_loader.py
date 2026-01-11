# common/vifactcheck_loader.py
"""
Load and process ViFactCheck dataset from HuggingFace.
Dataset: https://huggingface.co/datasets/tranthaihoa/vifactcheck
"""

from datasets import load_dataset
from typing import Dict, List, Optional
import json
from pathlib import Path
from collections import Counter


class ViFactCheckLoader:
    """Load and manage ViFactCheck dataset from HuggingFace."""

    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize loader.

        Args:
            cache_dir: Directory to cache downloaded dataset
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = None

    def load_dataset(self, split: Optional[str] = None):
        """
        Load ViFactCheck dataset from HuggingFace with caching.

        Args:
            split: Dataset split to load ('train', 'validation', 'test', or None for all)

        Returns:
            Dataset object or DatasetDict
        """
        # Check if already loaded in memory
        if self.dataset is not None:
            print(f"✅ Using cached dataset from memory")
            return self.dataset

        print(f"📥 Loading ViFactCheck dataset from HuggingFace...")

        try:
            if split:
                self.dataset = load_dataset(
                    "tranthaihoa/vifactcheck",
                    split=split,
                    cache_dir=str(self.cache_dir)
                )
                print(f"✅ Loaded {len(self.dataset)} samples from '{split}' split")
            else:
                self.dataset = load_dataset(
                    "tranthaihoa/vifactcheck",
                    cache_dir=str(self.cache_dir)
                )
                total = sum(len(self.dataset[key]) for key in self.dataset.keys())
                print(f"✅ Loaded {total} total samples across all splits")

            return self.dataset

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise

    def get_sample(self, index: int = 0, split: str = "train") -> Dict:
        """
        Get a single sample from dataset.

        Args:
            index: Sample index
            split: Dataset split

        Returns:
            Dictionary with sample data
        """
        if self.dataset is None:
            self.load_dataset()

        data = self.dataset[split] if isinstance(self.dataset, dict) else self.dataset
        sample = data[index]

        return {
            "claim": sample.get("claim", sample.get("text", "")),
            "label": sample.get("label", ""),
            "evidence": sample.get("evidence", ""),
            "metadata": {
                "source": sample.get("source", ""),
                "domain": sample.get("domain", ""),
            }
        }

    def convert_to_fire_format(self, output_path: Optional[str] = None, split: str = "test") -> List[Dict]:
        """
        Convert ViFactCheck dataset to FIRE format.

        Args:
            output_path: Path to save converted JSONL file
            split: Dataset split to convert

        Returns:
            List of converted samples
        """
        if self.dataset is None:
            self.load_dataset()

        dataset_split = self.dataset[split] if isinstance(self.dataset, dict) else self.dataset
        print(f"🔄 Converting {len(dataset_split)} samples to FIRE format...")

        converted = []
        for sample in dataset_split:
            # Map ViFactCheck labels to FIRE format
            label_map = {
                'supported': 'True',
                'refuted': 'False',
                'nei': 'Uncertain',
                'not enough info': 'Uncertain',
            }

            fire_sample = {
                'claim': sample.get('claim', sample.get('text', '')),
                'label': label_map.get(sample.get('label', '').lower(), sample.get('label', '')),
                'evidence': sample.get('evidence', ''),
                'metadata': {
                    'source': sample.get('source', ''),
                    'domain': sample.get('domain', ''),
                }
            }
            converted.append(fire_sample)

        # Save to JSONL if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in converted:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            print(f"✅ Converted {len(converted)} samples to {output_path}")

        return converted

    def get_statistics(self, split: Optional[str] = None) -> Dict:
        """
        Get dataset statistics.

        Args:
            split: Dataset split to analyze (None for all)

        Returns:
            Dictionary with statistics
        """
        if self.dataset is None:
            self.load_dataset()

        if split:
            data = self.dataset[split] if isinstance(self.dataset, dict) else self.dataset

            # Calculate statistics
            labels = [item.get('label', '') for item in data]
            claims = [item.get('claim', item.get('text', '')) for item in data]

            label_counts = Counter(labels)

            stats = {
                'split': split,
                'total_samples': len(data),
                'label_distribution': dict(label_counts),
                'avg_claim_length': sum(len(c) for c in claims) / len(claims) if claims else 0,
            }
        else:
            stats = {}
            for split_name in self.dataset.keys():
                stats[split_name] = self.get_statistics(split_name)

        return stats

    def print_sample_claims(self, n: int = 5, split: str = "train"):
        """Print sample claims for inspection."""
        if self.dataset is None:
            self.load_dataset()

        data = self.dataset[split] if isinstance(self.dataset, dict) else self.dataset

        print(f"\n📋 Sample claims from '{split}' split:")
        print("=" * 80)
        for i in range(min(n, len(data))):
            sample = data[i]
            claim = sample.get('claim', sample.get('text', 'N/A'))
            label = sample.get('label', 'N/A')
            print(f"\n{i+1}. Label: {label}")
            print(f"   Claim: {claim[:150]}...")
        print("=" * 80)


# Create global instance for easy import
loader = ViFactCheckLoader()


# Usage example
if __name__ == "__main__":
    # Load dataset
    loader = ViFactCheckLoader()
    dataset = loader.load_dataset()

    # Print statistics
    stats = loader.get_statistics()
    print("\n📊 Dataset Statistics:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Print sample claims
    loader.print_sample_claims(n=3)

    # Convert to FIRE format
    loader.convert_to_fire_format(
        output_path="datasets/vifactcheck/data.jsonl",
        split="test"
    )
