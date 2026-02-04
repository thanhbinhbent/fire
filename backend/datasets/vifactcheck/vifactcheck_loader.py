"""
Load and process ViFactCheck dataset from HuggingFace.
Dataset: https://huggingface.co/datasets/tranthaihoa/vifactcheck
"""

from datasets import load_dataset
from typing import Dict, List, Optional
import json
import argparse
from pathlib import Path
from collections import Counter


# Label mappings for compatibility with other datasets
LABEL_MAP_INT = {
    0: 'true',      # SUPPORTED
    1: 'false',     # REFUTED
    2: 'nei',       # NEI (Not Enough Info)
}

LABEL_MAP_STR = {
    'support': 'true',
    'supported': 'true',
    'refute': 'false',
    'refuted': 'false',
    'nei': 'nei',
    'not enough info': 'nei',
    'not enough information': 'nei',
}


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
        if self.dataset is not None:
            return self.dataset


        try:
            if split:
                self.dataset = load_dataset(
                    "tranthaihoa/vifactcheck",
                    split=split,
                    cache_dir=str(self.cache_dir)
                )
                print(f"Loaded {len(self.dataset)} samples from '{split}' split")
            else:
                self.dataset = load_dataset(
                    "tranthaihoa/vifactcheck",
                    cache_dir=str(self.cache_dir)
                )
                total = sum(len(self.dataset[key]) for key in self.dataset.keys())
                print(f"Loaded {total} total samples across all splits")

            return self.dataset

        except Exception as e:
            print(f"Error loading dataset: {e}")
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
            "claim": sample.get("Statement", sample.get("claim", sample.get("text", ""))),
            "label": sample.get("labels", sample.get("label", "")),
            "evidence": sample.get("Evidence", sample.get("evidence", "")),
            "metadata": {
                "source": sample.get("Url", sample.get("source", "")),
                "domain": sample.get("Topic", sample.get("domain", "")),
            }
        }

    def convert_to_fire_format(self, output_path: Optional[str] = None, split: str = "test", include_nei: bool = False) -> List[Dict]:
        """
        Convert ViFactCheck dataset to FIRE format.

        Args:
            output_path: Path to save converted JSONL file
            split: Dataset split to convert
            include_nei: Whether to include NEI (Not Enough Info) samples

        Returns:
            List of converted samples
        """
        if self.dataset is None:
            self.load_dataset()

        dataset_split = self.dataset[split] if isinstance(self.dataset, dict) else self.dataset
        print(f"Converting {len(dataset_split)} samples to FIRE format...")

        converted = []
        skipped_nei = 0
        label_stats = Counter()
        
        for sample in dataset_split:
            # Get claim text - ViFactCheck uses 'Statement' field
            claim = sample.get('Statement', sample.get('claim', sample.get('text', '')))
            if not claim or not claim.strip():
                continue
            
            # ViFactCheck uses 'labels' field (integer: 0=supported, 1=refuted, 2=NEI)
            label_value = sample.get('labels', sample.get('label', ''))
            
            # Map labels to lowercase 'true'/'false' for compatibility with other datasets
            if isinstance(label_value, int):
                mapped_label = LABEL_MAP_INT.get(label_value, 'nei')
            else:
                mapped_label = LABEL_MAP_STR.get(
                    str(label_value).lower(), 
                    'nei'
                )
            
            # Skip NEI samples if not included
            if mapped_label == 'nei':
                if not include_nei:
                    skipped_nei += 1
                    continue
            
            label_stats[mapped_label] += 1
            
            # Use minimal format matching other datasets (only claim and label)
            fire_sample = {
                'claim': claim.strip(),
                'label': mapped_label,
            }
            converted.append(fire_sample)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in converted:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            print(f"Converted {len(converted)} samples to {output_path}")
            if skipped_nei > 0:
                print(f"Skipped {skipped_nei} NEI samples")
            print(f"Label distribution: {dict(label_stats)}")

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

        print(f"\nSample claims from '{split}' split:")
        print("=" * 80)
        for i in range(min(n, len(data))):
            sample = data[i]
            claim = sample.get('claim', sample.get('text', 'N/A'))
            label = sample.get('label', 'N/A')
            print(f"\n{i+1}. Label: {label}")
            print(f"   Claim: {claim[:150]}...")
        print("=" * 80)

    def download_and_convert(
        self,
        output_dir: str = "datasets/vifactcheck",
        split: str = "all",
        include_nei: bool = False,
        verbose: bool = True
    ) -> bool:
        """
        Download ViFactCheck from HuggingFace and convert to FIRE format.
        
        Args:
            output_dir: Directory to save the converted dataset
            split: Dataset split to use ('train', 'validation', 'test', or 'all')
            include_nei: Whether to include NEI (Not Enough Info) samples
            verbose: Print progress information
        
        Returns:
            True if successful, False otherwise
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print("=" * 70)
            print("ViFactCheck Dataset Downloader for FIRE Evaluation")
            print("=" * 70)
            print(f"Source: tranthaihoa/vifactcheck (HuggingFace)")
            print(f"Output: {output_path / 'data.jsonl'}")
            print(f"Split: {split}")
            print(f"Include NEI: {include_nei}")
            print("=" * 70)
        
        # Download dataset
        if verbose:
            print("\nDownloading dataset from HuggingFace...")
        
        try:
            if split == "all":
                dataset = load_dataset("tranthaihoa/vifactcheck")
                all_samples = []
                for split_name in dataset.keys():
                    all_samples.extend(list(dataset[split_name]))
                if verbose:
                    print(f"Downloaded {len(all_samples)} samples from all splits")
            else:
                dataset = load_dataset("tranthaihoa/vifactcheck", split=split)
                all_samples = list(dataset)
                if verbose:
                    print(f"Downloaded {len(all_samples)} samples from '{split}' split")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
        
        # Convert samples
        if verbose:
            print("\nConverting to FIRE format...")
        
        converted = []
        skipped_nei = 0
        label_stats = Counter()
        
        for sample in all_samples:
            # Get claim text - ViFactCheck uses 'Statement' field
            claim = sample.get('Statement', sample.get('claim', sample.get('text', '')))
            if not claim or not claim.strip():
                continue
            
            # Map label - ViFactCheck uses 'labels' field (integer: 0=supported, 1=refuted, 2=NEI)
            label_value = sample.get('labels', sample.get('label', ''))
            
            if isinstance(label_value, int):
                mapped_label = LABEL_MAP_INT.get(label_value, 'nei')
            else:
                mapped_label = LABEL_MAP_STR.get(str(label_value).lower(), 'nei')
            
            # Skip NEI if not included
            if mapped_label == 'nei':
                if not include_nei:
                    skipped_nei += 1
                    continue
            
            label_stats[mapped_label] += 1
            
            # Create FIRE-compatible sample (minimal format)
            fire_sample = {
                'claim': claim.strip(),
                'label': mapped_label,
            }
            converted.append(fire_sample)
        
        # Save to JSONL file
        output_file = output_path / "data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in converted:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        if verbose:
            print(f"\nConversion complete!")
            print(f"\nStatistics:")
            print(f"  Total samples: {len(converted)}")
            if skipped_nei > 0:
                print(f"  Skipped NEI: {skipped_nei}")
            print(f"\n  Label distribution:")
            for label, count in sorted(label_stats.items()):
                pct = count / len(converted) * 100 if converted else 0
                print(f"    '{label}': {count} ({pct:.1f}%)")
            
            # Calculate average claim length
            avg_len = sum(len(s['claim']) for s in converted) / len(converted) if converted else 0
            print(f"\n  Average claim length: {avg_len:.1f} characters")
            
            print(f"\nSaved to: {output_file}")
            print("\n" + "=" * 70)
            print("Dataset ready for FIRE evaluation!")
            print(f"Run: python run_fire.py --dataset vifactcheck --model <model>")
            print("=" * 70)
        
        return True


loader = ViFactCheckLoader()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and convert ViFactCheck dataset for FIRE evaluation"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="datasets/vifactcheck",
        help="Output directory for converted dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
        help="Dataset split to download"
    )
    parser.add_argument(
        "--include-nei",
        action="store_true",
        help="Include NEI (Not Enough Info) samples"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of sample claims to display"
    )
    
    args = parser.parse_args()
    
    loader = ViFactCheckLoader()
    
    # Download and convert
    success = loader.download_and_convert(
        output_dir=args.output_dir,
        split=args.split,
        include_nei=args.include_nei
    )
    
    # Show statistics if requested
    if success and args.stats:
        dataset = loader.load_dataset()
        stats = loader.get_statistics()
        print("\nDetailed Dataset Statistics:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # Show sample claims if requested
    if success and args.samples > 0:
        dataset = loader.load_dataset()
        loader.print_sample_claims(n=args.samples)
