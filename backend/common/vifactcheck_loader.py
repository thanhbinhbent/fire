"""
Load and process ViFactCheck dataset from local Parquet files.
Dataset: https://huggingface.co/datasets/tranthaihoa/vifactcheck
"""

import pandas as pd
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
from collections import Counter


class ViFactCheckLoader:
    """Load and manage ViFactCheck dataset from local Parquet files."""

    def __init__(self, data_dir: str = "./data/vifactcheck"):
        """
        Initialize loader.

        Args:
            data_dir: Directory containing the git cloned parquet files
        """
        self.data_dir = Path(data_dir)
        self.dataset: Dict[str, pd.DataFrame] = {}

    def load_dataset(self, split: Optional[str] = None):
        """
        Load ViFactCheck dataset from local parquet files.

        Args:
            split: Dataset split to load ('train', 'validation', 'test', or None for all)
        """
        # Mapping các file tương ứng (Giả sử cấu trúc file sau khi git clone)
        # Nếu file của bạn có tên khác (vd: train-00000-of-00001.parquet), hãy điều chỉnh ở đây
        root_path = Path(r"G:/hcmus/khaithacdulieu/vifactcheck/data")
        file_mapping = {
            "train": root_path / "train.parquet",
            "test": root_path / "test.parquet",
            "dev": root_path / "dev.parquet",
        }

        splits_to_load = [split] if split else file_mapping.keys()

        try:
            for s in splits_to_load:
                file_path = file_mapping[s]
                print(f"Loading '{s}' split from {file_path}...")
                # Tìm file có chứa tên split nếu không tìm thấy file chính xác
                if not file_path.exists():
                    files = list(self.data_dir.glob(f"*{s}*.parquet"))
                    if files:
                        file_path = files[0]
                    else:
                        print(f"Warning: No parquet file found for split '{s}' at {self.data_dir}")
                        continue
                df = pd.read_parquet(file_path)
                
                rename_mapping = {
                    'Statement': 'claim', 
                    'labels': 'label', 
                    'Evidence': 'evidence',
                    'Url': 'source',
                    'Topic': 'domain'
                }                                
                
                df.rename(columns=rename_mapping, inplace=True)
                    
                self.dataset[s] = df
                print(f"Loaded {len(self.dataset[s])} samples from local '{s}' split. Columns available: {self.dataset[s].columns.tolist()}")
            print(self.dataset)
            return self.dataset

        except Exception as e:
            print(f"Error loading local dataset: {e}")
            raise

    def get_sample(self, index: int = 0, split: str = "train") -> Dict:
        """Get a single sample from dataset."""
        if split not in self.dataset:
            self.load_dataset(split=split)

        df = self.dataset[split]
        sample = df.iloc[index].to_dict()

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
        """Convert ViFactCheck dataset to FIRE format."""
        if split not in self.dataset:
            self.load_dataset(split=split)

        df_split = self.dataset[split]
        print(f"Converting {len(df_split)} samples to FIRE format...")

        converted = []
        # Chuyển DataFrame sang list dict để loop
        samples = df_split.to_dict('records')

        for sample in samples:
            label_value = sample.get('label', '')
            
            # Map nhãn tương tự code cũ của bạn
            if isinstance(label_value, (int, float)):
                label_map = {0: 'True', 1: 'False', 2: 'Not Enough Info'}
                # Ép kiểu int nếu là float
                key = int(label_value)
            else:
                label_map = {
                    'support': 'True', 'supported': 'True',
                    'refute': 'False', 'refuted': 'False',
                    'nei': 'Not Enough Info', 'not enough info': 'Not Enough Info',
                    'not enough information': 'Not Enough Info',
                }
                key = str(label_value).lower()
            
            fire_sample = {
                'claim': sample.get('claim', sample.get('text', '')),
                'label': label_map.get(key, 'Not Enough Info'),
                'evidence': sample.get('evidence', ''),
                'metadata': {
                    'source': sample.get('source', ''),
                    'domain': sample.get('domain', ''),
                    'original_label': label_value,
                }
            }
            converted.append(fire_sample)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in converted:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            print(f"Converted {len(converted)} samples to {output_path}")

        return converted

    def get_statistics(self, split: Optional[str] = None) -> Dict:
        """Get dataset statistics."""
        if not self.dataset:
            self.load_dataset()

        if split:
            if split not in self.dataset: return {}
            df = self.dataset[split]

            label_counts = df['label'].value_counts().to_dict()
            # Xử lý lấy text/claim
            claim_col = 'claim' if 'claim' in df.columns else 'text'
            
            stats = {
                'split': split,
                'total_samples': len(df),
                'label_distribution': label_counts,
                'avg_claim_length': df[claim_col].str.len().mean() if claim_col in df.columns else 0,
            }
        else:
            stats = {split_name: self.get_statistics(split_name) for split_name in self.dataset.keys()}

        return stats

    def print_sample_claims(self, n: int = 5, split: str = "train"):
        """Print sample claims for inspection."""
        if split not in self.dataset:
            self.load_dataset(split=split)

        df = self.dataset[split]
        print(f"\nSample claims from '{split}' split:")
        print("=" * 80)
        
        claim_col = 'claim' if 'claim' in df.columns else 'text'
        for i in range(min(n, len(df))):
            row = df.iloc[i]
            print(f"\n{i+1}. Label: {row.get('label', 'N/A')}")
            print(f"   Claim: {str(row.get(claim_col, 'N/A'))[:150]}...")
        print("=" * 80)


# --- Khởi chạy ---
if __name__ == "__main__":
    # ĐƯỜNG DẪN: Thay đổi thành folder chứa các file .parquet bạn đã tải về
    loader = ViFactCheckLoader(data_dir="./vifactcheck") 
    
    # Load toàn bộ
    loader.load_dataset()

    stats = loader.get_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    loader.print_sample_claims(n=3, split="train")

    loader.convert_to_fire_format(
        output_path="datasets/vifactcheck/data.jsonl",
        split="test"
    )