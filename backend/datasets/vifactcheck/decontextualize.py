"""
Decontextualization module for Vietnamese fact-checking claims.
Supports both cloud LLMs (OpenAI, Anthropic) and local LLMs (Ollama).
"""

import os
import json
import argparse
from typing import Literal, Optional
from openai import OpenAI
from anthropic import Anthropic
from tqdm import tqdm

# Add import for ViFactCheckLoader
from vifactcheck_loader import ViFactCheckLoader


class DecontextualizeConfig:
    """Configuration for decontextualization"""
    def __init__(
        self,
        provider: Literal["openai", "anthropic", "ollama"] = "ollama",
        model: str = "gemma3:4b",
        temperature: float = 0.3,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        # Set base URL based on provider
        if provider == "ollama":
            self.base_url = base_url or "http://localhost:8001/v1"
        else:
            self.base_url = base_url


class Decontextualizer:
    """Decontextualizes Vietnamese claims using LLMs"""
    
    def __init__(self, config: DecontextualizeConfig):
        self.config = config
        self.client = self._init_client()
    
    def _init_client(self):
        """Initialize LLM client based on provider"""
        if self.config.provider == "openai":
            return OpenAI(api_key=self.config.api_key)
        elif self.config.provider == "anthropic":
            return Anthropic(api_key=self.config.api_key)
        elif self.config.provider == "ollama":
            # Ollama uses OpenAI-compatible API
            return OpenAI(
                base_url=self.config.base_url,
                api_key="ollama"  # Ollama doesn't require real API key
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _create_prompt(self, claim: str, context: str) -> str:
        """Create Vietnamese decontextualization prompt"""
        return f"""NHIỆM VỤ: Viết lại 'Tuyên bố' thành một câu độc lập, có thể kiểm chứng được bằng cách sử dụng 'Ngữ cảnh' được cung cấp.

QUY TẮC:
1. Thay thế đại từ (anh ấy, cô ấy, họ, nó, người này...) và các cụm từ chỉ định mơ hồ (hành động này, việc này, điều đó, tại đây, lúc đó...) bằng danh từ/thực thể/mô tả cụ thể từ ngữ cảnh.
2. BẮT BUỘC bao gồm ngày tháng cụ thể nếu có trong ngữ cảnh (ví dụ: "tháng 2-2023", "ngày 21-3", "ngày 24-3")
3. Nếu "hành động này" đề cập đến một sự việc, hãy tóm tắt ngắn gọn sự việc đó (ví dụ: "Việc bác sĩ A hiến máu" thay vì "Hành động này").
4. KHÔNG thay đổi ý nghĩa hoặc giá trị chân lý của tuyên bố
5. Nếu thiếu thông tin trong ngữ cảnh, giữ nguyên và KHÔNG suy đoán
6. Giữ nguyên số liệu, tên riêng, tổ chức; KHÔNG thêm thực thể mới
7. Chỉ xuất ra 1 câu tuyên bố đã được viết lại, KHÔNG thêm giải thích hay dấu ngoặc kép

VÍ DỤ:
- Ngữ cảnh: "Ngày 15-1-2024, Thủ tướng Phạm Minh Chính đã ký quyết định..."
- Tuyên bố gốc: "Ông ấy đã ký quyết định"
- Tuyên bố độc lập: "Ngày 15-1-2024, Thủ tướng Phạm Minh Chính đã ký quyết định"

NGỮ CẢNH: {context}

TUYÊN BỐ: {claim}

TUYÊN BỐ ĐỘC LẬP:"""
    
    def decontextualize(self, claim: str, context: str) -> str:
        """
        Decontextualize a Vietnamese claim using the provided context.
        
        Args:
            claim: Original claim that may contain pronouns or context-dependent references
            context: Background information to resolve references
            
        Returns:
            Standalone, context-independent claim
        """
        prompt = self._create_prompt(claim, context)
        
        if self.config.provider == "anthropic":
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        else:
            # OpenAI and Ollama use the same API format
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content.strip()
    
    def process_dataset(self, dataset_path: str, output_path: str):
        """
        Process a dataset of claims and save decontextualized versions.
        
        Args:
            dataset_path: Path to input JSON file
            output_path: Path to save processed dataset
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single record and list of records
        records = [data] if isinstance(data, dict) else data
        
        processed_records = []
        for i, record in enumerate(tqdm(records, desc="Decontextualizing records")):
            
            original_claim = record.get('claim', '')
            context = record.get('context', '')
            evidence = record.get('evidence', context)
            label = record.get('label', '')

            if isinstance(label, int):
                mapped_label = {0: 'true', 1: 'false', 2: 'nei'}.get(label, 'nei')
            else:
                mapped_label = str(label).lower()
            
            try:
                standalone_claim = self.decontextualize(original_claim, context or evidence)
                
                processed_record = {
                    'id': i,  # Add sample index for reference
                    'claim': standalone_claim,
                    'original_claim': original_claim,
                    'label': mapped_label,
                    'evidence': evidence,
                    'decontextualized': True
                }
                processed_records.append(processed_record)
                
                print(f"  Original: {original_claim[:100]}...")
                print(f"  Standalone: {standalone_claim[:100]}...")
                print()
                
            except Exception as e:
                print(f"  Error: {e}")
                processed_records.append({
                    'id': i,
                    'claim': original_claim,
                    'label': mapped_label,
                    'evidence': evidence,
                    'decontextualized': False,
                    'error': str(e)
                })
        
        # Save processed dataset
        if output_path.lower().endswith('.jsonl'):
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in processed_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_records, f, ensure_ascii=False, indent=2)
        
        print(f"Processed {len(processed_records)} records")
        print(f"Saved to {output_path}")

    def process_vifactcheck_dataset(self, split: str = "test", output_path: str = "vifactcheck_decontextualized.jsonl", include_nei: bool = True):
        """
        Process ViFactCheck dataset and save decontextualized versions.
        
        Args:
            split: Dataset split to process ('train', 'validation', 'test')
            output_path: Path to save processed dataset (JSONL format)
            include_nei: Whether to include NEI samples
        """
        loader = ViFactCheckLoader()
        dataset = loader.load_dataset(split=split)
        
        processed_records = []
        for i, sample in enumerate(tqdm(dataset, desc=f"Decontextualizing {split} split")):
            
            original_claim = sample.get('Statement', sample.get('claim', ''))
            evidence = sample.get('Evidence', sample.get('evidence', ''))
            label = sample.get('labels', sample.get('label', ''))
            context = sample.get('Context', sample.get('context', ''))
            
            # Skip if no claim or context/evidence
            if not original_claim or not (context or evidence):
                continue
            
            # Map label for filtering
            if isinstance(label, int):
                mapped_label = {0: 'true', 1: 'false', 2: 'nei'}.get(label, 'nei')
            else:
                mapped_label = str(label).lower()
            
            if mapped_label == 'nei' and not include_nei:
                continue
            
            try:
                standalone_claim = self.decontextualize(original_claim, context or evidence)
                
                processed_record = {
                    'claim': standalone_claim,
                    'original_claim': original_claim,
                    'label': mapped_label,
                    'evidence': evidence,
                    'context': context,
                    'decontextualized': True
                }
                processed_records.append(processed_record)
                
                print(f"  Original: {original_claim}")
                print(f"  Standalone: {standalone_claim}")
                print()
                
            except Exception as e:
                print(f"  Error: {e}")
                processed_records.append({
                    'claim': original_claim,
                    'label': mapped_label,
                    'evidence': evidence,
                    'context': context,
                    'decontextualized': False,
                    'error': str(e)
                })
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in processed_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Processed {len(processed_records)} records from '{split}' split")
        print(f"Saved to {output_path}")



# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decontextualize Vietnamese claims")
    parser.add_argument("--dataset", type=str, default="", help="Path to input JSON file (single or list). If set, runs process_dataset.")
    parser.add_argument("--output", type=str, default="", help="Output path. Defaults based on mode if omitted.")
    parser.add_argument("--split", type=str, default="test", help="ViFactCheck split: train/validation/test")
    parser.add_argument("--exclude-nei", action="store_true", help="Exclude NEI samples for ViFactCheck mode")
    parser.add_argument("--provider", type=str, default="ollama", choices=["openai", "anthropic", "ollama"], help="LLM provider")
    parser.add_argument("--model", type=str, default="gemma3:4b", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for provider (e.g., Ollama)")
    args = parser.parse_args()

    config = DecontextualizeConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
    )

    decontextualizer = Decontextualizer(config)

    if args.dataset:
        output_path = args.output or "sample_data_decontextualized.json"
        decontextualizer.process_dataset(
            dataset_path=args.dataset,
            output_path=output_path,
        )
    else:
        output_path = args.output or "vifactcheck_decontextualized.jsonl"
        decontextualizer.process_vifactcheck_dataset(
            split=args.split,
            output_path=output_path,
            include_nei=not args.exclude_nei,
        )