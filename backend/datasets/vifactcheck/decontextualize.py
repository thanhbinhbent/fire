"""
Decontextualization module for Vietnamese fact-checking claims.
Supports both cloud LLMs (OpenAI, Anthropic) and local LLMs (Ollama).
"""

import os
import json
from typing import Literal, Optional
from openai import OpenAI
from anthropic import Anthropic


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
1. Thay thế đại từ (anh ấy, cô ấy, họ, nó, các đối tượng, người này, v.v.) bằng tên hoặc thực thể cụ thể
2. BẮT BUỘC bao gồm ngày tháng cụ thể nếu có trong ngữ cảnh (ví dụ: "tháng 2-2023", "ngày 21-3", "ngày 24-3")
3. Bao gồm địa điểm hoặc sự kiện cụ thể được đề cập trong ngữ cảnh
4. KHÔNG thay đổi ý nghĩa hoặc giá trị chân lý của tuyên bố
5. Chỉ xuất ra tuyên bố đã được viết lại, KHÔNG thêm giải thích

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
        for i, record in enumerate(records):
            print(f"Processing record {i+1}/{len(records)}...")
            
            original_claim = record['claim']
            context = record['context']
            
            try:
                standalone_claim = self.decontextualize(original_claim, context)
                
                processed_record = {
                    **record,
                    'original_claim': original_claim,
                    'claim': standalone_claim,  # Replace with decontextualized version
                    'decontextualized': True
                }
                processed_records.append(processed_record)
                
                print(f"  Original: {original_claim[:100]}...")
                print(f"  Standalone: {standalone_claim[:100]}...")
                print()
                
            except Exception as e:
                print(f"  Error: {e}")
                processed_records.append({
                    **record,
                    'decontextualized': False,
                    'error': str(e)
                })
        
        # Save processed dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_records, f, ensure_ascii=False, indent=2)
        
        print(f"Processed {len(processed_records)} records")
        print(f"Saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Option 1: Using local Ollama
    config_local = DecontextualizeConfig(
        provider="ollama",
        model="gemma3:4b",
        temperature=0.3
    )
    
    # Option 2: Using OpenAI
    # config_cloud = DecontextualizeConfig(
    #     provider="openai",
    #     model="gpt-4o-mini",
    #     temperature=0.3
    # )
    
    # Option 3: Using Anthropic
    # config_cloud = DecontextualizeConfig(
    #     provider="anthropic",
    #     model="claude-3-5-sonnet-20241022",
    #     temperature=0.3
    # )
    
    decontextualizer = Decontextualizer(config_local)
    
    # Process sample data (updated paths for running from vifactcheck directory)
    decontextualizer.process_dataset(
        dataset_path="sample_data.json",
        output_path="sample_data_decontextualized.json"
    )