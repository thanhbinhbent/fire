"""
Quick Script: Run continuous batches of 10 claims until all data is processed
Just run this Python script, no complex command line needed
"""

import os
import sys
import json
import dataclasses
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from loguru import logger
from common.modeling import Model
from common.shared_config import openai_api_key, serper_api_key, anthropic_api_key
from eval.fire.verify_atomic_claim import verify_atomic_claim

# Setup logger
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"fire_continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.remove()  # Remove default handler
logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="DEBUG")
logger.add(lambda msg: sys.stderr.write(msg), format="{message}", level="INFO")  # Console output

# ===== CONFIGURATION =====
MODEL_NAME = "ollama/qwen3:1.7b"          # Change if needed
DATASET = "vifactcheck"
BATCH_SIZE = 10                            # Claims per batch
START_ROW = 0                              # Starting row number (0-indexed)
MAX_BATCHES = 100                          # Process 100 batches = 1000 claims (set None for all)
OUTPUT_DIR = "results"
FRAMEWORK = "fire"
ENABLE_THINKING = False

# ===== END CONFIGURATION =====

def main():
    # Setup
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key
    
    dataset_path = f'datasets/{DATASET}/data.jsonl'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all data
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_claims = len(lines)
    logger.info("="*80)
    logger.info(f"🔄 Continuous FIRE Processing")
    logger.info("="*80)
    logger.info(f"Dataset:     {DATASET}")
    logger.info(f"Total:       {total_claims} claims")
    logger.info(f"Start Row:   {START_ROW}")
    logger.info(f"Batch Size:  {BATCH_SIZE}")
    logger.info(f"Max Batches: {MAX_BATCHES if MAX_BATCHES else 'all'}")
    logger.info("="*80)
    
    # Initialize model
    rater = Model(MODEL_NAME, enable_thinking=ENABLE_THINKING)
    model_name = MODEL_NAME.split('/')[- 1].split(':')[0]
    
    # Output file
    output_file = f'{OUTPUT_DIR}/{FRAMEWORK}_{DATASET}_{model_name}_continuous.jsonl'
    
    # Calculate batches
    remaining_claims = total_claims - START_ROW
    num_batches = (remaining_claims + BATCH_SIZE - 1) // BATCH_SIZE
    if MAX_BATCHES:
        num_batches = min(num_batches, MAX_BATCHES)
    
    logger.info(f"📦 Processing {num_batches} batches")
    
    all_results = []
    total_failed = 0
    
    start_time = time.time()
    
    # Process batches
    for batch_num in range(num_batches):
        start_idx = START_ROW + (batch_num * BATCH_SIZE)
        end_idx = min(start_idx + BATCH_SIZE, total_claims)
        batch_lines = lines[start_idx:end_idx]
        
        logger.info(f"{'='*80}")
        logger.info(f"Batch {batch_num + 1}/{num_batches} (Claims {start_idx + 1}-{end_idx})")
        logger.info(f"{'='*80}")
        
        batch_results = 0
        batch_failed = 0
        
        for line in tqdm(batch_lines, desc=f"Batch {batch_num + 1}", unit="claim", leave=False):
            try:
                data = json.loads(line)
                claim = data['claim']
                label = data['label']
                
                result, searches, usage = verify_atomic_claim(claim, rater)
                
                if result is None:
                    batch_failed += 1
                    total_failed += 1
                    continue
                
                all_results.append({
                    'claim': claim,
                    'label': label,
                    'result': dataclasses.asdict(result),
                    'searches': searches
                })
                batch_results += 1
            
            except Exception as e:
                batch_failed += 1
                total_failed += 1
                continue
        
        # Save after each batch
        logger.success(f"Results: {batch_results} | Failed: {batch_failed}")
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for result in all_results[-batch_results:]:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 Checkpoint saved to: {output_file}")
    
    # Final stats
    elapsed = time.time() - start_time
    
    logger.info("="*80)
    logger.success("✅ DONE!")
    logger.info("="*80)
    logger.info(f"Processed: {len(all_results)} claims")
    logger.info(f"Failed:    {total_failed}")
    logger.info(f"Time:      {elapsed:.1f}s ({elapsed/max(len(all_results), 1):.2f}s per claim)")
    logger.info(f"📁 Output: {output_file}")
    logger.info(f"📊 Verify: wc -l {output_file}")
    logger.info("="*80)

if __name__ == '__main__':
    main()
