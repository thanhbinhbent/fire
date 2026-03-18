"""
Continuous FIRE Processing - Process 10 claims at a time, repeat until all data processed
Runs --limit 10 repeatedly until all claims in dataset are processed
"""

import os
import sys
import json
import argparse
import dataclasses
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from loguru import logger
from langchain_community.callbacks.manager import get_openai_callback
from common.modeling import Model
from common.shared_config import openai_api_key, serper_api_key, anthropic_api_key

# Setup logger
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"fire_continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.remove()  # Remove default handler
logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="DEBUG")
logger.add(lambda msg: sys.stderr.write(msg), format="{message}", level="INFO")  # Console output


def main():
    parser = argparse.ArgumentParser(
        description='Continuous FIRE processing - Process data in 10-claim chunks repeatedly')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., ollama/qwen3:1.7b)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to evaluate (e.g., vifactcheck)')
    parser.add_argument('--framework', type=str, default='fire',
                        choices=['fire', 'safe'],
                        help='Framework to use (default: fire)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for each run (default: 10)')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Max number of batches to process (None = all)')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting row number (0-indexed, default: 0)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--thinking', action='store_true', default=False,
                        help='Enable thinking mode')

    args = parser.parse_args()

    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key

    # Dataset mapping
    dataset_map = {
        'vifactcheck': 'vifactcheck',
        'factcheck_bench': 'factcheckbench',
        'factcheckbench': 'factcheckbench',
        'bingcheck': 'bingcheck',
        'factool_qa': 'factool_qa',
        'felm_wk': 'felm_wk',
    }

    benchmark = dataset_map.get(args.dataset.lower(), args.dataset)
    framework = args.framework

    model_name_full = args.model
    if not any(prefix in args.model for prefix in ['openai:', 'anthropic:', 'ollama:']):
        if args.model.startswith('gpt') or args.model.startswith('o1'):
            model_name_full = f'openai:{args.model}'
        elif args.model.startswith('claude'):
            model_name_full = f'anthropic:{args.model}'

    if framework == 'fire':
        from eval.fire.verify_atomic_claim import verify_atomic_claim
    elif framework == 'safe':
        from eval.safe.rate_atomic_fact import check_atomic_fact
        verify_atomic_claim = check_atomic_fact

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_path = f'datasets/{benchmark}/data.jsonl'

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return

    # Count total claims
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_claims = len(lines)
    
    logger.info("="*80)
    logger.info("🔄 FIRE Continuous Processing (Batch by Batch)")
    logger.info("="*80)
    logger.info(f"Model:       {model_name_full}")
    logger.info(f"Dataset:     {benchmark} ({total_claims} total claims)")
    logger.info(f"Start Row:   {args.start}")
    logger.info(f"Batch Size:  {args.batch_size}")
    logger.info(f"Max Batches: {args.max_batches if args.max_batches else 'all'}")
    logger.info(f"Thinking:    {'Enabled' if args.thinking else 'Disabled'}")
    logger.info("="*80)

    # Initialize model
    rater = Model(model_name_full, enable_thinking=args.thinking)
    model_name = model_name_full.split(':')[-1].split('/')[-1]

    # Setup output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = '_thinking' if args.thinking else ''
    output_file = f'{args.output_dir}/{framework}{mode_suffix}_{benchmark}_{model_name}_continuous.jsonl'
    
    # Open output file in append mode (in case of resume)
    all_results = []
    failed_count = 0
    total_usage = {'input_tokens': 0, 'output_tokens': 0}
    
    # Calculate batches
    remaining_claims = total_claims - args.start
    num_batches = (remaining_claims + args.batch_size - 1) // args.batch_size
    if args.max_batches:
        num_batches = min(num_batches, args.max_batches)
    
    logger.info(f"📦 Processing {num_batches} batches of {args.batch_size} claims each")

    with get_openai_callback() as cb:
        for batch_num in range(num_batches):
            start_idx = args.start + (batch_num * args.batch_size)
            end_idx = min(start_idx + args.batch_size, total_claims)
            batch_lines = lines[start_idx:end_idx]
            
            logger.info(f"{'='*80}")
            logger.info(f"Batch {batch_num + 1}/{num_batches} (Claims {start_idx + 1}-{end_idx} / {total_claims})")
            logger.info(f"{'='*80}")
            
            batch_results = 0
            batch_failed = 0
            batch_usage = {'input_tokens': 0, 'output_tokens': 0}
            
            with tqdm(batch_lines, desc=f"Batch {batch_num + 1}", unit="claim") as pbar:
                for line in pbar:
                    try:
                        data = json.loads(line)
                        claim = data['claim']
                        label = data['label']
                        
                        start_claim = time.perf_counter()
                        result, searches, usage = verify_atomic_claim(claim, rater)
                        claim_elapsed = time.perf_counter() - start_claim
                        
                        if usage is not None:
                            batch_usage['input_tokens'] += usage.get('input_tokens', 0)
                            batch_usage['output_tokens'] += usage.get('output_tokens', 0)
                            total_usage['input_tokens'] += usage.get('input_tokens', 0)
                            total_usage['output_tokens'] += usage.get('output_tokens', 0)
                        
                        if result is None:
                            batch_failed += 1
                            failed_count += 1
                            continue
                        
                        all_results.append({
                                                                                                                                                                                                                                                                                                                             'claim': claim,
                            'label': label,
                            'result': dataclasses.asdict(result),
                            'searches': searches,
                            'runtime_seconds': round(claim_elapsed, 4)
                        })
                        batch_results += 1
                    
                    except Exception as e:
                        batch_failed += 1
                        failed_count += 1
                        pbar.set_postfix({'error': str(e)[:30]})
                        continue
            
            # Save batch results immediately
            logger.success(f"Batch {batch_num + 1}: Processed {batch_results} | Failed {batch_failed}")
            logger.debug(f"Tokens: {batch_usage['input_tokens']} in, {batch_usage['output_tokens']} out")
            
            # Write to file after each batch (checkpoint)
            with open(output_file, 'a', encoding='utf-8') as f:
                # Only write the new results from this batch
                for result in all_results[max(0, len(all_results) - batch_results):]:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            logger.info(f"📝 Checkpoint saved to: {output_file}")

        # Final summary
        logger.info("="*80)
        logger.success("✅ PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Total Results: {len(all_results)}")
        logger.info(f"Total Failed:  {failed_count}")
        logger.info(f"📋 Token Usage:")
        logger.info(f"   Input:  {total_usage['input_tokens']:,}")
        logger.info(f"   Output: {total_usage['output_tokens']:,}")
        logger.info(f"📁 Output File: {output_file}")
        logger.info(f"   Verify: wc -l {output_file}")
        logger.info(f"💰 Cost Info:")
        logger.info(str(cb))
        logger.info("="*80)


if __name__ == '__main__':
    main()
