"""
FIRE - Batch Processing Script
Chạy hết tất cả claims theo từng batch, lưu kết quả từng batch + merged final output
"""

import os
import json
import argparse
import dataclasses
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from langchain_community.callbacks.manager import get_openai_callback
from common.modeling import Model
from common.shared_config import openai_api_key, serper_api_key, anthropic_api_key


def process_batch(batch_lines, rater, verify_atomic_claim, batch_num, output_dir, dataset, model_name, framework):
    """
    Process a single batch of claims.
    
    Returns:
        (results_list, failed_count, usage_stats)
    """
    results = []
    failed_cnt = 0
    total_usage = {'input_tokens': 0, 'output_tokens': 0}
    
    for line in tqdm(batch_lines, desc=f"Batch {batch_num}", unit="claim", leave=False):
        try:
            data = json.loads(line)
            claim = data['claim']
            label = data['label']
            
            result, searches, usage = verify_atomic_claim(claim, rater)
            
            if usage is not None:
                total_usage['input_tokens'] += usage.get('input_tokens', 0)
                total_usage['output_tokens'] += usage.get('output_tokens', 0)
            
            if result is None:
                failed_cnt += 1
                continue
            
            results.append({
                'claim': claim,
                'label': label,
                'result': dataclasses.asdict(result),
                'searches': searches
            })
        
        except Exception as e:
            print(f"⚠️  Error: {claim[:50] if 'claim' in locals() else 'unknown'}... - {str(e)}")
            failed_cnt += 1
            continue
    
    # Save batch output
    if results:
        batch_output_file = f'{output_dir}/batch_{batch_num:04d}_{dataset}_{model_name}.jsonl'
        with open(batch_output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"✅ Batch {batch_num}: Saved {len(results)} results to {batch_output_file}")
    
    return results, failed_cnt, total_usage


def main():
    parser = argparse.ArgumentParser(
        description='Run FIRE fact-checking batch processing (processes all claims)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., ollama/qwen3:1.7b, gpt-4o-mini)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to evaluate (e.g., vifactcheck, factcheck_bench)')
    parser.add_argument('--framework', type=str, default='fire',
                        choices=['fire', 'safe'],
                        help='Framework to use (default: fire)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for processing (default: 50)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--thinking', action='store_true', default=False,
                        help='Enable thinking mode for models that support it')
    parser.add_argument('--skip-merge', action='store_true',
                        help='Skip merging batch outputs into single file')

    args = parser.parse_args()

    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key

    # Dataset mapping
    dataset_map = {
        'factcheck_bench': 'factcheckbench',
        'factcheckbench': 'factcheckbench',
        'bing_check': 'bingcheck',
        'bingcheck': 'bingcheck',
        'factool_qa': 'factool_qa',
        'felm_wk': 'felm_wk',
        'vifactcheck': 'vifactcheck',
    }

    benchmark = dataset_map.get(args.dataset.lower(), args.dataset)
    framework = args.framework

    model_name_full = args.model
    if not any(prefix in args.model for prefix in ['openai:', 'anthropic:', 'together:', 'ollama:']):
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

    # Create dated subdirectory for batch outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = f'{args.output_dir}/batch_{timestamp}'
    os.makedirs(batch_dir, exist_ok=True)

    dataset_path = f'datasets/{benchmark}/data.jsonl'

    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset file not found: {dataset_path}")
        print(f"Available datasets:")
        for root, dirs, files in os.walk('datasets'):
            for file in files:
                if file == 'data.jsonl':
                    print(f"  - {os.path.join(root, file)}")
        return

    print("=" * 80)
    print("🔥 FIRE Batch Processing Framework")
    print("=" * 80)
    print(f"Model:       {model_name_full}")
    print(f"Dataset:     {benchmark}")
    print(f"Framework:   {framework}")
    print(f"Batch Size:  {args.batch_size}")
    print(f"Thinking:    {'Enabled' if args.thinking else 'Disabled'}")
    print(f"Output Dir:  {batch_dir}")
    print("=" * 80 + "\n")

    # Load all data
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_claims = len(lines)
    print(f"📊 Total claims to process: {total_claims}")
    print(f"📦 Batch size: {args.batch_size}")
    num_batches = (total_claims + args.batch_size - 1) // args.batch_size
    print(f"📈 Number of batches: {num_batches}\n")

    # Initialize model
    rater = Model(model_name_full, enable_thinking=args.thinking)
    model_name = model_name_full.split(':')[-1].split('/')[-1]
    
    # Process batches
    all_results = []
    total_failed = 0
    total_usage = {'input_tokens': 0, 'output_tokens': 0}
    batch_start_time = time.time()

    with get_openai_callback() as cb:
        for batch_num in range(1, num_batches + 1):
            start_idx = (batch_num - 1) * args.batch_size
            end_idx = min(batch_num * args.batch_size, total_claims)
            batch_lines = lines[start_idx:end_idx]
            
            print(f"\n{'='*80}")
            print(f"Processing Batch {batch_num}/{num_batches} (claims {start_idx+1}-{end_idx})")
            print(f"{'='*80}")
            
            batch_results, batch_failed, batch_usage = process_batch(
                batch_lines, rater, verify_atomic_claim, batch_num,
                batch_dir, benchmark, model_name, framework
            )
            
            all_results.extend(batch_results)
            total_failed += batch_failed
            total_usage['input_tokens'] += batch_usage['input_tokens']
            total_usage['output_tokens'] += batch_usage['output_tokens']
            
            # Show batch stats
            print(f"   Results: {len(batch_results)} | Failed: {batch_failed}")
            print(f"   Tokens: {batch_usage['input_tokens']} in, {batch_usage['output_tokens']} out")

        # Merge all batch results into single file (unless skipped)
        if not args.skip_merge:
            print(f"\n{'='*80}")
            print("🔀 Merging batch results into final output file...")
            print(f"{'='*80}")
            
            mode_suffix = '_thinking' if args.thinking else ''
            final_output_file = f'{args.output_dir}/{framework}{mode_suffix}_{benchmark}_{model_name}_BATCH.jsonl'
            
            with open(final_output_file, 'w', encoding='utf-8') as fout:
                for result in all_results:
                    fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"✅ Merged output saved to: {final_output_file}")
        else:
            final_output_file = None

        # Print final summary
        batch_elapsed = time.time() - batch_start_time
        print(f"\n{'='*80}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Total processed: {len(all_results)} claims")
        print(f"❌ Failed: {total_failed} claims")
        print(f"⏱️  Total time: {batch_elapsed:.2f}s ({batch_elapsed/len(all_results):.2f}s per claim)")
        print(f"\n🔤 Token Usage:")
        print(f"   Input tokens:  {total_usage['input_tokens']:,}")
        print(f"   Output tokens: {total_usage['output_tokens']:,}")
        print(f"\n📂 Output locations:")
        print(f"   Batch files:   {batch_dir}/batch_*.jsonl")
        if final_output_file:
            print(f"   Merged output: {final_output_file}")
        print(f"\n💰 Cost Information:")
        print(cb)
        print("=" * 80)

        # Save metadata/summary
        summary_file = f'{batch_dir}/summary.json'
        summary = {
            'timestamp': timestamp,
            'model': model_name_full,
            'dataset': benchmark,
            'framework': framework,
            'batch_size': args.batch_size,
            'total_claims': total_claims,
            'processed_claims': len(all_results),
            'failed_claims': total_failed,
            'total_time_seconds': batch_elapsed,
            'avg_time_per_claim': batch_elapsed / len(all_results) if all_results else 0,
            'usage': total_usage,
            'batch_dir': batch_dir,
            'final_output_file': final_output_file,
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
