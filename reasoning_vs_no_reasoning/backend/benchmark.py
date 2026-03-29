"""
Benchmark script comparing Thinking vs No-Thinking modes for Ollama models
Evaluates FIRE fact-checking framework with different model configurations
Similar structure to run_fire.py
"""

import os
import json
import argparse
import time
import dataclasses
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from common.modeling import Model
from common.shared_config import openai_api_key, serper_api_key, anthropic_api_key


def run_benchmark_mode(model_name, dataset_path, enable_thinking=False, limit=None, output_dir='results'):
    """
    Run FIRE fact-checking with specific thinking mode
    
    Args:
        model_name: Model identifier (e.g., 'ollama/qwen3:1.7b')
        dataset_path: Path to dataset JSONL file
        enable_thinking: Whether to enable thinking mode
        limit: Limit number of claims to process
        output_dir: Output directory for results
    
    Returns:
        Dictionary with metrics and stats
    """
    
    mode_name = "THINKING" if enable_thinking else "NO-THINKING (RAW)"
    print(f"\n{'='*70}")
    print(f"Running: {mode_name} Mode")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    
    # Import here to avoid circular imports
    from eval.fire.verify_atomic_claim import verify_atomic_claim
    
    # Initialize model - similar to run_fire.py
    rater = Model(model_name)
    failed_cnt = 0
    
    results = []
    total_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
    }
    
    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        return None
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if limit:
        lines = lines[:limit]
    
    print(f"Processing {len(lines)} claims...\n")
    
    # Process each claim - similar to run_fire.py
    for line in tqdm(lines, desc=f"{mode_name} Mode", unit="claim"):
        try:
            data = json.loads(line)
            claim = data['claim']
            true_label = data['label']  # True/False
            
            start_time = time.time()
            
            # Run verification
            result, searches, usage = verify_atomic_claim(claim, rater)
            
            inference_time = time.time() - start_time
            
            if usage is not None:
                total_usage['input_tokens'] += usage.get('input_tokens', 0)
                total_usage['output_tokens'] += usage.get('output_tokens', 0)
            
            if result is None:
                failed_cnt += 1
                continue
            
            # Extract metrics
            predicted_label = result.answer
            num_searches = len(searches.get('google_searches', [])) if searches else 0
            
            # Map labels to binary (True=1, False=0)
            true_bin = 1 if true_label == "True" or true_label == True else 0
            pred_bin = 1 if predicted_label == "True" or predicted_label == True else 0
            
            results.append({
                'claim': claim,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'true_bin': true_bin,
                'pred_bin': pred_bin,
                'inference_time': inference_time,
                'num_searches': num_searches,
                'response': result.response
            })
        
        except Exception as e:
            print(f"\nError processing claim: {claim[:50] if 'claim' in locals() else 'unknown'}...")
            print(f"   Error: {str(e)}")
            failed_cnt += 1
            continue
    
    # Calculate metrics only if we have results
    if not results:
        print(f"ERROR: No successful results for {mode_name} mode")
        return None
    
    y_true = [r['true_bin'] for r in results]
    y_pred = [r['pred_bin'] for r in results]
    
    # Calculate metrics - same as run_fire.py
    metrics = {
        'mode': mode_name,
        'enable_thinking': enable_thinking,
        'model': model_name,
        'num_samples': len(results),
        'failed_claims': failed_cnt,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'total_searches': sum(r['num_searches'] for r in results),
        'avg_searches': sum(r['num_searches'] for r in results) / len(results) if results else 0,
        'total_inference_time': sum(r['inference_time'] for r in results),
        'avg_inference_time': sum(r['inference_time'] for r in results) / len(results) if results else 0,
        'total_input_tokens': total_usage['input_tokens'],
        'total_output_tokens': total_usage['output_tokens'],
        'avg_input_tokens': total_usage['input_tokens'] / len(results) if results else 0,
        'avg_output_tokens': total_usage['output_tokens'] / len(results) if results else 0,
    }
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    mode_suffix = "thinking" if enable_thinking else "no_thinking"
    model_name_clean = model_name.replace('/', '_').replace(':', '_')
    results_file = f'{output_dir}/benchmark_{model_name_clean}_{mode_suffix}.jsonl'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps({
                'claim': result['claim'],
                'true_label': result['true_label'],
                'predicted_label': result['predicted_label'],
                'inference_time': result['inference_time'],
                'num_searches': result['num_searches'],
            }, ensure_ascii=False) + '\n')
    
    print(f"\nResults saved to: {results_file}")
    
    return metrics



def main():
    """Main entry point - similar structure to run_fire.py"""
    
    parser = argparse.ArgumentParser(
        description='Run FIRE fact-checking benchmark with Thinking vs No-Thinking modes')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., ollama/qwen3:1.7b, gpt-4o-mini, claude-3-5-sonnet-20240620)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to evaluate (e.g., vifactcheck, factcheck_bench, bingcheck, factool_qa, felm_wk)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of claims to process (default: all)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')

    args = parser.parse_args()

    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key

    # Map dataset names - same as run_fire.py
    dataset_map = {
        'vifactcheck': 'vifactcheck',
        'factcheck_bench': 'factcheckbench',
        'factcheckbench': 'factcheckbench',
        'bing_check': 'bingcheck',
        'bingcheck': 'bingcheck',
        'factool_qa': 'factool_qa',
        'felm_wk': 'felm_wk'
    }

    benchmark = dataset_map.get(args.dataset.lower(), args.dataset)
    dataset_path = f'datasets/{benchmark}/data.jsonl'

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        print(f"Available datasets:")
        for root, dirs, files in os.walk('datasets'):
            for file in files:
                if file == 'data.jsonl':
                    print(f"  - {os.path.join(root, file)}")
        return

    print(f"\n{'='*70}")
    print(f"FIRE Benchmark: Thinking vs No-Thinking")
    print(f"{'='*70}")
    print(f"Model:     {args.model}")
    print(f"Dataset:   {benchmark}")
    print(f"Limit:     {args.limit if args.limit else 'all'}")
    print(f"{'='*70}")

    # Run benchmark with no-thinking first
    no_thinking_metrics = run_benchmark_mode(
        model_name=args.model,
        dataset_path=dataset_path,
        enable_thinking=False,
        limit=args.limit,
        output_dir=args.output_dir
    )

    # Then run with thinking enabled
    thinking_metrics = run_benchmark_mode(
        model_name=args.model,
        dataset_path=dataset_path,
        enable_thinking=True,
        limit=args.limit,
        output_dir=args.output_dir
    )

    # Print detailed comparison
    if thinking_metrics and no_thinking_metrics:
        print_comparison(thinking_metrics, no_thinking_metrics, args.output_dir)
    else:
        print("ERROR: Could not complete benchmark")


def print_comparison(thinking_metrics, no_thinking_metrics, output_dir='results'):
    """Print and save detailed comparison between thinking and no-thinking modes"""
    
    print(f"\n{'='*100}")
    print(f"BENCHMARK RESULTS: THINKING vs NO-THINKING")
    print(f"{'='*100}\n")
    
    # Performance metrics
    print(f"{'Metric':<30} {'NO-THINKING':>20} {'THINKING':>20} {'Difference':>20}")
    print("-" * 110)
    
    metrics_to_compare = [
        ('Accuracy', 'accuracy'),
        ('F1 Score', 'f1'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
    ]
    
    for display_name, metric_key in metrics_to_compare:
        nt_val = no_thinking_metrics.get(metric_key, 0)
        t_val = thinking_metrics.get(metric_key, 0)
        diff = t_val - nt_val
        
        print(f"{display_name:<30} {nt_val:>20.4f} {t_val:>20.4f} {diff:>+20.4f}")
    
    print("\n" + "-" * 110 + "\n")
    
    # Cost metrics
    print(f"{'Metric':<30} {'NO-THINKING':>20} {'THINKING':>20} {'Difference':>20}")
    print("-" * 110)
    
    cost_metrics = [
        ('Avg Inference Time (s)', 'avg_inference_time'),
        ('Total Inference Time (s)', 'total_inference_time'),
        ('Avg Web Searches', 'avg_searches'),
        ('Total Web Searches', 'total_searches'),
        ('Avg Input Tokens', 'avg_input_tokens'),
        ('Avg Output Tokens', 'avg_output_tokens'),
    ]
    
    for display_name, metric_key in cost_metrics:
        nt_val = no_thinking_metrics.get(metric_key, 0)
        t_val = thinking_metrics.get(metric_key, 0)
        diff = t_val - nt_val
        
        if 'time' in metric_key.lower():
            print(f"{display_name:<30} {nt_val:>19.2f}s {t_val:>19.2f}s {diff:>+19.2f}s")
        else:
            print(f"{display_name:<30} {nt_val:>20.2f} {t_val:>20.2f} {diff:>+20.2f}")
    
    print("\n" + "-" * 110 + "\n")
    
    # Summary
    print(f"{'Metric':<30} {'Value':>30}")
    print("-" * 110)
    print(f"{'Number of Samples':<30} {thinking_metrics['num_samples']:>30}")
    print(f"{'Model':<30} {thinking_metrics['model']:>30}")
    print(f"{'Failed Claims (NO-THINKING)':<30} {no_thinking_metrics['failed_claims']:>30}")
    print(f"{'Failed Claims (THINKING)':<30} {thinking_metrics['failed_claims']:>30}")
    
    print("\n" + "=" * 110)
    
    # Save summary to file
    summary_file = f'{output_dir}/benchmark_summary.json'
    summary = {
        'model': thinking_metrics['model'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'no_thinking': no_thinking_metrics,
        'thinking': thinking_metrics,
        'comparison': {
            'accuracy_improvement': thinking_metrics['accuracy'] - no_thinking_metrics['accuracy'],
            'f1_improvement': thinking_metrics['f1'] - no_thinking_metrics['f1'],
            'precision_improvement': thinking_metrics['precision'] - no_thinking_metrics['precision'],
            'recall_improvement': thinking_metrics['recall'] - no_thinking_metrics['recall'],
            'time_overhead': thinking_metrics['avg_inference_time'] - no_thinking_metrics['avg_inference_time'],
            'search_overhead': thinking_metrics['avg_searches'] - no_thinking_metrics['avg_searches'],
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()