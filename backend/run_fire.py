"""
FIRE - Fact-checking with Iterative Retrieval and Verification
Command-line interface for running fact-checking evaluation
"""

import os
import json
import argparse
import dataclasses
from tqdm import tqdm
from langchain_community.callbacks.manager import get_openai_callback
from common.modeling import Model
from common.shared_config import openai_api_key, serper_api_key, anthropic_api_key


def main():
    parser = argparse.ArgumentParser(
        description='Run FIRE fact-checking framework')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., gpt-4o-mini, claude-3-5-sonnet, gemma3:4b for local)')
    parser.add_argument('--provider', type=str, default=None,
                        choices=['openai', 'anthropic', 'local', 'ollama'],
                        help='LLM provider (auto-detected if not specified)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to evaluate (e.g., factcheck_bench, bingcheck, factool_qa, felm_wk, vifactcheck)')
    parser.add_argument('--framework', type=str, default='fire',
                        choices=['fire', 'safe'],
                        help='Framework to use (default: fire)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of claims to process (default: all)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')

    args = parser.parse_args()

    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key

    dataset_map = {
        'factcheck_bench': 'factcheckbench',
        'factcheckbench': 'factcheckbench',
        'bing_check': 'bingcheck',
        'bingcheck': 'bingcheck',
        'factool_qa': 'factool_qa',
        'felm_wk': 'felm_wk',
        'vifactcheck': 'vifactcheck',
        'vi_factcheck': 'vifactcheck',
        'vifact': 'vifactcheck',
        'vifactcheck_decontextualized': 'vifactcheck_decontextualized',
    }

    benchmark = dataset_map.get(args.dataset.lower(), args.dataset)
    framework = args.framework

    # Determine provider and format model name
    model_name_full = args.model
    provider = args.provider
    
    if provider:
        # Explicit provider specified
        if provider in ['local', 'ollama']:
            model_name_full = f'ollama:{args.model}'
        else:
            model_name_full = f'{provider}:{args.model}'
    elif not any(prefix in args.model for prefix in ['openai:', 'anthropic:', 'together:', 'local:', 'ollama:']):
        # Auto-detect provider from model name
        if args.model.startswith('gpt') or args.model.startswith('o1'):
            model_name_full = f'openai:{args.model}'
        elif args.model.startswith('claude'):
            model_name_full = f'anthropic:{args.model}'
        elif args.model.startswith('gemma') or args.model.startswith('llama') or args.model.startswith('mistral') or args.model.startswith('qwen'):
            # Common local model prefixes - default to local Ollama
            model_name_full = f'ollama:{args.model}'

    if framework == 'fire':
        from eval.fire.verify_atomic_claim import verify_atomic_claim

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_path = f'datasets/{benchmark}/data.jsonl'
    
    # TEST: Modify dataset_path to handle decontextualized dataset
    if benchmark == 'vifactcheck_decontextualized':
        dataset_path = 'datasets/vifactcheck/vifactcheck_decontextualized.jsonl'

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        print(f"Available datasets:")
        for root, dirs, files in os.walk('datasets'):
            for file in files:
                if file == 'data.jsonl':
                    print(f"  - {os.path.join(root, file)}")
        return

    print(f"FIRE Fact-Checking Framework")
    print(f"=" * 60)
    print(f"Model:     {model_name_full}")
    print(f"Dataset:   {benchmark}")
    print(f"Framework: {framework}")
    print(f"=" * 60)

    with get_openai_callback() as cb:
        print(f'\nRunning model: {model_name_full}')
        rater = Model(model_name_full)
        failed_cnt = 0
        model_name = model_name_full.split(':')[-1].split('/')[-1]

        total_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
        }

        output_file = f'{args.output_dir}/{framework}_{benchmark}_{model_name}.jsonl'

        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if args.limit:
            lines = lines[:args.limit]

        print(f"Processing {len(lines)} claims...")

        with open(output_file, 'w', encoding='utf-8') as fout:
            for line in tqdm(lines, desc="Processing claims"):
                data = json.loads(line)
                claim = data['claim']
                label = data['label']

                try:
                    result, searches, usage = verify_atomic_claim(claim, rater)

                    if usage is not None:
                        total_usage['input_tokens'] += usage.get(
                            'input_tokens', 0)
                        total_usage['output_tokens'] += usage.get(
                            'output_tokens', 0)

                    if result is None:
                        failed_cnt += 1
                        continue

                    fout.write(json.dumps({
                        'claim': claim,
                        'label': label,
                        'result': dataclasses.asdict(result),
                        'searches': searches
                    }, ensure_ascii=False) + '\n')

                except Exception as e:
                    print(f"\nError processing claim: {claim[:50]}...")
                    print(f"   Error: {str(e)}")
                    failed_cnt += 1
                    continue

        print(f"\n" + "=" * 60)
        print(f'All fact checking results saved to: {output_file}')
        print(f'Failed claims: {failed_cnt}')
        print(f"   Input tokens:  {total_usage['input_tokens']:,}")
        print(f"   Output tokens: {total_usage['output_tokens']:,}")
        print(f"\nCost Information:")
        print(cb)
        print("=" * 60)


if __name__ == '__main__':
    main()
