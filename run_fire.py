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
from common.utils import calculate_cost_claude


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run FIRE fact-checking framework')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., gpt-4o-mini, gpt-4o, claude-3-5-sonnet-20240620)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to evaluate (e.g., factcheck_bench, bingcheck, factool_qa, felm_wk)')
    parser.add_argument('--framework', type=str, default='fire',
                        choices=['fire', 'safe'],
                        help='Framework to use (default: fire)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of claims to process (default: all)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')

    args = parser.parse_args()

    # Set API keys for various services
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key

    # Map dataset names
    dataset_map = {
        'factcheck_bench': 'factcheckbench',
        'factcheckbench': 'factcheckbench',
        'bing_check': 'bingcheck',
        'bingcheck': 'bingcheck',
        'factool_qa': 'factool_qa',
        'felm_wk': 'felm_wk'
    }

    benchmark = dataset_map.get(args.dataset.lower(), args.dataset)
    framework = args.framework

    # Prepare model name with prefix if needed
    model_name_full = args.model
    if not any(prefix in args.model for prefix in ['openai:', 'anthropic:', 'together:']):
        # Add openai prefix for common models
        if args.model.startswith('gpt') or args.model.startswith('o1'):
            model_name_full = f'openai:{args.model}'
        elif args.model.startswith('claude'):
            model_name_full = f'anthropic:{args.model}'

    # Import the appropriate fact-checking function based on the framework
    if framework == 'fire':
        from eval.fire.verify_atomic_claim import verify_atomic_claim
    elif framework == 'safe':
        from eval.safe.rate_atomic_fact import check_atomic_fact
        verify_atomic_claim = check_atomic_fact  # Alias for compatibility

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset path
    dataset_path = f'datasets/{benchmark}/data.jsonl'

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset file not found: {dataset_path}")
        print(f"Available datasets:")
        for root, dirs, files in os.walk('datasets'):
            for file in files:
                if file == 'data.jsonl':
                    print(f"  - {os.path.join(root, file)}")
        return

    print(f"🔥 FIRE Fact-Checking Framework")
    print(f"=" * 60)
    print(f"Model:     {model_name_full}")
    print(f"Dataset:   {benchmark}")
    print(f"Framework: {framework}")
    print(f"=" * 60)

    with get_openai_callback() as cb:
        print(f'\n🚀 Running model: {model_name_full}')
        rater = Model(model_name_full)
        failed_cnt = 0
        # Extract model name for file saving
        model_name = model_name_full.split(':')[-1].split('/')[-1]

        # Initialize total token usage tracking
        total_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
        }

        # Output file path
        output_file = f'{args.output_dir}/{framework}_{benchmark}_{model_name}.jsonl'

        # Read dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Apply limit if specified
        if args.limit:
            lines = lines[:args.limit]

        print(f"📊 Processing {len(lines)} claims...")

        # Open output file to save the results
        with open(output_file, 'w', encoding='utf-8') as fout:
            # Read and process each line from the dataset
            for line in tqdm(lines, desc="Processing claims"):
                data = json.loads(line)  # Load JSON data
                claim = data['claim']    # Extract the claim
                label = data['label']    # Extract the label

                try:
                    # Run the fact-checking function for the claim
                    result, searches, usage = verify_atomic_claim(claim, rater)

                    # Track token usage if available
                    if usage is not None:
                        total_usage['input_tokens'] += usage.get(
                            'input_tokens', 0)
                        total_usage['output_tokens'] += usage.get(
                            'output_tokens', 0)

                    # If result is None, count it as a failure and skip further processing
                    if result is None:
                        failed_cnt += 1
                        continue

                    # Write the result to the output file
                    fout.write(json.dumps({
                        'claim': claim,
                        'label': label,
                        'result': dataclasses.asdict(result),
                        'searches': searches
                    }, ensure_ascii=False) + '\n')

                except Exception as e:
                    print(f"\n⚠️  Error processing claim: {claim[:50]}...")
                    print(f"   Error: {str(e)}")
                    failed_cnt += 1
                    continue

        print(f"\n" + "=" * 60)
        print(f'✅ All fact checking results saved to: {output_file}')
        print(f'❌ Failed claims: {failed_cnt}')
        print(f"\n📊 Token Usage:")
        print(f"   Input tokens:  {total_usage['input_tokens']:,}")
        print(f"   Output tokens: {total_usage['output_tokens']:,}")
        print(f"\n💰 Cost Information:")
        print(cb)
        print("=" * 60)


if __name__ == '__main__':
    main()
