"""
FIRE - Fact-checking with Iterative Retrieval and Verification
Command-line interface for running fact-checking evaluation
"""

import os
import json
import argparse
import dataclasses
import time
from tqdm import tqdm
from langchain_community.callbacks.manager import get_openai_callback
from common.modeling import Model
from common.shared_config import openai_api_key, serper_api_key, anthropic_api_key
from common.utils import calculate_cost_claude
from common.translator import Translator

def main():
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
    parser.add_argument("--use-translate", action="store_true",
                        help="Use translated claim for fire check")

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
        'felm_wk': 'felm_wk'
    }

    benchmark = dataset_map.get(args.dataset.lower(), args.dataset)
    framework = args.framework

    model_name_full = args.model
    if not any(prefix in args.model for prefix in ['openai:', 'anthropic:', 'together:']):
        if args.model.startswith('gpt') or args.model.startswith('o1'):
            model_name_full = f'openai:{args.model}'
        elif args.model.startswith('claude'):
            model_name_full = f'anthropic:{args.model}'

    if framework == 'fire':
        from eval.fire.verify_atomic_claim import verify_atomic_claim
        from eval.fire.verify_atomic_claim_original import verify_atomic_claim_original
    elif framework == 'safe':
        from eval.safe.rate_atomic_fact import check_atomic_fact
        verify_atomic_claim = check_atomic_fact

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_path = f'datasets/{benchmark}/data.jsonl'

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

    if (args.use_translate):
        translator = Translator(cache=True)
    total_search_calls = 0
    total_inference_time = 0
    y_true = []
    y_pred = []

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
                claim_vi = data['claim']
                label = data['label']
                claim_en = None
                
                start_time = time.time()
                if (args.use_translate):
                    # ===== TRANSLATE =====
                    claim_en = translator.vi_to_en(claim_vi)
                    print(f"\nOriginal (VI): {claim_vi}")
                    print(f"Translated (EN): {claim_en}")

                try:
                    if (args.use_translate):
                        result, searches, usage = verify_atomic_claim_original(claim_en, rater)
                    else:
                        result, searches, usage = verify_atomic_claim(claim_vi, rater)

                    inference_time = time.time() - start_time
                    total_inference_time += inference_time

                    if usage is not None:
                        total_usage['input_tokens'] += usage.get(
                            'input_tokens', 0)
                        total_usage['output_tokens'] += usage.get(
                            'output_tokens', 0)

                    if result is None:
                        failed_cnt += 1
                        continue

                    result_dict = dataclasses.asdict(result)
                    print(f"Result: {result_dict}")
                    print(f"True label: {label}")

                    pred_label = (
                        result_dict.get("answer")
                        or result_dict.get("final_answer")
                        or result_dict.get("verdict")
                        or result_dict.get("label")
                        or result_dict.get("prediction")
                    )

                    if pred_label:
                        y_true.append(label)
                        y_pred.append(pred_label)

                    fout.write(json.dumps({
                        "claim_vi": claim_vi,
                        "claim_en": claim_en,
                        "label": label,
                        "prediction": pred_label,
                        "result": result_dict,
                        "searches": searches,
                        "inference_time": inference_time
                    }, ensure_ascii=False) + '\n')

                except Exception as e:
                    print(f"\nError processing claim: {claim_vi[:50]}...")
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
