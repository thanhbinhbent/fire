"""
Transform local ViFactCheck data to FIRE format
Usage: python transform_local_data.py --input your_data.jsonl --output datasets/vifactcheck/data.jsonl
"""

import argparse
import json
from pathlib import Path
from common.vifactcheck_loader import ViFactCheckLoader


def main():
    parser = argparse.ArgumentParser(
        description='Transform local ViFactCheck data to FIRE format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input data file (JSON or JSONL)')
    parser.add_argument('--output', type=str, default='datasets/vifactcheck/data.jsonl',
                        help='Output FIRE format file (default: datasets/vifactcheck/data.jsonl)')
    parser.add_argument('--preview', type=int, default=5,
                        help='Number of samples to preview (default: 5)')
    parser.add_argument('--stats', action='store_true',
                        help='Show dataset statistics')

    args = parser.parse_args()

    print("=" * 80)
    print("🔄 ViFactCheck Data Transformer")
    print("=" * 80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("=" * 80 + "\n")

    # Initialize loader
    loader = ViFactCheckLoader()

    # Load local data
    try:
        loader.load_from_local_file(args.input)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {args.input}")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Show statistics
    if args.stats:
        stats = loader.get_statistics()
        print("\n📊 Dataset Statistics:")
        print("-" * 80)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print()

    # Preview samples
    if args.preview > 0:
        loader.print_sample_claims(n=args.preview)
        print()

    # Transform to FIRE format
    try:
        loader.convert_to_fire_format(output_path=args.output)
        print(f"\n✅ Success! Data transformed to: {args.output}")
        print("\n🚀 Next steps:")
        print(f"   1. Verify: head -n 3 {args.output}")
        print(f"   2. Test:   python benchmark.py --model ollama/qwen3:1.7b --dataset vifactcheck --limit 10")
    except Exception as e:
        print(f"\n❌ Error during transformation: {e}")


if __name__ == "__main__":
    main()
