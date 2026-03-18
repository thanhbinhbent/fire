import argparse
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


def normalize_label(label: Optional[str]) -> str:
    if not label:
        return "nei"
    label = label.strip().lower()
    mapping = {
        "true": "true",
        "false": "false",
        "nei": "nei",
        "not enough info": "nei",
        "not_enough_info": "nei",
        "not enoughinfo": "nei",
        "supported": "true",
        "refuted": "false",
    }
    return mapping.get(label, label)


def safe_get_searches(response_json: Dict) -> int:
    metadata = response_json.get("metadata") or {}
    if "searches" in metadata:
        return int(metadata.get("searches") or 0)
    verification = metadata.get("verification") or {}
    return int(verification.get("searches") or 0)


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    labels = ["true", "false", "nei"]
    confusion = {l: defaultdict(int) for l in labels}

    for t, p in zip(y_true, y_pred):
        if t not in labels:
            continue
        if p not in labels:
            p = "nei"
        confusion[t][p] += 1

    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }

    macro_precision = sum(per_class[l]["precision"] for l in labels) / len(labels)
    macro_recall = sum(per_class[l]["recall"] for l in labels) / len(labels)
    macro_f1 = sum(per_class[l]["f1"] for l in labels) / len(labels)

    total = sum(sum(confusion[l].values()) for l in labels)
    correct = sum(confusion[l][l] for l in labels)
    micro_precision = correct / total if total else 0.0
    micro_recall = micro_precision
    micro_f1 = micro_precision

    return {
        "per_class": per_class,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate /api/check on a dataset")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/api/check",
        help="API endpoint for /api/check",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/vifactcheck/vifactcheck_decontextualized.jsonl",
        help="Path to JSONL dataset with claim and label",
    )
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "accurate"])
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--start", type=int, default=0, help="Start index (skip first N samples)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout seconds")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON file")
    parser.add_argument(
        "--results-jsonl",
        type=str,
        default=None,
        help="Optional JSONL file to write per-sample results",
    )
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Append to --results-jsonl if it exists",
    )
    parser.add_argument(
        "--summary-from-results",
        type=str,
        default=None,
        help="Compute summary from an existing results JSONL (no API calls)",
    )

    args = parser.parse_args()

    if args.summary_from_results:
        y_true = []
        y_pred = []
        with open(args.summary_from_results, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                label = normalize_label(item.get("label"))
                pred = normalize_label(item.get("pred")) if item.get("pred") else None
                if pred is None:
                    continue
                y_true.append(label)
                y_pred.append(pred)

        metrics = compute_metrics(y_true, y_pred)
        summary = {
            "metrics": metrics,
            "samples": len(y_true),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        return

    y_true: List[str] = []
    y_pred: List[str] = []
    total_searches = 0
    latencies: List[float] = []
    failures = 0

    with open(args.dataset, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if args.start:
        lines = lines[args.start :]

    if args.limit:
        lines = lines[: args.limit]

    total = len(lines)

    results_file = None
    if args.results_jsonl:
        mode = "a" if args.append_results else "w"
        results_file = open(args.results_jsonl, mode, encoding="utf-8")

    for line in tqdm(lines, total=total, desc="Evaluating"):
        item = json.loads(line)
        claim = item.get("claim", "").strip()
        label = normalize_label(item.get("label"))
        if not claim:
            continue

        payload = {"claim": claim, "mode": args.mode}
        if args.model:
            payload["model"] = args.model

        start = time.perf_counter()
        try:
            response = requests.post(args.api_url, json=payload, timeout=args.timeout)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            if response.status_code != 200:
                failures += 1
                if results_file:
                    results_file.write(
                        json.dumps(
                            {
                                "claim": claim,
                                "label": label,
                                "pred": None,
                                "status": response.status_code,
                                "error": response.text[:500],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            data = response.json()
            raw_verdict = normalize_label(data.get("raw_verdict"))
            y_true.append(label)
            y_pred.append(raw_verdict)
            total_searches += safe_get_searches(data)
            if results_file:
                results_file.write(
                    json.dumps(
                        {
                            "claim": claim,
                            "label": label,
                            "pred": raw_verdict,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:
            failures += 1
            if results_file:
                results_file.write(
                    json.dumps(
                        {
                            "claim": claim,
                            "label": label,
                            "pred": None,
                            "error": str(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            continue

    if results_file:
        results_file.close()

    metrics = compute_metrics(y_true, y_pred)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0.0

    summary = {
        "metrics": metrics,
        "total_searches": total_searches,
        "avg_latency_sec": avg_latency,
        "p95_latency_sec": p95_latency,
        "samples": len(y_true),
        "failures": failures,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
