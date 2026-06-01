#!/usr/bin/env python3
# Copyright (c) Cerebras challenge submission.
# compare_runs.py — compare full vs pruned evaluation results.
#
# Usage:
#   python -m evalscope.benchmarks.cerebras_pruner.tools.compare_runs \
#       --full ./results_full/ \
#       --pruned ./results_pruned/
#
# The tool reads evalscope output directories, extracts per-model scores,
# computes Spearman rank correlation, and prints a summary table.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_score_files(results_dir: Path) -> List[Path]:
    """Find all score JSON files in an evalscope output directory."""
    score_files = []
    for pattern in ['**/scores.json', '**/report.json', '**/*_scores.json']:
        score_files.extend(results_dir.glob(pattern))
    return score_files


def load_scores_from_dir(results_dir: Path) -> Dict[str, float]:
    """
    Load model scores from an evalscope output directory.

    Looks for JSON files containing model scores in standard evalscope format.
    Returns dict mapping model_name -> score.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise ValueError(f'Results directory not found: {results_dir}')

    model_scores: Dict[str, float] = {}

    # Try to find score files
    score_files = find_score_files(results_dir)

    for score_file in score_files:
        try:
            with open(score_file) as f:
                data = json.load(f)

            # Handle different evalscope output formats
            if isinstance(data, dict):
                # Format 1: {model_name: {metric: score}}
                for key, val in data.items():
                    if isinstance(val, dict):
                        score = val.get('acc', val.get('pass', val.get('score')))
                        if score is not None:
                            model_scores[key] = float(score)
                    elif isinstance(val, (int, float)):
                        model_scores[key] = float(val)

            elif isinstance(data, list):
                # Format 2: [{model: name, score: val}]
                for item in data:
                    if isinstance(item, dict) and 'model' in item:
                        model = item['model']
                        score = item.get('acc', item.get('pass', item.get('score')))
                        if score is not None:
                            model_scores[model] = float(score)

        except (json.JSONDecodeError, KeyError):
            continue

    return model_scores


def load_scores_from_pruner_output(output_file: Path) -> Dict[str, float]:
    """Load scores from a pruner verification output JSON file."""
    with open(output_file) as f:
        data = json.load(f)

    verification = data.get('verification', {})
    return verification.get('pruned_scores', {})


def spearman_correlation(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
) -> float:
    """Compute Spearman rank correlation between two score dicts."""
    models = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    n = len(models)

    if n < 2:
        return 1.0

    rank_a = {m: r for r, m in enumerate(
        sorted(models, key=lambda m: scores_a[m], reverse=True)
    )}
    rank_b = {m: r for r, m in enumerate(
        sorted(models, key=lambda m: scores_b[m], reverse=True)
    )}

    d_sq = sum((rank_a[m] - rank_b[m]) ** 2 for m in models)
    return 1.0 - (6 * d_sq) / (n * (n ** 2 - 1))


def print_comparison_table(
    full_scores: Dict[str, float],
    pruned_scores: Dict[str, float],
    benchmark: str = '',
) -> None:
    """Print a formatted comparison table."""
    models = sorted(
        set(full_scores.keys()) | set(pruned_scores.keys()),
        key=lambda m: full_scores.get(m, 0),
        reverse=True,
    )

    print(f'\n{"=" * 70}')
    if benchmark:
        print(f'  Benchmark: {benchmark}')
    print(f'{"=" * 70}')
    print(f'  {"Model":<30} {"Full":>10} {"Pruned":>10} {"Delta":>10}')
    print(f'  {"-" * 62}')

    for model in models:
        full = full_scores.get(model)
        pruned = pruned_scores.get(model)
        full_str = f'{full:.4f}' if full is not None else 'N/A'
        pruned_str = f'{pruned:.4f}' if pruned is not None else 'N/A'
        if full is not None and pruned is not None:
            delta = pruned - full
            delta_str = f'{delta:+.4f}'
        else:
            delta_str = 'N/A'
        print(f'  {model:<30} {full_str:>10} {pruned_str:>10} {delta_str:>10}')

    if full_scores and pruned_scores:
        spearman = spearman_correlation(full_scores, pruned_scores)
        print(f'  {"-" * 62}')
        print(f'  Spearman rank correlation: {spearman:.4f}')
        if spearman >= 0.9:
            print(f'  ✅ Rankings preserved (Spearman ≥ 0.9)')
        elif spearman >= 0.5:
            print(f'  ⚠️  Partial rank preservation (Spearman ≥ 0.5)')
        else:
            print(f'  ❌ Rankings not preserved (Spearman < 0.5)')

    print(f'{"=" * 70}\n')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare full vs pruned evalscope evaluation results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare evalscope output directories
  python -m evalscope.benchmarks.cerebras_pruner.tools.compare_runs \\
      --full ./results_full/ \\
      --pruned ./results_pruned/

  # Compare pruner JSON output files directly
  python -m evalscope.benchmarks.cerebras_pruner.tools.compare_runs \\
      --full-json ./pruned_aa_lcr.json \\
      --pruned-json ./pruned_lcb.json

  # Specify benchmark name for display
  python -m evalscope.benchmarks.cerebras_pruner.tools.compare_runs \\
      --full ./results_full/ \\
      --pruned ./results_pruned/ \\
      --benchmark aa_lcr
        """,
    )

    parser.add_argument('--full', type=Path, help='Directory with full evaluation results')
    parser.add_argument('--pruned', type=Path, help='Directory with pruned evaluation results')
    parser.add_argument('--full-json', type=Path, help='Pruner JSON output for full run')
    parser.add_argument('--pruned-json', type=Path, help='Pruner JSON output for pruned run')
    parser.add_argument('--benchmark', type=str, default='', help='Benchmark name for display')

    args = parser.parse_args()

    if not any([args.full, args.pruned, args.full_json, args.pruned_json]):
        parser.print_help()
        return

    print('\n⚡ Cerebras PerfLens — Benchmark Compression Comparison')
    print('  Strategy: Discriminative Stratified Sampling')

    # Load full scores
    full_scores: Dict[str, float] = {}
    if args.full_json and args.full_json.exists():
        with open(args.full_json) as f:
            data = json.load(f)
        full_scores = data.get('verification', {}).get('full_scores', {})
    elif args.full and args.full.exists():
        full_scores = load_scores_from_dir(args.full)

    # Load pruned scores
    pruned_scores: Dict[str, float] = {}
    if args.pruned_json and args.pruned_json.exists():
        with open(args.pruned_json) as f:
            data = json.load(f)
        pruned_scores = data.get('verification', {}).get('pruned_scores', {})
    elif args.pruned and args.pruned.exists():
        pruned_scores = load_scores_from_dir(args.pruned)

    if not full_scores and not pruned_scores:
        print('\n⚠️  No scores found. Try running the pruner first:')
        print('  python -c "from evalscope.benchmarks.cerebras_pruner.pruner import prune_benchmark; ...')
        return

    print_comparison_table(full_scores, pruned_scores, benchmark=args.benchmark)


if __name__ == '__main__':
    main()
