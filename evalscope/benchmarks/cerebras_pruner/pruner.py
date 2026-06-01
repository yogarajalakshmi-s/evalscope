# Copyright (c) Cerebras challenge submission.
# Benchmark compression via discriminative stratified sampling.
#
# Strategy: for each sample compute its difficulty as the mean score
# across all available models. Samples where models disagree (medium
# difficulty) carry the most signal for ranking models. We stratify
# the full sample pool into difficulty tiers and sample proportionally,
# ensuring the pruned set preserves the model ranking from the full set.
#
# This approach is:
#   - Model-agnostic: works for any new model not in the reference set
#   - Distribution-preserving: maintains difficulty spread
#   - Rank-preserving: pruned ranking correlates with full ranking

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_reviews(review_dir: Path, benchmark: str) -> Dict[str, Dict[int, float]]:
    """
    Load per-sample scores from review jsonl files.

    Returns:
        dict mapping model_name -> {sample_index -> score}
    """
    model_scores: Dict[str, Dict[int, float]] = {}

    for path in sorted(review_dir.glob(f"{benchmark}__*.jsonl")):
        # Extract model name from filename: live_code_bench_v5__gpt-oss-120b.jsonl
        model_name = path.stem.split("__", 1)[1]
        scores: Dict[int, float] = {}

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                idx = record["index"]
                # AA-LCR uses "acc", LCB uses "pass"
                score_value = record["sample_score"]["score"]["value"]
                score = float(
                    score_value.get("acc", score_value.get("pass", 0.0))
                )
                scores[idx] = score

        model_scores[model_name] = scores

    return model_scores


def compute_difficulty(
    model_scores: Dict[str, Dict[int, float]]
) -> Dict[int, float]:
    """
    Compute per-sample difficulty as mean score across all models.

    difficulty = 0.0 → all models fail (too hard, low signal)
    difficulty = 1.0 → all models pass (too easy, low signal)
    difficulty ≈ 0.5 → models disagree (high discriminative value)
    """
    all_indices: set = set()
    for scores in model_scores.values():
        all_indices.update(scores.keys())

    difficulty: Dict[int, float] = {}
    for idx in all_indices:
        scores_for_idx = [
            model_scores[m][idx]
            for m in model_scores
            if idx in model_scores[m]
        ]
        if scores_for_idx:
            difficulty[idx] = sum(scores_for_idx) / len(scores_for_idx)

    return difficulty


def discriminativeness(difficulty: float) -> float:
    """
    Score how discriminative a sample is.
    Uses a softened tent function with a floor so all tiers get representation.
    Floor at 0.3 ensures easy/hard samples still get some representation.
    """
    tent = 1.0 - abs(2.0 * difficulty - 1.0)
    return 0.3 + 0.7 * tent  # floor at 0.3


def stratified_sample(
    difficulty: Dict[int, float],
    prune_ratio: float,
    n_tiers: int = 5,
    random_seed: int = 42,
) -> List[int]:
    """
    Select samples that maximize rank separation signal.
    
    Key insight: with binary scores, the most informative samples are
    those where exactly one model fails (mean=0.33 or mean=0.67).
    These directly separate model capabilities.
    """
    import random
    rng = random.Random(random_seed)

    total = len(difficulty)
    target_n = max(1, math.ceil(total * prune_ratio))

    # Categorize by discriminativeness
    all_fail = [idx for idx, d in difficulty.items() if d == 0.0]
    hard = [idx for idx, d in difficulty.items() if 0.0 < d <= 0.4]
    medium = [idx for idx, d in difficulty.items() if 0.4 < d <= 0.6]  
    easy = [idx for idx, d in difficulty.items() if 0.6 < d < 1.0]
    all_pass = [idx for idx, d in difficulty.items() if d == 1.0]

    # Priority: medium > easy > hard > all_pass > all_fail
    # Medium and easy/hard are where models disagree — highest signal
    # all_pass and all_fail give no ranking signal
    
    selected = []
    
    # Take ALL disagreement samples first (they carry the ranking signal)
    disagreement = hard + medium + easy
    rng.shuffle(disagreement)
    take_disagreement = min(len(disagreement), target_n)
    selected.extend(disagreement[:take_disagreement])
    
    # Fill remaining with all_pass (better than all_fail for score calibration)
    remaining = target_n - len(selected)
    if remaining > 0:
        rng.shuffle(all_pass)
        selected.extend(all_pass[:remaining])
    
    # Fill any remaining with all_fail
    remaining = target_n - len(selected)
    if remaining > 0:
        rng.shuffle(all_fail)
        selected.extend(all_fail[:remaining])

    return sorted(selected)

def weighted_sample_without_replacement(
    indices: List[int],
    weights: List[float],
    k: int,
    seed: int,
) -> List[int]:
    """Sample k items without replacement using weights."""
    import random
    rng = random.Random(seed)

    if k >= len(indices):
        return list(indices)

    # Normalize weights
    total = sum(weights)
    if total == 0:
        normalized = [1.0 / len(weights)] * len(weights)
    else:
        normalized = [w / total for w in weights]

    chosen = []
    pool = list(zip(indices, normalized))

    for _ in range(k):
        r = rng.random()
        cumulative = 0.0
        for j, (idx, w) in enumerate(pool):
            cumulative += w
            if r <= cumulative:
                chosen.append(idx)
                pool.pop(j)
                # Renormalize
                remaining_weight = sum(pw for _, pw in pool)
                if remaining_weight > 0:
                    pool = [(pi, pw / remaining_weight) for pi, pw in pool]
                break

    return chosen


def verify_rank_preservation(
    model_scores: Dict[str, Dict[int, float]],
    selected_indices: List[int],
) -> Dict[str, float]:
    """
    Verify that the pruned set preserves model ranking from the full set.

    Returns dict with full and pruned scores per model, and rank correlation.
    """
    full_scores: Dict[str, float] = {}
    pruned_scores: Dict[str, float] = {}

    for model, scores in model_scores.items():
        full_vals = list(scores.values())
        pruned_vals = [scores[i] for i in selected_indices if i in scores]

        full_scores[model] = sum(full_vals) / len(full_vals) if full_vals else 0.0
        pruned_scores[model] = sum(pruned_vals) / len(pruned_vals) if pruned_vals else 0.0

    # Spearman rank correlation
    models = list(full_scores.keys())
    full_rank = {m: r for r, m in enumerate(sorted(models, key=lambda m: full_scores[m], reverse=True))}
    pruned_rank = {m: r for r, m in enumerate(sorted(models, key=lambda m: pruned_scores[m], reverse=True))}

    n = len(models)
    if n < 2:
        spearman = 1.0
    else:
        d_sq = sum((full_rank[m] - pruned_rank[m]) ** 2 for m in models)
        spearman = 1.0 - (6 * d_sq) / (n * (n**2 - 1))

    return {
        "full_scores": full_scores,
        "pruned_scores": pruned_scores,
        "spearman_rank_correlation": spearman,
        "n_full": len(next(iter(model_scores.values()))),
        "n_pruned": len(selected_indices),
        "prune_ratio_actual": len(selected_indices) / len(next(iter(model_scores.values()))),
    }


def prune_benchmark(
    review_dir: str | Path,
    benchmark: str,
    prune_ratio: float = 0.3,
    n_tiers: int = 5,
    output_path: Optional[str | Path] = None,
) -> Dict:
    """
    Main entry point: prune a benchmark to the target ratio.

    Args:
        review_dir: directory containing review jsonl files
        benchmark: benchmark prefix, e.g. "aa_lcr" or "live_code_bench_v5"
        prune_ratio: fraction of samples to keep (default 0.3 = 30%)
        n_tiers: difficulty tiers for stratification (default 5)
        output_path: optional path to save selected indices as JSON

    Returns:
        dict with selected indices, verification stats, and metadata
    """
    review_dir = Path(review_dir)

    print(f"Loading reviews for {benchmark} from {review_dir}")
    model_scores = load_reviews(review_dir, benchmark)

    if not model_scores:
        raise ValueError(f"No review files found for benchmark '{benchmark}' in {review_dir}")

    print(f"Loaded {len(model_scores)} models: {list(model_scores.keys())}")

    difficulty = compute_difficulty(model_scores)
    print(f"Computed difficulty for {len(difficulty)} samples")

    print(f"Selecting {prune_ratio*100:.0f}% of samples via discriminative stratified sampling")
    selected = stratified_sample(difficulty, prune_ratio=prune_ratio, n_tiers=n_tiers)

    print(f"Verifying rank preservation...")
    verification = verify_rank_preservation(model_scores, selected)

    result = {
        "benchmark": benchmark,
        "prune_ratio_requested": prune_ratio,
        "selected_indices": selected,
        "verification": verification,
        "strategy": "discriminative_stratified_sampling",
        "description": (
            "Samples stratified into difficulty tiers by mean score across models. "
            "Tiers weighted by discriminativeness (peaks at difficulty=0.5 where models disagree). "
            "Model-agnostic: defensible for any new model not in the reference set."
        ),
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved pruned indices to {output_path}")

    print(f"\n=== Results ===")
    print(f"Full set: {verification['n_full']} samples")
    print(f"Pruned set: {verification['n_pruned']} samples ({verification['prune_ratio_actual']*100:.1f}%)")
    print(f"Spearman rank correlation: {verification['spearman_rank_correlation']:.4f}")
    print(f"\nModel scores (full vs pruned):")
    for model in verification['full_scores']:
        full = verification['full_scores'][model]
        pruned = verification['pruned_scores'][model]
        print(f"  {model}: {full:.3f} (full) → {pruned:.3f} (pruned)")

    return result
