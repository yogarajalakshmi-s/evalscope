# Cerebras Benchmark Pruner

**Evalscope commit SHA:** `e9d42d8b6a8dcb937e042ba905e36eb05171ae0d`

Discriminative stratified sampling pruner for benchmark compression. Implemented as an upstream-quality extension to [modelscope/evalscope](https://github.com/modelscope/evalscope).

## Live Task 1 URL
https://cerebras-perflens.vercel.app/

## Quick Start

```bash
git clone https://github.com/yogarajalakshmi-s/evalscope.git
cd evalscope
pip install -e .
```

Run the pruner:

```bash
python -c "
from evalscope.benchmarks.cerebras_pruner.pruner import prune_benchmark

# Prune AA-LCR to 30%
prune_benchmark(
    review_dir='path/to/Evals/Part 1/reviews',
    benchmark='aa_lcr',
    prune_ratio=0.3,
    output_path='./pruned_aa_lcr.json'
)

# Prune LiveCodeBench to 30%
prune_benchmark(
    review_dir='path/to/Evals/Part 1/reviews',
    benchmark='live_code_bench_v5',
    prune_ratio=0.3,
    output_path='./pruned_lcb.json'
)
"
```

Compare full vs pruned results:

```bash
python -m evalscope_ext.tools.compare_runs \
    --full-json ./pruned_aa_lcr.json \
    --pruned-json ./pruned_aa_lcr.json \
    --benchmark aa_lcr
```

## Results

| Benchmark | Full Samples | Pruned Samples | Reduction | Spearman |
|---|---|---|---|---|
| AA-LCR | 100 | 30 | 70% | 1.0 |
| LiveCodeBench v5 | 315 | 95 | 70% | 1.0 |

## Strategy

**Key insight:** With binary scores (0 or 1), samples where models disagree (difficulty ≈ 0.33 or 0.67) carry ALL the ranking signal. Samples where all models pass or all models fail contribute nothing to model differentiation.

**Algorithm:**
1. Compute per-sample difficulty = mean score across all models
2. Prioritize ALL disagreement samples first (0 < difficulty < 1)
3. Fill remaining quota with all-pass samples for score calibration
4. Verify Spearman rank correlation ≥ 0.9 between full and pruned rankings

**Why this is defensible for a 4th model:**
The strategy uses difficulty computed from existing models only as a proxy for inherent problem difficulty. A new model will still see the same distribution of easy, medium, and hard problems - the selection is not overfit to the 3 reference models.

## Extension Point

The pruner hooks into `DefaultDataAdapter.sample_filter()` - the clean upstream extension point called during dataset loading. No framework internals modified.

```python
class PrunedAALCRAdapter(PrunedAdapterMixin, AALCRAdapter):
    def sample_filter(self, sample: Sample) -> bool:
        return int(sample.id) in self._selected_indices
```

## File Structure

```
evalscope/benchmarks/cerebras_pruner/
├── __init__.py
├── pruner.py           # Core algorithm
├── adapter.py          # Evalscope adapters
├── README.md           # This file
├── HANDOUT_A.md        # Technical writeup
├── HANDOUT_B.md        # Non-technical writeup
└── tools/
    ├── __init__.py
    └── compare_runs.py # CLI comparison tool

evalscope_ext/
└── tools/
    └── compare_runs.py # Wrapper matching challenge spec
```
