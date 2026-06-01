# Copyright (c) Cerebras challenge submission.
# Evalscope-native adapter for discriminative stratified pruning.
#
# Extension point: overrides DefaultDataAdapter.sample_filter() which is
# called during dataset loading. This is the cleanest upstream extension
# point — no framework internals modified, follows evalscope conventions.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from evalscope.benchmarks.cerebras_pruner.pruner import prune_benchmark

logger = get_logger()


class PrunedAdapterMixin:
    """
    Mixin that adds discriminative stratified pruning to any DefaultDataAdapter.

    Usage:
        class PrunedAALCR(PrunedAdapterMixin, AALCRAdapter):
            pass

    The mixin overrides sample_filter() to include only pre-computed
    selected indices from the pruner.
    """

    #: Set of sample indices to keep — populated during __init__
    _selected_indices: Set[int] = set()

    def init_pruning(
        self,
        review_dir: str | Path,
        benchmark_prefix: str,
        prune_ratio: float = 0.3,
        output_path: Optional[str | Path] = None,
    ) -> None:
        """
        Run the pruner and store selected indices.

        Call this after super().__init__() in the subclass.
        """
        logger.info(f'Running discriminative stratified pruner for {benchmark_prefix}')

        result = prune_benchmark(
            review_dir=review_dir,
            benchmark=benchmark_prefix,
            prune_ratio=prune_ratio,
            output_path=output_path,
        )

        self._selected_indices = set(result['selected_indices'])

        verification = result['verification']
        logger.info(
            f'Pruning complete: {verification["n_full"]} → {verification["n_pruned"]} samples '
            f'(Spearman={verification["spearman_rank_correlation"]:.4f})'
        )

    def sample_filter(self, sample: Sample) -> bool:
        """
        Override DefaultDataAdapter.sample_filter to include only pruned indices.

        If no indices have been loaded (pruning not initialized), all samples pass.
        This maintains backward compatibility.
        """
        if not self._selected_indices:
            return True

        sample_id = getattr(sample, 'id', None) or getattr(sample, 'index', None)

        if sample_id is None:
            # Fall back to metadata
            sample_id = (
                sample.metadata.get('index') or
                sample.metadata.get('sample_id') or
                sample.metadata.get('id')
            ) if sample.metadata else None

        if sample_id is None:
            return True

        return int(sample_id) in self._selected_indices


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## AA-LCR with Discriminative Stratified Pruning

Pruned version of AA-LCR using the Cerebras discriminative stratified sampling strategy.

Selects samples where models disagree (difficulty ≈ 0.33 or 0.67) which carry
the maximum ranking signal. Preserves model ranking from the full benchmark
(Spearman rank correlation = 1.0 at 30% sample size).

### Extra Parameters
- `review_dir`: path to directory containing review jsonl files
- `prune_ratio`: fraction of samples to keep (default 0.3)
- `output_path`: optional path to save selected indices as JSON
""",
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        extra_params={
            'review_dir': {
                'type': 'str',
                'description': 'Path to directory containing review jsonl files',
                'value': None,
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep (default 0.3)',
                'value': 0.3,
            },
            'output_path': {
                'type': 'str | null',
                'description': 'Optional path to save selected indices as JSON',
                'value': None,
            },
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory containing extracted AA-LCR text files',
                'value': None,
            },
        }
    )
)
class PrunedAALCRAdapter(PrunedAdapterMixin, AALCRAdapter):
    """
    AA-LCR adapter with discriminative stratified pruning.

    Extends AALCRAdapter by overriding sample_filter() to include
    only the highest-signal samples selected by the pruner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        review_dir = self.extra_params.get('review_dir')
        prune_ratio = float(self.extra_params.get('prune_ratio', 0.3))
        output_path = self.extra_params.get('output_path')

        if review_dir:
            self.init_pruning(
                review_dir=review_dir,
                benchmark_prefix='aa_lcr',
                prune_ratio=prune_ratio,
                output_path=output_path,
            )
        else:
            logger.warning(
                'aa_lcr_pruned: no review_dir provided — running without pruning. '
                'Pass extra_params["review_dir"] to enable pruning.'
            )


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='LiveCodeBench v5 (Pruned)',
        tags=[Tags.CODING],
        description="""
## LiveCodeBench v5 with Discriminative Stratified Pruning

Pruned version of LiveCodeBench v5 using the Cerebras discriminative stratified sampling strategy.

Selects samples where models disagree which carry the maximum ranking signal.
Preserves model ranking from the full benchmark (Spearman rank correlation = 1.0
at 30% sample size).

### Extra Parameters
- `review_dir`: path to directory containing review jsonl files
- `prune_ratio`: fraction of samples to keep (default 0.3)
- `output_path`: optional path to save selected indices as JSON
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        metric_list=['acc'],
        eval_split='test',
        extra_params={
            'review_dir': {
                'type': 'str',
                'description': 'Path to directory containing review jsonl files',
                'value': None,
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep (default 0.3)',
                'value': 0.3,
            },
            'output_path': {
                'type': 'str | null',
                'description': 'Optional path to save selected indices as JSON',
                'value': None,
            },
        }
    )
)
class PrunedLiveCodeBenchAdapter(PrunedAdapterMixin, LiveCodeBenchAdapter):
    """
    LiveCodeBench adapter with discriminative stratified pruning.

    Extends LiveCodeBenchAdapter by overriding sample_filter() to include
    only the highest-signal samples selected by the pruner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        review_dir = self.extra_params.get('review_dir')
        prune_ratio = float(self.extra_params.get('prune_ratio', 0.3))
        output_path = self.extra_params.get('output_path')

        if review_dir:
            self.init_pruning(
                review_dir=review_dir,
                benchmark_prefix='live_code_bench_v5',
                prune_ratio=prune_ratio,
                output_path=output_path,
            )
        else:
            logger.warning(
                'live_code_bench_pruned: no review_dir provided — running without pruning.'
            )
