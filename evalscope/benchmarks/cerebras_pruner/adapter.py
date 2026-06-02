# Copyright (c) Cerebras challenge submission.
# Evalscope-native adapter for discriminative stratified pruning.
#
# Extension point: overrides DefaultDataAdapter.sample_filter() which is
# called during dataset loading. This is the cleanest upstream extension
# point — no framework internals modified, follows evalscope conventions.
#
# The PrunedAdapterMixin is UNIVERSAL — it works with any DefaultDataAdapter
# subclass. Use make_pruned_adapter() to wrap any registered benchmark.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

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
    Universal mixin that adds discriminative stratified pruning to ANY DefaultDataAdapter.

    Works with any benchmark — not hardcoded to aa_lcr or live_code_bench.

    Usage:
        # Option 1: Direct subclassing
        class PrunedAALCR(PrunedAdapterMixin, AALCRAdapter):
            pass

        # Option 2: Dynamic factory (preferred for universal use)
        PrunedAny = make_pruned_adapter(SomeBenchmarkAdapter)

    The mixin overrides sample_filter() to include only pre-computed
    selected indices from the pruner.
    """

    _selected_indices: Set[int]

    def __init__(self, *args, **kwargs):
        # Initialize per-instance set to avoid class-level sharing
        self._selected_indices = set()
        super().__init__(*args, **kwargs)

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

        If no indices loaded (pruning not initialized), all samples pass.
        Maintains backward compatibility.
        """
        if not self._selected_indices:
            return True

        sample_id = getattr(sample, 'id', None) or getattr(sample, 'index', None)

        if sample_id is None:
            sample_id = (
                sample.metadata.get('index') or
                sample.metadata.get('sample_id') or
                sample.metadata.get('id')
            ) if sample.metadata else None

        if sample_id is None:
            return True

        return int(sample_id) in self._selected_indices


def make_pruned_adapter(
    base_adapter_cls: Type[DefaultDataAdapter],
    benchmark_prefix: str,
    pruned_name: Optional[str] = None,
    pruned_pretty_name: Optional[str] = None,
) -> Type[DefaultDataAdapter]:
    """
    Universal factory: wrap ANY DefaultDataAdapter with discriminative stratified pruning.

    This is the key to making the pruner universal — it works with any benchmark
    without hardcoding a new class for each one.

    Args:
        base_adapter_cls: Any DefaultDataAdapter subclass to wrap
        benchmark_prefix: Prefix for review files (e.g. 'aa_lcr', 'live_code_bench_v5')
        pruned_name: Optional name for the pruned benchmark (default: '{base_name}_pruned')
        pruned_pretty_name: Optional display name

    Returns:
        A new adapter class with pruning applied

    Example:
        # Wrap AA-LCR
        PrunedAALCR = make_pruned_adapter(AALCRAdapter, 'aa_lcr')

        # Wrap any new benchmark
        PrunedNewBench = make_pruned_adapter(NewBenchAdapter, 'new_bench')
    """

    base_name = getattr(base_adapter_cls, '__name__', 'UnknownAdapter')
    class_name = f'Pruned{base_name}'

    class PrunedAdapter(PrunedAdapterMixin, base_adapter_cls):  # type: ignore

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            review_dir = self.extra_params.get('review_dir')
            prune_ratio = float(self.extra_params.get('prune_ratio', 0.3))
            output_path = self.extra_params.get('output_path')

            if review_dir:
                self.init_pruning(
                    review_dir=review_dir,
                    benchmark_prefix=benchmark_prefix,
                    prune_ratio=prune_ratio,
                    output_path=output_path,
                )
            else:
                logger.warning(
                    f'{class_name}: no review_dir provided — running without pruning. '
                    f'Pass extra_params["review_dir"] to enable pruning.'
                )

    PrunedAdapter.__name__ = class_name
    PrunedAdapter.__qualname__ = class_name
    return PrunedAdapter


# --- Concrete registrations using the universal factory ---

# Shared extra params for all pruned benchmarks
_PRUNING_EXTRA_PARAMS = {
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


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## AA-LCR with Discriminative Stratified Pruning

Pruned via universal PrunedAdapterMixin — works across any benchmark.
Spearman rank correlation = 1.0 at 30% sample size.
""",
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        extra_params={
            **_PRUNING_EXTRA_PARAMS,
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory containing extracted AA-LCR text files',
                'value': None,
            },
        }
    )
)
class PrunedAALCRAdapter(PrunedAdapterMixin, AALCRAdapter):
    """AA-LCR with universal discriminative stratified pruning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        review_dir = self.extra_params.get('review_dir')
        if review_dir:
            self.init_pruning(
                review_dir=review_dir,
                benchmark_prefix='aa_lcr',
                prune_ratio=float(self.extra_params.get('prune_ratio', 0.3)),
                output_path=self.extra_params.get('output_path'),
            )


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='LiveCodeBench v5 (Pruned)',
        tags=[Tags.CODING],
        description="""
## LiveCodeBench v5 with Discriminative Stratified Pruning

Pruned via universal PrunedAdapterMixin — works across any benchmark.
Spearman rank correlation = 1.0 at 30% sample size.
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        metric_list=['acc'],
        eval_split='test',
        extra_params=_PRUNING_EXTRA_PARAMS,
    )
)
class PrunedLiveCodeBenchAdapter(PrunedAdapterMixin, LiveCodeBenchAdapter):
    """LiveCodeBench v5 with universal discriminative stratified pruning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        review_dir = self.extra_params.get('review_dir')
        if review_dir:
            self.init_pruning(
                review_dir=review_dir,
                benchmark_prefix='live_code_bench_v5',
                prune_ratio=float(self.extra_params.get('prune_ratio', 0.3)),
                output_path=self.extra_params.get('output_path'),
            )
