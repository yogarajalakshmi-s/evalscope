# Copyright (c) Cerebras challenge submission.
# MMMU Multimodal Probe — image encoder stress testing.
#
# Goal: select samples from MMMU that stress the IMAGE ENCODER specifically,
# not general language understanding. This detects encoder degradation
# before it shows up in random benchmark sampling.
#
# Strategy:
# 1. Prioritize image types that stress the encoder:
#    - Fine-grained technical: Chemical Structures, Technical Blueprints,
#      Microscopic Images, Medical Images, Body Scans
#    - Spatial reasoning: Diagrams, Plots and Charts, Geometric Shapes
#    - OCR-heavy: Screenshots, Mathematical Notations
# 2. Stratify by subject for domain coverage (all 22 subjects)
# 3. Prioritize Hard > Medium > Easy difficulty
# 4. Result: a probe set that surfaces encoder failures specifically

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Image types ranked by encoder stress (higher = more encoder-dependent)
ENCODER_STRESS_SCORES: Dict[str, float] = {
    # Highest stress — fine-grained technical detail required
    'Chemical Structures': 1.0,
    'Technical Blueprints': 1.0,
    'Microscopic Images': 1.0,
    'Mathematical Notations': 0.95,
    'Body Scans: MRI, CT scans, and X-rays': 0.95,
    'Pathological Images': 0.95,
    'Medical Images': 0.9,

    # High stress — spatial/structural reasoning
    'Geometric Shapes': 0.85,
    'Diagrams': 0.85,
    'Trees and Graphs': 0.85,
    'Plots and Charts': 0.8,
    'Maps': 0.8,
    'Screenshots': 0.75,

    # Medium stress — some visual detail needed
    'Tables': 0.6,
    'Icons and Symbols': 0.6,
    'Comics and Cartoons': 0.5,
    'Sketches and Drafts': 0.5,

    # Low stress — language model can compensate
    'Photographs': 0.3,
    'Paintings': 0.2,
    'Portraits': 0.15,
    'Sculpture': 0.1,
    'Landscapes': 0.1,
    'Other': 0.3,
}

DIFFICULTY_SCORES = {
    'Hard': 1.0,
    'Medium': 0.6,
    'Easy': 0.3,
}


def compute_encoder_stress(img_type_str: str) -> float:
    """
    Compute encoder stress score for a sample based on its image type(s).
    img_type is stored as a string representation of a list, e.g. "['Diagrams']"
    """
    try:
        img_types = json.loads(img_type_str.replace("'", '"'))
        if not isinstance(img_types, list):
            img_types = [img_type_str]
    except Exception:
        img_types = [img_type_str]

    if not img_types:
        return 0.3

    # Take max stress across all image types in the sample
    scores = [ENCODER_STRESS_SCORES.get(t.strip(), 0.3) for t in img_types]
    return max(scores)


def load_mmmu_samples(
    pred_dir: Path,
    review_dir: Path,
) -> List[Dict]:
    samples = []

    for pred_file in sorted(pred_dir.glob('mmmu_*.jsonl')):
        subject = pred_file.stem.replace('mmmu_', '')
        review_file = review_dir / pred_file.name

        pred_data: Dict[int, Dict] = {}
        with open(pred_file) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                idx = d['index']
                pred_data[idx] = d.get('metadata', {})

        review_data: Dict[int, float] = {}
        if review_file.exists():
            with open(review_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    idx = d['index']
                    score_val = d.get('sample_score', {}).get('score', {}).get('value', {})
                    score = float(score_val.get('acc', 0.0)) if isinstance(score_val, dict) else 0.0
                    review_data[idx] = score

        for idx, meta in pred_data.items():
            img_type = meta.get('img_type', 'Other')
            difficulty = meta.get('topic_difficulty', 'Medium')
            score = review_data.get(idx, 0.0)
            encoder_stress = compute_encoder_stress(str(img_type))
            diff_score = DIFFICULTY_SCORES.get(difficulty, 0.6)

            samples.append({
                'index': idx,
                'subject': subject,
                # Composite unique key — indices repeat across subjects
                'uid': f'{subject}:{idx}',
                'img_type': img_type,
                'difficulty': difficulty,
                'score': score,
                'encoder_stress': encoder_stress,
                'difficulty_score': diff_score,
                'priority': encoder_stress * 0.7 + diff_score * 0.3,
            })

    return samples


def select_mmmu_probe(
    samples: List[Dict],
    prune_ratio: float = 0.15,
    min_per_subject: int = 2,
) -> List[str]:
    """Returns list of uid strings like 'Accounting:22'"""
    total = len(samples)
    target_n = max(1, math.ceil(total * prune_ratio))

    by_subject: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_subject[s['subject']].append(s)

    for subj in by_subject:
        by_subject[subj].sort(key=lambda x: x['priority'], reverse=True)

    selected_uids = set()

    # Step 1: min per subject
    for subj, subj_samples in by_subject.items():
        take = min(min_per_subject, len(subj_samples))
        for s in subj_samples[:take]:
            selected_uids.add(s['uid'])

    # Step 2: fill remaining by priority
    remaining = target_n - len(selected_uids)
    if remaining > 0:
        candidates = [
            s for s in sorted(samples, key=lambda x: x['priority'], reverse=True)
            if s['uid'] not in selected_uids
        ]
        for s in candidates[:remaining]:
            selected_uids.add(s['uid'])

    return sorted(selected_uids)


def verify_encoder_coverage(
    samples: List[Dict],
    selected_uids: List[str],
) -> Dict:
    """
    Verify the probe covers high-stress encoder types.
    """
    selected_set = set(selected_uids)  # now a set of uid strings
    selected = [s for s in samples if s['uid'] in selected_set]

    # Distribution of encoder stress in selected vs full
    full_avg_stress = sum(s['encoder_stress'] for s in samples) / len(samples)
    probe_avg_stress = sum(s['encoder_stress'] for s in selected) / len(selected) if selected else 0

    # Subject coverage
    full_subjects = set(s['subject'] for s in samples)
    probe_subjects = set(s['subject'] for s in selected)

    # High-stress sample counts
    high_stress_threshold = 0.8
    full_high_stress = sum(1 for s in samples if s['encoder_stress'] >= high_stress_threshold)
    probe_high_stress = sum(1 for s in selected if s['encoder_stress'] >= high_stress_threshold)

    return {
        'n_full': len(samples),
        'n_probe': len(selected),
        'prune_ratio_actual': len(selected) / len(samples),
        'full_avg_encoder_stress': round(full_avg_stress, 3),
        'probe_avg_encoder_stress': round(probe_avg_stress, 3),
        'encoder_stress_uplift': round(probe_avg_stress - full_avg_stress, 3),
        'full_subject_coverage': len(full_subjects),
        'probe_subject_coverage': len(probe_subjects),
        'full_high_stress_samples': full_high_stress,
        'probe_high_stress_samples': probe_high_stress,
        'high_stress_concentration': round(probe_high_stress / len(selected), 3) if selected else 0,
    }


def prune_mmmu(
    pred_dir: str | Path,
    review_dir: str | Path,
    prune_ratio: float = 0.15,
    min_per_subject: int = 2,
    output_path: Optional[str | Path] = None,
) -> Dict:
    """
    Main entry point: build MMMU encoder stress probe.

    Args:
        pred_dir: directory containing prediction jsonl files per subject
        review_dir: directory containing review jsonl files per subject
        prune_ratio: fraction of samples to keep (default 0.15 = 15%)
        min_per_subject: minimum samples per subject for coverage
        output_path: optional path to save results as JSON

    Returns:
        dict with selected indices, verification stats, and metadata
    """
    pred_dir = Path(pred_dir)
    review_dir = Path(review_dir)

    print(f'Loading MMMU samples from {pred_dir}')
    samples = load_mmmu_samples(pred_dir, review_dir)
    print(f'Loaded {len(samples)} samples across {len(set(s["subject"] for s in samples))} subjects')

    print(f'Selecting encoder stress probe ({prune_ratio*100:.0f}% of samples)')
    selected = select_mmmu_probe(samples, prune_ratio=prune_ratio, min_per_subject=min_per_subject)

    print('Verifying encoder coverage...')
    verification = verify_encoder_coverage(samples, selected)

    result = {
        'benchmark': 'mmmu',
        'prune_ratio_requested': prune_ratio,
        'selected_uids': selected,
        'selected_indices': selected,
        'verification': verification,
        'strategy': 'encoder_stress_stratified_sampling',
        'description': (
            'Samples selected to maximize image encoder stress. '
            'Prioritizes fine-grained technical images (Chemical Structures, '
            'Technical Blueprints, Microscopic Images, Medical Images) over '
            'photographs and paintings. Ensures all 22 subjects covered. '
            'Detects encoder degradation that random sampling would miss.'
        ),
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Saved probe indices to {output_path}')

    print(f'\n=== MMMU Encoder Stress Probe Results ===')
    v = verification
    print(f'Full set: {v["n_full"]} samples across {v["full_subject_coverage"]} subjects')
    print(f'Probe set: {v["n_probe"]} samples across {v["probe_subject_coverage"]} subjects')
    print(f'Reduction: {(1-v["prune_ratio_actual"])*100:.1f}%')
    print(f'Avg encoder stress — full: {v["full_avg_encoder_stress"]:.3f}, probe: {v["probe_avg_encoder_stress"]:.3f}')
    print(f'Encoder stress uplift: +{v["encoder_stress_uplift"]:.3f}')
    print(f'High-stress samples in probe: {v["probe_high_stress_samples"]}/{v["n_probe"]} ({v["high_stress_concentration"]*100:.1f}%)')

    return result
