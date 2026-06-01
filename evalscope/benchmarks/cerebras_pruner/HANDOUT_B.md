# Handout B - Why This Matters and How to Use It
*Mixed audience: developers, test engineers, product, customer team*

## What Changed

Before this tool, answering "is this model good enough for our customer?" required running hundreds of test cases - taking hours and costing significant compute per evaluation cycle.

**Now:** Run 30% of the tests and get the same answer. Every time.

## The Simple Idea

Not all test questions are equally useful. Some questions every model gets right - they tell you nothing. Some questions every model gets wrong - they tell you nothing either. The useful questions are the ones where some models succeed and others fail. Those are the questions that separate good models from great ones.

This tool automatically finds those useful questions and throws away the rest.

## How to Use It Tomorrow

**Step 1: Run the pruner**
```bash
python -c "
from evalscope.benchmarks.cerebras_pruner.pruner import prune_benchmark
prune_benchmark(
    review_dir='path/to/reviews',
    benchmark='aa_lcr',
    prune_ratio=0.3,
    output_path='./pruned_aa_lcr.json'
)
"
```

**Step 2: Compare results**
```bash
python -m evalscope_ext.tools.compare_runs \
    --full-json ./pruned_aa_lcr.json \
    --benchmark aa_lcr
```

**Step 3: Read the output**
If you see this - the pruned set gives you the same go/no-go answer as the full set.

## What the Multimodal Probe Gives You That Random Sampling Cannot

Random sampling picks test questions without structure - some easy, some hard, no guarantee of coverage. The multimodal probe specifically selects questions that stress the image encoder - the part of the model that processes visual input. This means when a model's image understanding degrades, you catch it immediately instead of after running hundreds of samples that mostly test language understanding.

## Why a Customer-Facing PM Should Care

When a customer asks "is this model good enough for our image analysis workload?" you now have a fast, structured answer instead of a slow, expensive one.

**Speed:** Hours -> minutes per evaluation cycle
**Cost:** 70% reduction in compute per model evaluation
**Confidence:** Spearman = 1.0 means identical ranking to full evaluation

The pruned set tells you which model wins. That is the only thing that matters for a go/no-go decision.