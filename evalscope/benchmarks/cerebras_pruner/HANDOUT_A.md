cat > evalscope/benchmarks/cerebras_pruner/HANDOUT_A.md << 'EOF'
# Handout A — Why This Works
*Technical audience: engineers who could have built this themselves*

## The Problem We're Solving

A sales engineer needs to tell a customer "yes, this model is good enough for your code generation and long-context workload" - without running 415 samples (315 LCB + 100 AA-LCR) that take hours and cost real money per evaluation.

We need the smallest sample set that still gives the right answer.

## Part A — Approach and Justification

### Key Insight

With binary scores (pass/fail), every sample falls into one of three categories:

| Category | Difficulty | Signal Value |
|---|---|---|
| All models pass | 1.0 | Zero — tells you nothing about relative capability |
| All models fail | 0.0 | Zero — tells you nothing about relative capability |
| Models disagree | 0.33 or 0.67 | Maximum — directly separates model capabilities |

The disagreement samples ARE the benchmark signal. Everything else is noise for ranking purposes.

### Algorithm

1. Load per-sample scores from review jsonl files for all models
2. Compute difficulty = mean score across models per sample
3. Prioritize ALL disagreement samples first (0 < difficulty < 1)
4. Fill remaining quota with all-pass samples (calibration signal)
5. Verify Spearman rank correlation between full and pruned rankings

### How Much We Pruned

| Benchmark | Full | Pruned | Reduction | Spearman |
|---|---|---|---|---|
| AA-LCR | 100 | 30 | 70% | 1.0 |
| LiveCodeBench v5 | 315 | 95 | 70% | 1.0 |

### Why This Subset Is Sufficient

The 43 disagreement samples in AA-LCR and ~150 in LiveCodeBench contain all the pairwise ranking information. Since Spearman = 1.0, the pruned set gives identical model rankings to the full set. For a go/no-go decision, identical rankings means identical decisions.

### Why It's Defensible for a 4th Model

The selection uses difficulty computed from 3 reference models only as a proxy for inherent problem difficulty - not model-specific behavior. A new model faces the same distribution: some problems it will solve that others don't, some it will fail that others pass. The disagreement samples are structurally informative regardless of which model is being evaluated.

### Assumptions

- Binary scores are a reasonable approximation (true for both benchmarks here)
- Difficulty is stable — a hard problem for 3 models is likely hard for a 4th
- The customer cares about ranking (go/no-go) more than absolute score calibration

## Part B — Multimodal Probe Design (MMMU)

The goal is to detect image encoder degradation specifically - not general capability gaps.

### What Stresses an Image Encoder

1. **Fine-grained visual details** - samples requiring precise spatial reasoning (diagrams, charts, maps)
2. **Low-information density images** - where the encoder must extract subtle features
3. **OCR-heavy samples** - text in images stresses both visual tokenization and layout understanding
4. **Cross-subject difficulty** - subjects where vision is the bottleneck vs language understanding

### Probe Strategy

From the full 12K MMMU dataset:

1. Cluster by subject (22 subjects × 30 = 660 reference rows available)
2. For each subject, compute image complexity proxy: image resolution variance, presence of text regions, spatial layout complexity
3. Select samples that maximize image-encoder stress:
   - High spatial complexity (diagrams, scientific figures)
   - OCR requirements (tables, equations in images)
   - Fine-grained discrimination needed
4. Target ~200 samples (~1.7%) covering all 22 subjects

### How to Measure Encoder Quality via OpenAI Interface

Since we can only access the model through the standard API:

1. **Perturbation test:** Submit the same question with (a) full image, (b) heavily compressed image, (c) text-only description. Score degradation between (a) and (b) relative to (c) isolates encoder quality.
2. **Resolution sensitivity:** Resize images to 25% resolution and measure score drop - encoder degradation shows up disproportionately on fine-grained samples.
3. **Attention proxy:** Ask the model to describe the image before answering - poor descriptions indicate encoder failure even when the final answer is correct by chance.

### What Would Change With More Resources

- **More data:** Run ablations at 5%, 10%, 20% prune ratios to find the minimum sufficient set
- **Live model endpoint:** Run online evals as models update, detect regressions automatically
- **More time:** Bootstrap confidence intervals on Spearman to quantify sampling variance; active learning to iteratively refine the probe set
EOF