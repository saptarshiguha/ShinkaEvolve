# RBF (Gaussian Similarity) Scoring Simulation for Likert Scale Assessment

## Overview

This simulation explores a scoring system for assessing performance on a Likert scale assessment (1-5 scale) using an RBF (Radial Basis Function) kernel instead of binary accuracy.

## Problem Statement

**Traditional approach**: Binary 0-1 loss - either the answer is exactly correct (1 point) or wrong (0 points).

**RBF approach**: Acknowledges that Likert scale responses can have natural variation. Uses Gaussian similarity:

```
Score = exp(-|prediction - ground_truth|² / (2σ²))
```

**Question**: What constitutes a "good" RBF score? If 80% accuracy is considered good in binary scoring, what's the equivalent RBF score?

## Simulation Setup

- **Number of questions**: 100
- **Scale**: 1-5 (5-point Likert scale)
- **Ground truth distribution**:
  - 1: 10%
  - 2: 15%
  - 3: 30%
  - 4: 30%
  - 5: 15%
- **Sigma (σ)**: 0.3
- **Number of simulations**: 1000

## Grader Types Simulated

1. **Accuracy-based graders**:
   - Grader 1: 80% accuracy
   - Grader 2: 70% accuracy
   - Grader 3: 60% accuracy

2. **Correlation-based graders**:
   - Grader 4: 70% Spearman correlation
   - Grader 5: 60% Spearman correlation

## Key Results

### Summary Statistics

| Grader Type | RBF Score (Mean ± SD) | Accuracy (Mean ± SD) | Spearman Corr (Mean ± SD) | Normalized RBF % |
|-------------|----------------------|---------------------|--------------------------|------------------|
| accuracy_80pct | 84.78 ± 1.94 | 84.8% ± 1.9% | 0.800 ± 0.055 | 84.8% |
| accuracy_70pct | 76.99 ± 2.39 | 76.9% ± 2.4% | 0.693 ± 0.066 | 77.0% |
| accuracy_60pct | 69.61 ± 2.71 | 69.5% ± 2.7% | 0.600 ± 0.076 | 69.6% |
| correlation_70pct | 41.83 ± 5.16 | 41.6% ± 5.2% | 0.631 ± 0.058 | 41.8% |
| correlation_60pct | 37.09 ± 4.78 | 36.9% ± 4.8% | 0.529 ± 0.067 | 37.1% |

### Key Findings

1. **RBF Score ≈ Accuracy** (with σ=0.3):
   - The RBF score is nearly identical to accuracy percentage
   - This is because σ=0.3 makes the kernel very strict
   - Being off by 1 point gives only 0.39% credit
   - Being off by 2+ points gives essentially 0% credit

2. **Accuracy-based graders**:
   - 80% accuracy → RBF score ≈ 84.8 (84.8% of max)
   - 70% accuracy → RBF score ≈ 77.0 (77.0% of max)
   - 60% accuracy → RBF score ≈ 69.6 (69.6% of max)
   - RBF scores are slightly higher than accuracy due to small partial credit for near-misses

3. **Correlation-based graders**:
   - 70% correlation → only 41.6% accuracy, RBF ≈ 41.8
   - 60% correlation → only 36.9% accuracy, RBF ≈ 37.1
   - **Important**: High correlation does NOT imply high accuracy!
   - A grader can have good rank ordering (correlation) but poor exact matching

4. **RBF Kernel with σ=0.3 is VERY STRICT**:
   - Error of 0 (exact match): 1.000 score
   - Error of 1 (off by one): 0.0039 score (0.39%)
   - Error of 2+: essentially 0.000 score
   - This makes RBF scoring very similar to binary accuracy

## Interpreting "Good" Scores

Based on the simulation with σ=0.3:

- **Excellent**: RBF score ≥ 85 (equivalent to ≥80% accuracy)
- **Good**: RBF score ≥ 77 (equivalent to ≥70% accuracy)
- **Acceptable**: RBF score ≥ 70 (equivalent to ≥60% accuracy)
- **Poor**: RBF score < 70 (equivalent to <60% accuracy)

## Effect of Sigma (σ)

The choice of σ dramatically affects the scoring:

- **Small σ (e.g., 0.1-0.3)**: Very strict, close to binary accuracy
  - Only gives significant credit for exact matches
  - Small errors are heavily penalized

- **Medium σ (e.g., 0.5-1.0)**: More lenient
  - Gives partial credit for near-misses
  - Error of 1 might give 30-60% credit

- **Large σ (e.g., >1.5)**: Very lenient
  - Substantial credit for answers within 1-2 points
  - May be too forgiving for assessment purposes

## Files Generated

1. `simulate_rbf_likert.py` - Main simulation script
2. `visualize_results.py` - Visualization script
3. `rbf_kernel_analysis.py` - Detailed kernel analysis
4. `simulation_results.csv` - Raw simulation data (1000 runs × 5 graders)
5. `summary_statistics.csv` - Summary statistics
6. `rbf_analysis.png` - Multi-panel visualization
7. `summary_table.png` - Summary table visualization
8. `rbf_kernel_analysis.png` - RBF kernel analysis

## Running the Simulation

```bash
# Run main simulation
python simulate_rbf_likert.py

# Generate visualizations
python visualize_results.py

# Analyze RBF kernel properties
python rbf_kernel_analysis.py
```

## Recommendations

1. **Current σ=0.3 is very strict**: Consider whether you want more lenient scoring
   - If Likert scale variation is expected, use larger σ (0.5-1.0)
   - If exact matches are critical, keep σ small (0.2-0.3)

2. **Correlation ≠ Accuracy**: Be cautious about using correlation as a proxy for performance
   - A grader with 70% correlation may only have 40% accuracy
   - Correlation measures rank ordering, not exact matching

3. **Benchmark interpretation**:
   - With current settings (σ=0.3), interpret RBF scores similarly to accuracy percentages
   - RBF score of 80 ≈ 80% accuracy

## Future Explorations

1. Test different σ values (0.5, 1.0, 1.5) to find optimal leniency
2. Allow different σ for different scale values (more uncertainty at extremes?)
3. Compare to other loss functions (ordinal regression, weighted accuracy)
4. Explore effect of ground truth distribution on scores
