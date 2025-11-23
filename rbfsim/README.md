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
- **Sigma (σ)**: 0.7
- **Number of simulations**: 1000

## Grader Types Simulated

1. **Accuracy-based graders**:
   - Grader 1: 80% known (answers 80% correctly, guesses on remaining 20%)
   - Grader 2: 70% known (answers 70% correctly, guesses on remaining 30%)
   - Grader 3: 60% known (answers 60% correctly, guesses on remaining 40%)

   **Note**: These graders have higher actual accuracy than the label suggests because random guessing
   sometimes produces correct answers by chance. The probability of a correct random guess is 23.5%
   (based on the ground truth distribution). Therefore:
   - 80% known → 84.7% actual accuracy (80% + 20%×23.5%)
   - 70% known → 77.0% actual accuracy (70% + 30%×23.5%)
   - 60% known → 69.4% actual accuracy (60% + 40%×23.5%)

   This more realistically models human behavior: when uncertain, people guess rather than
   deliberately choosing wrong answers.

2. **Correlation-based graders**:
   - Grader 4: 70% Spearman correlation
   - Grader 5: 60% Spearman correlation

## Key Results

### Summary Statistics (σ=0.7)

| Grader Type | RBF Score (Mean ± SD) | Accuracy (Mean ± SD) | Spearman Corr (Mean ± SD) | Normalized RBF % |
|-------------|----------------------|---------------------|--------------------------|------------------|
| accuracy_80pct | 87.61 ± 1.70 | 84.7% ± 1.9% | 0.799 ± 0.053 | 87.6% |
| accuracy_70pct | 81.32 ± 2.06 | 77.0% ± 2.3% | 0.699 ± 0.063 | 81.3% |
| accuracy_60pct | 75.18 ± 2.34 | 69.4% ± 2.6% | 0.600 ± 0.073 | 75.2% |
| correlation_70pct | 58.32 ± 3.78 | 41.4% ± 5.0% | 0.629 ± 0.059 | 58.3% |
| correlation_60pct | 53.61 ± 3.81 | 36.8% ± 4.7% | 0.527 ± 0.072 | 53.6% |

### Key Findings

1. **Balanced Partial Credit** (with σ=0.7):
   - Being off by 1 point gives **36.0% credit** (meaningful partial credit)
   - Being off by 2 points gives **1.7% credit** (small partial credit)
   - Being off by 3+ points gives essentially 0% credit

2. **Accuracy-based graders**:
   - 80% known (84.7% actual) → RBF score ≈ 87.6 (87.6% of max)
   - 70% known (77.0% actual) → RBF score ≈ 81.3 (81.3% of max)
   - 60% known (69.4% actual) → RBF score ≈ 75.2 (75.2% of max)
   - RBF scores are higher than actual accuracy due to meaningful partial credit for near-misses

3. **Correlation-based graders**:
   - 70% correlation → only 41.4% accuracy, but RBF ≈ 58.3
   - 60% correlation → only 36.8% accuracy, but RBF ≈ 53.6
   - **Important**: High correlation does NOT imply high accuracy!
   - However, correlation-based graders benefit significantly from partial credit (they tend to be "close")

4. **RBF Kernel with σ=0.7 provides BALANCED scoring**:
   - Error of 0 (exact match): 1.000 score (100%)
   - Error of 1 (off by one): 0.360 score (36%)
   - Error of 2 (off by two): 0.017 score (1.7%)
   - Error of 3+: essentially 0.000 score
   - This appropriately rewards "close" answers on Likert scales

## Interpreting "Good" Scores

Based on the simulation with σ=0.7:

- **Excellent**: RBF score ≥ 87 (equivalent to ~80% known/~85% actual accuracy)
- **Good**: RBF score ≥ 81 (equivalent to ~70% known/~77% actual accuracy)
- **Acceptable**: RBF score ≥ 75 (equivalent to ~60% known/~69% actual accuracy)
- **Needs Improvement**: RBF score < 75 (equivalent to <60% known/<69% actual accuracy)

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
