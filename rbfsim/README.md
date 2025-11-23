# RBF (Gaussian Similarity) Scoring Simulation for Likert Scale Assessment

## Overview

This simulation explores a **principled scoring system** for Likert scale assessments that accounts for **uncertainty in the ground truth itself**. When multiple experts disagree on the "correct" answer, we shouldn't heavily penalize examinees for being close.

## Problem Statement

**Traditional approach**: Binary 0-1 loss - either the answer is exactly correct (1 point) or wrong (0 points).
- **Issue**: Assumes ground truth is perfectly certain
- **Issue**: Doesn't account for inter-rater disagreement among experts

**RBF approach**: Models ground truth as having uncertainty (variance σ²):

```
Score = exp(-|examinee_answer - ground_truth|² / (2σ²))
```

This is the **unnormalized Gaussian likelihood** of the examinee's answer under N(ground_truth, σ).

**Interpretation**:
- σ represents the **standard deviation of expert disagreement**
- If 3 experts give [4, 3, 3] → mean=3.33, SD≈0.5, so use σ=0.5
- The score reflects how consistent the examinee's answer is with the uncertain ground truth

**Question**: What constitutes a "good" RBF score given ground truth uncertainty σ?

## Simulation Setup

**Current parameters** (4-point scale with moderate ground truth uncertainty):

- **Number of questions**: 100
- **Scale**: 1-4 (4-point Likert scale)
- **Ground truth distribution**:
  - 1: 10%
  - 2: 25%
  - 3: 40%
  - 4: 25%
- **Sigma (σ)**: 0.5
  - **Interpretation**: Ground truth has SD=0.5 (moderate inter-expert disagreement)
  - **Example**: If experts give [3, 3, 4], SD≈0.47≈0.5
  - **Rationale**: Should be estimated from actual inter-rater reliability studies
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

### Summary Statistics (4-point scale, σ=0.5)

| Grader Type | RBF Score (Mean ± SD) | Accuracy (Mean ± SD) | Spearman Corr (Mean ± SD) | Normalized RBF % |
|-------------|----------------------|---------------------|--------------------------|------------------|
| accuracy_80pct | 87.3 ± 1.9 | 86.1% ± 2.0% | 0.801 ± 0.052 | 87.3% |
| accuracy_70pct | 80.6 ± 2.3 | 78.8% ± 2.5% | 0.698 ± 0.065 | 80.6% |
| accuracy_60pct | 74.3 ± 2.8 | 71.8% ± 3.0% | 0.596 ± 0.076 | 74.3% |
| correlation_70pct | 56.4 ± 4.7 | 50.4% ± 5.3% | 0.603 ± 0.061 | 56.4% |
| correlation_60pct | 51.2 ± 4.6 | 44.9% ± 5.1% | 0.502 ± 0.073 | 51.2% |

### Key Findings

1. **Partial Credit reflects Ground Truth Uncertainty** (with σ=0.5):
   - Error of 0 (exact match): 1.000 score (100% - examinee matches GT exactly)
   - Error of 1 (off by one): **0.135 score (13.5%)** - within 2σ of GT distribution
   - Error of 2 (off by two): 0.003 score (0.3%) - outside reasonable GT variance
   - Error of 3 (opposite end): 0.000 score (0%) - incompatible with GT distribution

   **Interpretation**: With σ=0.5, being off by 1 unit gets modest credit because experts
   themselves might disagree by ±1 unit. Being off by 2+ units is outside the range of
   reasonable expert disagreement and gets essentially no credit.

2. **Accuracy-based graders**:
   - 80% known (86.1% actual) → RBF score ≈ 87.3 (87.3% of max)
   - 70% known (78.8% actual) → RBF score ≈ 80.6 (80.6% of max)
   - 60% known (71.8% actual) → RBF score ≈ 74.3 (74.3% of max)
   - RBF scores slightly higher than actual accuracy due to partial credit for answers
     within the ground truth uncertainty range

3. **Correlation-based graders**:
   - 70% correlation → 50.4% accuracy, RBF ≈ 56.4
   - 60% correlation → 44.9% accuracy, RBF ≈ 51.2
   - **Important**: High correlation does NOT imply high accuracy!
   - Correlation-based graders benefit moderately from partial credit (they tend to be
     "close" but not exact)

4. **Ground Truth Uncertainty Principle**:
   - σ=0.5 represents moderate expert disagreement
   - Examinee answers within ±1σ of ground truth receive partial credit
   - This is **fair**: if experts disagree, examinees shouldn't be heavily penalized
     for falling within the range of expert responses
   - σ should be empirically estimated from inter-rater reliability studies, not arbitrarily chosen

## Interpreting "Good" Scores

Based on the simulation with σ=0.5 (4-point scale):

- **Excellent**: RBF score ≥ 87 (equivalent to ~80% known/~86% actual accuracy)
- **Good**: RBF score ≥ 81 (equivalent to ~70% known/~79% actual accuracy)
- **Acceptable**: RBF score ≥ 74 (equivalent to ~60% known/~72% actual accuracy)
- **Needs Improvement**: RBF score < 74

## Understanding Sigma (σ): Ground Truth Uncertainty

**σ is NOT a tuning parameter** - it should reflect actual measurement uncertainty.

### How to Determine Sigma

1. **Empirical approach** (recommended):
   - Collect ratings from multiple independent experts on the same questions
   - For each question, calculate SD of expert ratings
   - Average these SDs across questions → this is your σ
   - Example: If experts consistently disagree by ±0.5 points, use σ=0.5

2. **Theoretical approach**:
   - Consider the scale granularity and expected inter-rater reliability
   - Higher inter-rater reliability → smaller σ
   - For Likert scales, typical values: σ ∈ [0.3, 0.7]

### Effect of Sigma on Scoring

The value of σ reflects **how uncertain the ground truth is**:

- **Small σ (e.g., 0.1-0.3)**: High expert agreement
  - Experts strongly agree on correct answers
  - Being off by 1 unit is penalized heavily (it's outside expert variation)
  - Use when inter-rater reliability is very high (Krippendorff's α > 0.9)

- **Medium σ (e.g., 0.4-0.7)**: Moderate expert disagreement
  - Experts sometimes disagree by ±1 unit
  - Being off by 1 unit receives modest partial credit
  - Use when inter-rater reliability is moderate (Krippendorff's α ≈ 0.7-0.9)
  - **Current choice: σ=0.5** assumes moderate uncertainty

- **Large σ (e.g., >0.8)**: Low expert agreement
  - Experts frequently disagree significantly
  - Being off by 1-2 units receives substantial credit
  - Use when inter-rater reliability is low (Krippendorff's α < 0.7)
  - May indicate the questions are ambiguous and need revision

### Key Principle

**Don't arbitrarily choose σ to make scores "look good"**. Estimate it from actual expert
disagreement. If σ is high, it means your assessment has measurement problems that should
be addressed by improving question clarity, not by manipulating the scoring function.

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
