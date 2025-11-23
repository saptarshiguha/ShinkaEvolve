"""
Demonstration of scale-aware sigma for RBF scoring.
Shows how sigma should scale with the range of the Likert scale.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

# Define the scale factor (tuned for 5-point scale with sigma=0.7)
SCALE_FACTOR = 0.175  # 0.7 / 4 = 0.175

def compute_rbf_score(error, sigma):
    """Compute RBF score for a given error and sigma."""
    return np.exp(-error**2 / (2 * sigma**2))

# Create comparison table
print("="*100)
print("SCALE-AWARE SIGMA DEMONSTRATION")
print("="*100)
print()
print("Current approach: Fixed σ = 0.7 for all scales")
print("Proposed approach: σ = 0.175 × (max - min)")
print()

scales = [
    (3, 1, 3, "3-point"),
    (5, 1, 5, "5-point"),
    (7, 1, 7, "7-point"),
    (10, 1, 10, "10-point (0-9 style)"),
]

# Fixed sigma comparison
print("="*100)
print("COMPARISON: Fixed σ=0.7 vs Scale-Aware σ")
print("="*100)
print()

for n_points, min_val, max_val, name in scales:
    scale_range = max_val - min_val
    max_error = scale_range

    # Fixed sigma
    fixed_sigma = 0.7

    # Scale-aware sigma
    adaptive_sigma = SCALE_FACTOR * scale_range

    print(f"\n{name.upper()} SCALE (range = {scale_range})")
    print("-"*100)
    print(f"  Fixed σ = {fixed_sigma:.3f}")
    print(f"  Scale-aware σ = {adaptive_sigma:.3f}")
    print()
    print(f"  {'Error':<10} {'Fixed σ Score':<20} {'Scale-aware σ Score':<25} {'Difference':<15}")
    print("-"*100)

    for error in range(max_error + 1):
        fixed_score = compute_rbf_score(error, fixed_sigma)
        adaptive_score = compute_rbf_score(error, adaptive_sigma)
        diff = adaptive_score - fixed_score

        print(f"  {error:<10} {fixed_score:>8.4f} ({fixed_score*100:>5.2f}%)   "
              f"{adaptive_score:>8.4f} ({adaptive_score*100:>5.2f}%)     "
              f"{diff:>+7.4f}")

    # Highlight the max error (opposite end)
    max_error_fixed = compute_rbf_score(max_error, fixed_sigma)
    max_error_adaptive = compute_rbf_score(max_error, adaptive_sigma)
    print()
    print(f"  → Max error ({max_error} units): Fixed gives {max_error_fixed*100:.3f}%, "
          f"Scale-aware gives {max_error_adaptive*100:.3f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Fixed σ vs Scale-Aware σ Comparison', fontsize=16, fontweight='bold')

# Plot 1: Comparison for 3-point scale
ax = axes[0, 0]
scale_range = 2
errors = np.linspace(0, scale_range, 100)
fixed_scores = compute_rbf_score(errors, 0.7)
adaptive_scores = compute_rbf_score(errors, SCALE_FACTOR * scale_range)

ax.plot(errors, fixed_scores, 'b-', linewidth=2, label='Fixed σ=0.7')
ax.plot(errors, adaptive_scores, 'r-', linewidth=2, label=f'Scale-aware σ={SCALE_FACTOR * scale_range:.2f}')
ax.set_xlabel('Error (units)', fontsize=11)
ax.set_ylabel('RBF Score', fontsize=11)
ax.set_title('3-Point Scale (range = 2)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axvline(scale_range, color='gray', linestyle='--', alpha=0.5, label='Max error')

# Add annotations for opposite end
ax.annotate(f'Opposite end:\nFixed: {compute_rbf_score(scale_range, 0.7)*100:.2f}%\nAdaptive: {compute_rbf_score(scale_range, SCALE_FACTOR * scale_range)*100:.4f}%',
           xy=(scale_range, compute_rbf_score(scale_range, 0.7)),
           xytext=(scale_range*0.5, 0.3),
           fontsize=9,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

# Plot 2: Comparison for 5-point scale
ax = axes[0, 1]
scale_range = 4
errors = np.linspace(0, scale_range, 100)
fixed_scores = compute_rbf_score(errors, 0.7)
adaptive_scores = compute_rbf_score(errors, SCALE_FACTOR * scale_range)

ax.plot(errors, fixed_scores, 'b-', linewidth=2, label='Fixed σ=0.7')
ax.plot(errors, adaptive_scores, 'r-', linewidth=2, label=f'Scale-aware σ={SCALE_FACTOR * scale_range:.2f}')
ax.set_xlabel('Error (units)', fontsize=11)
ax.set_ylabel('RBF Score', fontsize=11)
ax.set_title('5-Point Scale (range = 4) - Current Scale', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axvline(scale_range, color='gray', linestyle='--', alpha=0.5)

# Plot 3: Comparison for 7-point scale
ax = axes[1, 0]
scale_range = 6
errors = np.linspace(0, scale_range, 100)
fixed_scores = compute_rbf_score(errors, 0.7)
adaptive_scores = compute_rbf_score(errors, SCALE_FACTOR * scale_range)

ax.plot(errors, fixed_scores, 'b-', linewidth=2, label='Fixed σ=0.7')
ax.plot(errors, adaptive_scores, 'r-', linewidth=2, label=f'Scale-aware σ={SCALE_FACTOR * scale_range:.2f}')
ax.set_xlabel('Error (units)', fontsize=11)
ax.set_ylabel('RBF Score', fontsize=11)
ax.set_title('7-Point Scale (range = 6)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axvline(scale_range, color='gray', linestyle='--', alpha=0.5)

# Plot 4: Partial credit for error=1 across different scales
ax = axes[1, 1]
scale_sizes = np.arange(2, 11)  # 3-point to 11-point scales
fixed_error1_scores = [compute_rbf_score(1, 0.7) for _ in scale_sizes]
adaptive_error1_scores = [compute_rbf_score(1, SCALE_FACTOR * s) for s in scale_sizes]

x = scale_sizes + 1  # Convert range to number of points
ax.plot(x, [s*100 for s in fixed_error1_scores], 'bo-', linewidth=2, markersize=8, label='Fixed σ=0.7')
ax.plot(x, [s*100 for s in adaptive_error1_scores], 'ro-', linewidth=2, markersize=8, label='Scale-aware σ')
ax.set_xlabel('Number of Scale Points', fontsize=11)
ax.set_ylabel('Credit for Error=1 (%)', fontsize=11)
ax.set_title('Partial Credit for Being Off by 1', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(x)

plt.tight_layout()
plt.savefig('/home/user/ShinkaEvolve/rbfsim/scale_aware_sigma_comparison.png', dpi=300, bbox_inches='tight')
print("\n\nVisualization saved to: rbfsim/scale_aware_sigma_comparison.png")

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)
print()
print("With FIXED σ=0.7:")
print("  ✓ Being off by 1 always gives 36.0% credit (consistent across scales)")
print("  ✗ On smaller scales, opposite end still gets non-trivial credit")
print("    - 3-point: opposite end gets 1.7% (should be ~0%)")
print()
print("With SCALE-AWARE σ=0.175×range:")
print("  ✓ Opposite end always gets ~0% credit (consistent relative to scale)")
print("  ✗ Being off by 1 gives different credit on different scales")
print("    - 3-point: error=1 gets 1.7% (only 50% of scale)")
print("    - 5-point: error=1 gets 36.0% (only 25% of scale)")
print("    - 7-point: error=1 gets 71.2% (only 17% of scale)")
print()
print("RECOMMENDATION:")
print("  Use scale-aware σ if you want errors to be penalized RELATIVE to scale size.")
print("  The same absolute error (e.g., off by 1) is more serious on a 3-point scale")
print("  than on a 7-point scale, so it should be penalized more heavily.")
print()
print("="*100)
