"""
Deep dive analysis of the RBF kernel for Likert scale scoring.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# RBF kernel parameters
sigma = 0.3

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RBF Kernel Analysis for Likert Scale Scoring (σ=0.3)',
             fontsize=16, fontweight='bold')

# 1. RBF Kernel Shape
ax = axes[0, 0]
differences = np.linspace(0, 4, 1000)
similarities = np.exp(-differences**2 / (2 * sigma**2))

ax.plot(differences, similarities, 'b-', linewidth=2)
ax.fill_between(differences, similarities, alpha=0.3)
ax.set_xlabel('|Prediction - Ground Truth|', fontsize=12)
ax.set_ylabel('Similarity Score', fontsize=12)
ax.set_title('RBF Kernel Function', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect match (diff=0)')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% similarity')

# Mark specific differences (integers for Likert scale)
for diff in range(5):
    score = np.exp(-diff**2 / (2 * sigma**2))
    ax.plot(diff, score, 'ro', markersize=8)
    ax.annotate(f'Δ={diff}\n{score:.3f}',
               xy=(diff, score),
               xytext=(diff, score + 0.15),
               ha='center',
               fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

ax.legend(fontsize=10)

# 2. Score Matrix: Ground Truth vs Prediction
ax = axes[0, 1]
scale_values = np.array([1, 2, 3, 4, 5])
score_matrix = np.zeros((5, 5))

for i, gt in enumerate(scale_values):
    for j, pred in enumerate(scale_values):
        score_matrix[i, j] = np.exp(-(gt - pred)**2 / (2 * sigma**2))

im = ax.imshow(score_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(scale_values)
ax.set_yticklabels(scale_values)
ax.set_xlabel('Predicted Answer', fontsize=12)
ax.set_ylabel('Ground Truth', fontsize=12)
ax.set_title('RBF Score Matrix (Likert Scale 1-5)', fontsize=13, fontweight='bold')

# Add text annotations
for i in range(5):
    for j in range(5):
        text = ax.text(j, i, f'{score_matrix[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=ax, label='Similarity Score')

# 3. Comparison: Binary Accuracy vs RBF Score
ax = axes[1, 0]

# Simulate different scenarios
scenarios = {
    'Perfect': ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    'Off by 1': ([1, 2, 3, 4, 5], [2, 3, 4, 5, 5]),
    'Off by 2': ([1, 2, 3, 4, 5], [3, 4, 5, 5, 5]),
    'Random': ([1, 2, 3, 4, 5], [5, 1, 4, 2, 3]),
}

binary_scores = []
rbf_scores = []
scenario_names = []

for name, (gt, pred) in scenarios.items():
    gt = np.array(gt)
    pred = np.array(pred)

    # Binary accuracy
    binary = np.mean(gt == pred) * 100

    # RBF score (normalized to percentage)
    rbf = np.mean(np.exp(-(gt - pred)**2 / (2 * sigma**2))) * 100

    binary_scores.append(binary)
    rbf_scores.append(rbf)
    scenario_names.append(name)

x = np.arange(len(scenario_names))
width = 0.35

bars1 = ax.bar(x - width/2, binary_scores, width, label='Binary Accuracy',
              color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, rbf_scores, width, label='RBF Score',
              color='coral', alpha=0.8)

ax.set_xlabel('Scenario', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Binary Accuracy vs RBF Score\n(Example Scenarios)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenario_names)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=8)

# 4. Effect of Sigma on RBF Score
ax = axes[1, 1]

differences = np.linspace(0, 4, 1000)
sigmas = [0.1, 0.3, 0.5, 1.0]
colors_sigma = sns.color_palette("viridis", len(sigmas))

for sigma_val, color in zip(sigmas, colors_sigma):
    similarities = np.exp(-differences**2 / (2 * sigma_val**2))
    ax.plot(differences, similarities, label=f'σ={sigma_val}',
           linewidth=2, color=color)

ax.set_xlabel('|Prediction - Ground Truth|', fontsize=12)
ax.set_ylabel('Similarity Score', fontsize=12)
ax.set_title('Effect of σ (Sigma) on RBF Kernel', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

# Add annotation
ax.annotate('Smaller σ = Stricter scoring\nLarger σ = More lenient',
           xy=(2, 0.5),
           xytext=(2.5, 0.7),
           fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig('/home/user/ShinkaEvolve/rbfsim/rbf_kernel_analysis.png', dpi=300, bbox_inches='tight')
print("RBF kernel analysis saved to: rbfsim/rbf_kernel_analysis.png")

# Print detailed analysis
print("\n" + "="*80)
print("RBF KERNEL DETAILED ANALYSIS (σ=0.3)")
print("="*80)
print("\nScore for different prediction errors:")
print("-"*80)
print(f"{'Error (|GT - Pred|)':<25} {'RBF Score':<20} {'Binary Accuracy':<20}")
print("-"*80)

for diff in range(5):
    rbf_score = np.exp(-diff**2 / (2 * 0.3**2))
    binary = 1.0 if diff == 0 else 0.0
    print(f"{diff:<25} {rbf_score:<20.4f} {binary:<20.1f}")

print("-"*80)
print("\nKey Observations:")
print("  • Error of 0 (exact match): 1.000 RBF score (100% similarity)")
print("  • Error of 1 (off by one): 0.011 RBF score (1.1% similarity)")
print("  • Error of 2 (off by two): 0.000 RBF score (essentially 0%)")
print("\nWith σ=0.3, the RBF kernel is VERY strict:")
print("  → Even being off by 1 point gives almost no credit (1.1%)")
print("  → This is close to binary accuracy but slightly more forgiving for small errors")
print("\n" + "="*80)

plt.close('all')
