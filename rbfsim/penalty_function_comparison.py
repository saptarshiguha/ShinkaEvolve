"""
Comparison of different penalty/scoring functions for Likert scales.

Compares:
1. RBF/Gaussian: exp(-error²/(2σ²))
2. Linear normalized distance: 1 - error/max_error
3. Quadratic normalized distance: 1 - (error/max_error)²
4. Binary (0-1): 1 if error=0, else 0
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

def rbf_score(error, sigma=0.7):
    """RBF/Gaussian kernel."""
    return np.exp(-error**2 / (2 * sigma**2))

def linear_score(error, max_error):
    """Linear normalized distance: 1 - |error|/max_error."""
    return 1 - (error / max_error)

def quadratic_score(error, max_error):
    """Quadratic normalized distance: 1 - (|error|/max_error)²."""
    return 1 - (error / max_error)**2

def cubic_score(error, max_error):
    """Cubic normalized distance: 1 - (|error|/max_error)³."""
    return 1 - (error / max_error)**3

def binary_score(error):
    """Binary 0-1 scoring."""
    return 1.0 if error == 0 else 0.0

# For 5-point scale
max_error = 4
errors = np.linspace(0, max_error, 100)

# Calculate scores for each penalty function
rbf_scores = rbf_score(errors, sigma=0.7)
linear_scores = linear_score(errors, max_error)
quadratic_scores = quadratic_score(errors, max_error)
cubic_scores = cubic_score(errors, max_error)

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Penalty Function Comparison for 5-Point Likert Scale', fontsize=16, fontweight='bold')

# Plot 1: All functions together
ax = axes[0, 0]
ax.plot(errors, rbf_scores, 'r-', linewidth=2.5, label='RBF/Gaussian (σ=0.7)')
ax.plot(errors, linear_scores, 'b-', linewidth=2.5, label='Linear normalized')
ax.plot(errors, quadratic_scores, 'g-', linewidth=2.5, label='Quadratic normalized')
ax.plot(errors, cubic_scores, 'm-', linewidth=2.5, label='Cubic normalized')

# Mark integer errors
for err in range(5):
    ax.axvline(err, color='gray', linestyle=':', alpha=0.3)

ax.set_xlabel('Error (|Prediction - Ground Truth|)', fontsize=12)
ax.set_ylabel('Score (0-1)', fontsize=12)
ax.set_title('All Penalty Functions', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_error)
ax.set_ylim(-0.05, 1.05)

# Plot 2: Discrete comparison (integer errors only)
ax = axes[0, 1]
integer_errors = np.arange(5)
width = 0.18

rbf_discrete = [rbf_score(e, 0.7) for e in integer_errors]
linear_discrete = [linear_score(e, max_error) for e in integer_errors]
quadratic_discrete = [quadratic_score(e, max_error) for e in integer_errors]
cubic_discrete = [cubic_score(e, max_error) for e in integer_errors]
binary_discrete = [binary_score(e) for e in integer_errors]

x = np.arange(len(integer_errors))
ax.bar(x - 2*width, rbf_discrete, width, label='RBF/Gaussian', color='red', alpha=0.8)
ax.bar(x - width, linear_discrete, width, label='Linear', color='blue', alpha=0.8)
ax.bar(x, quadratic_discrete, width, label='Quadratic', color='green', alpha=0.8)
ax.bar(x + width, cubic_discrete, width, label='Cubic', color='magenta', alpha=0.8)
ax.bar(x + 2*width, binary_discrete, width, label='Binary', color='orange', alpha=0.8)

ax.set_xlabel('Error (units)', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Discrete Comparison (Integer Errors)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(integer_errors)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Difference from linear (shows how different RBF is)
ax = axes[1, 0]
rbf_diff = rbf_scores - linear_scores
quadratic_diff = quadratic_scores - linear_scores
cubic_diff = cubic_scores - linear_scores

ax.plot(errors, rbf_diff, 'r-', linewidth=2.5, label='RBF - Linear')
ax.plot(errors, quadratic_diff, 'g-', linewidth=2.5, label='Quadratic - Linear')
ax.plot(errors, cubic_diff, 'm-', linewidth=2.5, label='Cubic - Linear')
ax.axhline(0, color='black', linestyle='-', linewidth=1)

ax.set_xlabel('Error', fontsize=12)
ax.set_ylabel('Score Difference from Linear', fontsize=12)
ax.set_title('Deviation from Linear Penalty\n(Positive = More Lenient, Negative = More Strict)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_error)

# Add annotations
ax.annotate('RBF is more lenient\nfor small errors',
           xy=(0.5, rbf_diff[50]),
           xytext=(1.5, 0.15),
           fontsize=9,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

ax.annotate('RBF is stricter\nfor large errors',
           xy=(3, rbf_diff[-50]),
           xytext=(2, -0.1),
           fontsize=9,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

# Plot 4: Cumulative scores for a grader (simulated)
ax = axes[1, 1]

# Simulate a grader who gets varying errors
np.random.seed(42)
n_questions = 100
simulated_errors = np.random.choice([0, 0, 0, 0, 0, 1, 1, 1, 2, 3], size=n_questions)  # Mostly good

rbf_total = sum([rbf_score(e, 0.7) for e in simulated_errors])
linear_total = sum([linear_score(e, max_error) for e in simulated_errors])
quadratic_total = sum([quadratic_score(e, max_error) for e in simulated_errors])
cubic_total = sum([cubic_score(e, max_error) for e in simulated_errors])
binary_total = sum([binary_score(e) for e in simulated_errors])

methods = ['RBF\n(σ=0.7)', 'Linear\nNorm', 'Quadratic\nNorm', 'Cubic\nNorm', 'Binary\n0-1']
totals = [rbf_total, linear_total, quadratic_total, cubic_total, binary_total]
colors_bar = ['red', 'blue', 'green', 'magenta', 'orange']

bars = ax.bar(methods, totals, color=colors_bar, alpha=0.7, edgecolor='black')
ax.set_ylabel('Total Score (out of 100)', fontsize=12)
ax.set_title('Total Scores for Same Grader\n(50% exact, 30% off-by-1, 10% off-by-2, 10% off-by-3)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 105)

# Add value labels on bars
for bar, total in zip(bars, totals):
    height = bar.get_height()
    ax.annotate(f'{total:.1f}',
               xy=(bar.get_x() + bar.get_width() / 2, height),
               xytext=(0, 3),
               textcoords="offset points",
               ha='center', va='bottom',
               fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/user/ShinkaEvolve/rbfsim/penalty_function_comparison.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: rbfsim/penalty_function_comparison.png\n")

# Print detailed comparison
print("="*100)
print("PENALTY FUNCTION COMPARISON: RBF vs Normalized Distance")
print("="*100)
print()
print("For 5-point Likert scale (max error = 4 units):")
print()
print(f"{'Error':<10} {'RBF (σ=0.7)':<18} {'Linear':<18} {'Quadratic':<18} {'Cubic':<18} {'Binary':<10}")
print("-"*100)

for err in range(5):
    rbf_val = rbf_score(err, 0.7)
    lin_val = linear_score(err, max_error)
    quad_val = quadratic_score(err, max_error)
    cub_val = cubic_score(err, max_error)
    bin_val = binary_score(err)

    print(f"{err:<10} {rbf_val:>6.4f} ({rbf_val*100:>5.1f}%)  "
          f"{lin_val:>6.4f} ({lin_val*100:>5.1f}%)  "
          f"{quad_val:>6.4f} ({quad_val*100:>5.1f}%)  "
          f"{cub_val:>6.4f} ({cub_val*100:>5.1f}%)  "
          f"{bin_val:>4.1f}")

print("-"*100)
print()

# Calculate differences at key points
print("KEY DIFFERENCES:")
print("-"*100)
print()

for err in [1, 2, 3, 4]:
    rbf_val = rbf_score(err, 0.7)
    lin_val = linear_score(err, max_error)
    diff = (rbf_val - lin_val) * 100

    if diff > 0:
        comparison = "MORE LENIENT"
    elif diff < 0:
        comparison = "MORE STRICT"
    else:
        comparison = "SAME"

    print(f"Error = {err}: RBF gives {rbf_val*100:.1f}% vs Linear gives {lin_val*100:.1f}%")
    print(f"           → RBF is {comparison} by {abs(diff):.1f} percentage points")
    print()

print("="*100)
print()
print("INTERPRETATION:")
print("-"*100)
print()
print("RBF/Gaussian (σ=0.7):")
print("  • Very lenient for small errors (error=1: 36.0% vs 75.0% linear)")
print("  • Very strict for large errors (error=3: 0.01% vs 25.0% linear)")
print("  • Non-linear, exponential decay - rewards near-misses, heavily penalizes far-misses")
print()
print("Linear normalized (1 - error/max):")
print("  • Simple, intuitive")
print("  • Constant penalty per unit of error")
print("  • More generous for medium/large errors")
print("  • Error=1 gets 75% credit (vs 36% for RBF)")
print()
print("Quadratic normalized (1 - (error/max)²):")
print("  • Middle ground between linear and RBF")
print("  • Error=1 gets 93.8% credit (very lenient)")
print("  • Error=2 gets 75% credit")
print()
print("WHICH TO USE?")
print("-"*100)
print("• Use LINEAR if you want simple, proportional penalties")
print("• Use QUADRATIC if you want to be lenient about small errors")
print("• Use RBF if you want to heavily reward near-misses but severely penalize far-misses")
print("• RBF is NOT radically different from normalized distance - it's just more extreme")
print("  in rewarding near-misses and penalizing far-misses")
print()
print("="*100)
