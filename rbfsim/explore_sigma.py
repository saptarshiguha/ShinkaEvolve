"""
Explore the effect of different sigma values on RBF scoring.
"""

import numpy as np
import pandas as pd
from simulate_rbf_likert import LikertRBFSimulator
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def run_simulation_for_sigma(sigma_value, n_simulations=500):
    """Run simulation for a specific sigma value."""
    simulator = LikertRBFSimulator(
        n_questions=100,
        scale_probs=[0.10, 0.15, 0.30, 0.30, 0.15],
        sigma=sigma_value,
        n_simulations=n_simulations
    )

    results = []
    for _ in range(n_simulations):
        sim_results = simulator.run_single_simulation()
        for grader_type, metrics in sim_results.items():
            results.append({
                'grader_type': grader_type,
                'sigma': sigma_value,
                **metrics
            })

    return pd.DataFrame(results)


# Test different sigma values
sigma_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
print("Running simulations for different sigma values...")
print("This may take a minute...\n")

all_results = []
for sigma in sigma_values:
    print(f"  σ = {sigma}...")
    df = run_simulation_for_sigma(sigma, n_simulations=500)
    all_results.append(df)

results_df = pd.concat(all_results, ignore_index=True)

# Summarize results
summary = results_df.groupby(['sigma', 'grader_type']).agg({
    'rbf_score': ['mean', 'std'],
    'accuracy': ['mean', 'std']
}).reset_index()

# Flatten column names
summary.columns = ['sigma', 'grader_type', 'rbf_mean', 'rbf_std', 'acc_mean', 'acc_std']

print("\n" + "="*100)
print("EFFECT OF SIGMA ON RBF SCORING")
print("="*100)
print("\nFor 80% Accuracy Grader:")
print("-"*100)
print(f"{'Sigma (σ)':<12} {'RBF Score':<25} {'Normalized RBF %':<20} {'Interpretation':<30}")
print("-"*100)

for sigma in sigma_values:
    data = summary[(summary['sigma'] == sigma) & (summary['grader_type'] == 'accuracy_80pct')]
    if len(data) > 0:
        rbf_mean = data['rbf_mean'].values[0]
        rbf_std = data['rbf_std'].values[0]
        normalized = (rbf_mean / 100) * 100

        if normalized >= 95:
            interp = "Very lenient"
        elif normalized >= 85:
            interp = "Slightly lenient"
        elif normalized >= 75:
            interp = "Close to accuracy"
        else:
            interp = "Strict"

        print(f"{sigma:<12.1f} {rbf_mean:>6.2f} ± {rbf_std:<6.2f}      {normalized:>6.2f}%            {interp:<30}")

print("-"*100)
print("\nFor 70% Accuracy Grader:")
print("-"*100)
print(f"{'Sigma (σ)':<12} {'RBF Score':<25} {'Normalized RBF %':<20} {'Interpretation':<30}")
print("-"*100)

for sigma in sigma_values:
    data = summary[(summary['sigma'] == sigma) & (summary['grader_type'] == 'accuracy_70pct')]
    if len(data) > 0:
        rbf_mean = data['rbf_mean'].values[0]
        rbf_std = data['rbf_std'].values[0]
        normalized = (rbf_mean / 100) * 100

        if normalized >= 85:
            interp = "Very lenient"
        elif normalized >= 75:
            interp = "Slightly lenient"
        elif normalized >= 65:
            interp = "Close to accuracy"
        else:
            interp = "Strict"

        print(f"{sigma:<12.1f} {rbf_mean:>6.2f} ± {rbf_std:<6.2f}      {normalized:>6.2f}%            {interp:<30}")

print("-"*100)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Impact of Sigma (σ) on RBF Scoring', fontsize=16, fontweight='bold')

# Plot 1: RBF Score vs Sigma for different accuracy levels
ax = axes[0]
accuracy_graders = ['accuracy_80pct', 'accuracy_70pct', 'accuracy_60pct']
colors = sns.color_palette("husl", len(accuracy_graders))

for grader, color in zip(accuracy_graders, colors):
    data = summary[summary['grader_type'] == grader]
    data = data.sort_values('sigma')

    ax.plot(data['sigma'], data['rbf_mean'], 'o-', label=grader,
           color=color, linewidth=2, markersize=8)
    ax.fill_between(data['sigma'],
                    data['rbf_mean'] - data['rbf_std'],
                    data['rbf_mean'] + data['rbf_std'],
                    alpha=0.2, color=color)

ax.set_xlabel('Sigma (σ)', fontsize=12)
ax.set_ylabel('RBF Score', fontsize=12)
ax.set_title('RBF Score vs Sigma for Accuracy-Based Graders', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Normalized RBF % vs Sigma
ax = axes[1]

for grader, color in zip(accuracy_graders, colors):
    data = summary[summary['grader_type'] == grader]
    data = data.sort_values('sigma')

    normalized = (data['rbf_mean'] / 100) * 100
    accuracy_pct = data['acc_mean'] * 100

    ax.plot(data['sigma'], normalized, 'o-', label=f'{grader} (RBF)',
           color=color, linewidth=2, markersize=8)
    ax.axhline(accuracy_pct.values[0], color=color, linestyle='--',
              alpha=0.5, label=f'{grader} (Accuracy)')

ax.set_xlabel('Sigma (σ)', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Normalized RBF Score vs Sigma\n(Dashed lines = true accuracy)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/ShinkaEvolve/rbfsim/sigma_exploration.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: rbfsim/sigma_exploration.png")

# Create detailed comparison table
print("\n" + "="*100)
print("DETAILED COMPARISON: How Sigma Affects Scoring")
print("="*100)

# Show partial credit for errors at different sigma values
print("\nPartial Credit for Prediction Errors (how much credit for being off by N points):")
print("-"*100)
print(f"{'Sigma':<10} {'Error=0':<12} {'Error=1':<12} {'Error=2':<12} {'Error=3':<12} {'Error=4':<12}")
print("-"*100)

for sigma in sigma_values:
    scores = []
    for error in range(5):
        score = np.exp(-error**2 / (2 * sigma**2))
        scores.append(score)

    print(f"{sigma:<10.1f} {scores[0]:>8.4f}    {scores[1]:>8.4f}    "
          f"{scores[2]:>8.4f}    {scores[3]:>8.4f}    {scores[4]:>8.4f}")

print("-"*100)

print("\n" + "="*100)
print("\nRECOMMENDATIONS:")
print("-"*100)
print("• σ = 0.1-0.3:  Very strict, almost identical to binary accuracy")
print("               Use if exact matches are critical")
print()
print("• σ = 0.5-0.7:  Balanced approach, gives modest credit for being off by 1")
print("               Good for Likert scales where ±1 variation is reasonable")
print()
print("• σ = 1.0-1.5:  Lenient, gives substantial credit for near-misses")
print("               Use if you want to reward 'close enough' answers")
print("="*100)

# Save results
results_df.to_csv('/home/user/ShinkaEvolve/rbfsim/sigma_exploration_results.csv', index=False)
print("\nDetailed results saved to: rbfsim/sigma_exploration_results.csv")
