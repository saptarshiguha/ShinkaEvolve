"""
Visualization of RBF simulation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load results
results_df = pd.read_csv('/home/user/ShinkaEvolve/rbfsim/simulation_results.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('RBF (Gaussian Similarity) Scoring Analysis for Likert Scale Assessment',
             fontsize=16, fontweight='bold')

# Color palette
colors = sns.color_palette("husl", 5)
grader_colors = {
    'accuracy_80pct': colors[0],
    'accuracy_70pct': colors[1],
    'accuracy_60pct': colors[2],
    'correlation_70pct': colors[3],
    'correlation_60pct': colors[4]
}

# 1. RBF Score Distribution
ax = axes[0, 0]
for grader in results_df['grader_type'].unique():
    data = results_df[results_df['grader_type'] == grader]['rbf_score']
    ax.hist(data, alpha=0.6, label=grader, bins=30, color=grader_colors[grader])
ax.set_xlabel('RBF Score', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of RBF Scores by Grader Type', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.axvline(100, color='red', linestyle='--', alpha=0.5, label='Max (100)')

# 2. RBF Score Box Plot
ax = axes[0, 1]
grader_order = ['accuracy_80pct', 'accuracy_70pct', 'accuracy_60pct',
                'correlation_70pct', 'correlation_60pct']
sns.boxplot(data=results_df, x='grader_type', y='rbf_score', ax=ax,
            order=grader_order, palette=grader_colors)
ax.set_xlabel('Grader Type', fontsize=11)
ax.set_ylabel('RBF Score', fontsize=11)
ax.set_title('RBF Score Variability', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# 3. Accuracy vs RBF Score
ax = axes[0, 2]
for grader in results_df['grader_type'].unique():
    data = results_df[results_df['grader_type'] == grader]
    ax.scatter(data['accuracy'], data['rbf_score'], alpha=0.3,
              label=grader, color=grader_colors[grader], s=10)
ax.set_xlabel('Accuracy', fontsize=11)
ax.set_ylabel('RBF Score', fontsize=11)
ax.set_title('Accuracy vs RBF Score', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.plot([0, 1], [0, 100], 'k--', alpha=0.3, label='y=100x')

# 4. Spearman Correlation Distribution
ax = axes[1, 0]
for grader in results_df['grader_type'].unique():
    data = results_df[results_df['grader_type'] == grader]['spearman']
    ax.hist(data, alpha=0.6, label=grader, bins=30, color=grader_colors[grader])
ax.set_xlabel('Spearman Correlation', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Spearman Correlation by Grader Type', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# 5. Correlation vs RBF Score
ax = axes[1, 1]
for grader in results_df['grader_type'].unique():
    data = results_df[results_df['grader_type'] == grader]
    ax.scatter(data['spearman'], data['rbf_score'], alpha=0.3,
              label=grader, color=grader_colors[grader], s=10)
ax.set_xlabel('Spearman Correlation', fontsize=11)
ax.set_ylabel('RBF Score', fontsize=11)
ax.set_title('Spearman Correlation vs RBF Score', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# 6. Accuracy vs Spearman Correlation
ax = axes[1, 2]
for grader in results_df['grader_type'].unique():
    data = results_df[results_df['grader_type'] == grader]
    ax.scatter(data['accuracy'], data['spearman'], alpha=0.3,
              label=grader, color=grader_colors[grader], s=10)
ax.set_xlabel('Accuracy', fontsize=11)
ax.set_ylabel('Spearman Correlation', fontsize=11)
ax.set_title('Accuracy vs Spearman Correlation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')

plt.tight_layout()
plt.savefig('/home/user/ShinkaEvolve/rbfsim/rbf_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: rbfsim/rbf_analysis.png")

# Create a summary table visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

summary = results_df.groupby('grader_type').agg({
    'rbf_score': ['mean', 'std'],
    'accuracy': ['mean', 'std'],
    'spearman': ['mean', 'std']
}).round(3)

# Create table data
table_data = []
table_data.append(['Grader Type', 'RBF Score (Mean ± SD)', 'Accuracy (Mean ± SD)',
                  'Spearman (Mean ± SD)', 'Normalized RBF %'])

for grader in grader_order:
    if grader in summary.index:
        rbf_mean = summary.loc[grader, ('rbf_score', 'mean')]
        rbf_std = summary.loc[grader, ('rbf_score', 'std')]
        acc_mean = summary.loc[grader, ('accuracy', 'mean')]
        acc_std = summary.loc[grader, ('accuracy', 'std')]
        spear_mean = summary.loc[grader, ('spearman', 'mean')]
        spear_std = summary.loc[grader, ('spearman', 'std')]
        normalized = (rbf_mean / 100) * 100

        table_data.append([
            grader,
            f'{rbf_mean:.2f} ± {rbf_std:.2f}',
            f'{acc_mean:.1%} ± {acc_std:.1%}',
            f'{spear_mean:.3f} ± {spear_std:.3f}',
            f'{normalized:.1f}%'
        ])

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.25, 0.20, 0.20, 0.20, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

plt.title('Summary Statistics - RBF Scoring Simulation',
         fontsize=14, fontweight='bold', pad=20)
plt.savefig('/home/user/ShinkaEvolve/rbfsim/summary_table.png', dpi=300, bbox_inches='tight')
print("Summary table saved to: rbfsim/summary_table.png")

plt.close('all')
print("\nAll visualizations complete!")
