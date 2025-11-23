"""
Bayesian approaches to scoring Likert scale assessments with ground truth uncertainty.

This script explores principled Bayesian methods for scoring when expert ratings
have inherent disagreement (variance σ²).

Key insight: Instead of treating expert consensus as the "true" answer, model the
true answer θ as unknown and compute the posterior P(θ | expert_ratings), then
use this posterior to score the examinee's answer.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")


class BayesianLikertScorer:
    """Bayesian scoring for Likert scales with uncertain ground truth."""

    def __init__(self, scale_values=np.array([1, 2, 3, 4]), sigma=0.5):
        """
        Initialize Bayesian scorer.

        Args:
            scale_values: Possible values on the Likert scale
            sigma: SD of expert disagreement (measurement noise)
        """
        self.scale_values = scale_values
        self.sigma = sigma
        self.min_val = np.min(scale_values)
        self.max_val = np.max(scale_values)

    def posterior_predictive_score(self, answer, expert_ratings, sigma_examinee=0.5):
        """
        Posterior predictive probability approach.

        Model:
        - True answer: θ (unknown)
        - Expert ratings: GTᵢ ~ N(θ, σ²)
        - Posterior: θ | {GTᵢ} ~ N(mean(GTᵢ), σ²/K)
        - Examinee answer: A | θ ~ N(θ, σ_examinee²)
        - Posterior predictive: A | {GTᵢ} ~ N(mean(GTᵢ), σ²/K + σ_examinee²)

        Args:
            answer: Examinee's answer
            expert_ratings: Array of expert ratings (can be single value or multiple)
            sigma_examinee: SD of examinee's response around true θ

        Returns:
            Score based on posterior predictive likelihood
        """
        expert_ratings = np.atleast_1d(expert_ratings)
        K = len(expert_ratings)

        # Posterior parameters
        posterior_mean = np.mean(expert_ratings)
        posterior_var = self.sigma**2 / K

        # Posterior predictive variance
        predictive_var = posterior_var + sigma_examinee**2

        # Score = unnormalized Gaussian likelihood
        score = np.exp(-(answer - posterior_mean)**2 / (2 * predictive_var))

        return score

    def expected_loss_score(self, answer, expert_ratings, loss_type='squared'):
        """
        Expected loss approach.

        Compute E[L(A, θ) | {GTᵢ}] where L is a loss function.

        For squared loss:
        E[(A - θ)² | {GTᵢ}] = (A - posterior_mean)² + posterior_variance

        Score = 1 - normalized_expected_loss

        Args:
            answer: Examinee's answer
            expert_ratings: Array of expert ratings
            loss_type: Type of loss ('squared', 'absolute')

        Returns:
            Score based on expected loss (higher is better)
        """
        expert_ratings = np.atleast_1d(expert_ratings)
        K = len(expert_ratings)

        # Posterior parameters
        posterior_mean = np.mean(expert_ratings)
        posterior_var = self.sigma**2 / K

        if loss_type == 'squared':
            # Expected squared loss
            expected_loss = (answer - posterior_mean)**2 + posterior_var
            # Normalize by max possible loss
            max_loss = (self.max_val - self.min_val)**2 + posterior_var
            score = 1 - (expected_loss / max_loss)

        elif loss_type == 'absolute':
            # Expected absolute loss (harder to compute analytically)
            # Approximate by sampling from posterior
            theta_samples = np.random.normal(posterior_mean, np.sqrt(posterior_var), 10000)
            # Clip to valid range
            theta_samples = np.clip(theta_samples, self.min_val, self.max_val)
            expected_loss = np.mean(np.abs(answer - theta_samples))
            max_loss = self.max_val - self.min_val
            score = 1 - (expected_loss / max_loss)

        return max(0, score)  # Ensure non-negative

    def credible_interval_score(self, answer, expert_ratings, credible_level=0.95):
        """
        Credible interval approach.

        Compute credible interval for θ given expert ratings.
        - Full credit if answer is within interval
        - Partial credit based on distance from interval

        Args:
            answer: Examinee's answer
            expert_ratings: Array of expert ratings
            credible_level: Credible interval level (e.g., 0.95)

        Returns:
            Score based on credible interval inclusion
        """
        expert_ratings = np.atleast_1d(expert_ratings)
        K = len(expert_ratings)

        # Posterior parameters
        posterior_mean = np.mean(expert_ratings)
        posterior_std = self.sigma / np.sqrt(K)

        # Credible interval
        z = stats.norm.ppf((1 + credible_level) / 2)
        lower = posterior_mean - z * posterior_std
        upper = posterior_mean + z * posterior_std

        # Score based on interval
        if lower <= answer <= upper:
            # Inside credible interval - full credit
            score = 1.0
        else:
            # Outside interval - partial credit based on distance
            distance = min(abs(answer - lower), abs(answer - upper))
            # Exponential decay with distance
            score = np.exp(-distance / self.sigma)

        return score

    def rbf_baseline(self, answer, expert_ratings):
        """
        Original RBF approach for comparison.

        Score = exp(-(A - mean(GT))² / (2σ²))

        Args:
            answer: Examinee's answer
            expert_ratings: Array of expert ratings

        Returns:
            RBF score
        """
        expert_ratings = np.atleast_1d(expert_ratings)
        gt_mean = np.mean(expert_ratings)

        score = np.exp(-(answer - gt_mean)**2 / (2 * self.sigma**2))

        return score


def demonstrate_bayesian_scoring():
    """Demonstrate different Bayesian scoring approaches."""

    scorer = BayesianLikertScorer(scale_values=np.array([1, 2, 3, 4]), sigma=0.5)

    print("="*80)
    print("BAYESIAN SCORING APPROACHES FOR LIKERT SCALES")
    print("="*80)
    print()

    # Scenario 1: Single expert (no uncertainty reduction)
    print("SCENARIO 1: Single expert rating = 3")
    print("-"*80)
    expert_ratings_1 = [3]

    print(f"{'Examinee Answer':<20} {'RBF':<12} {'Posterior Pred':<18} "
          f"{'Expected Loss':<18} {'Credible Int':<15}")
    print("-"*80)

    for answer in [1, 2, 3, 4]:
        rbf = scorer.rbf_baseline(answer, expert_ratings_1)
        post_pred = scorer.posterior_predictive_score(answer, expert_ratings_1)
        exp_loss = scorer.expected_loss_score(answer, expert_ratings_1)
        cred_int = scorer.credible_interval_score(answer, expert_ratings_1)

        print(f"{answer:<20} {rbf:<12.4f} {post_pred:<18.4f} "
              f"{exp_loss:<18.4f} {cred_int:<15.4f}")
    print()

    # Scenario 2: Three experts agree (low uncertainty)
    print("SCENARIO 2: Three experts all rate = 3 (perfect agreement)")
    print("-"*80)
    expert_ratings_2 = [3, 3, 3]

    print(f"{'Examinee Answer':<20} {'RBF':<12} {'Posterior Pred':<18} "
          f"{'Expected Loss':<18} {'Credible Int':<15}")
    print("-"*80)

    for answer in [1, 2, 3, 4]:
        rbf = scorer.rbf_baseline(answer, expert_ratings_2)
        post_pred = scorer.posterior_predictive_score(answer, expert_ratings_2)
        exp_loss = scorer.expected_loss_score(answer, expert_ratings_2)
        cred_int = scorer.credible_interval_score(answer, expert_ratings_2)

        print(f"{answer:<20} {rbf:<12.4f} {post_pred:<18.4f} "
              f"{exp_loss:<18.4f} {cred_int:<15.4f}")
    print()

    # Scenario 3: Three experts disagree (high uncertainty)
    print("SCENARIO 3: Three experts disagree [2, 3, 4] (mean=3, SD≈0.82)")
    print("-"*80)
    expert_ratings_3 = [2, 3, 4]

    print(f"{'Examinee Answer':<20} {'RBF':<12} {'Posterior Pred':<18} "
          f"{'Expected Loss':<18} {'Credible Int':<15}")
    print("-"*80)

    for answer in [1, 2, 3, 4]:
        rbf = scorer.rbf_baseline(answer, expert_ratings_3)
        post_pred = scorer.posterior_predictive_score(answer, expert_ratings_3)
        exp_loss = scorer.expected_loss_score(answer, expert_ratings_3)
        cred_int = scorer.credible_interval_score(answer, expert_ratings_3)

        print(f"{answer:<20} {rbf:<12.4f} {post_pred:<18.4f} "
              f"{exp_loss:<18.4f} {cred_int:<15.4f}")
    print()

    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()
    print("1. POSTERIOR PREDICTIVE accounts for both:")
    print("   - Expert disagreement (σ)")
    print("   - Number of experts (variance reduces as 1/K)")
    print("   - Examinee response uncertainty")
    print()
    print("2. EXPECTED LOSS approach:")
    print("   - Accounts for posterior uncertainty in the loss")
    print("   - More generous than RBF (1 - normalized loss)")
    print()
    print("3. CREDIBLE INTERVAL approach:")
    print("   - Binary-ish: full credit if within interval")
    print("   - Interval width decreases with more experts (√K effect)")
    print()
    print("4. With MORE experts (K↑):")
    print("   - Posterior variance decreases (σ²/K)")
    print("   - Scoring becomes stricter (more certainty about θ)")
    print("   - This is FAIR: if experts agree, examinee should match closely")
    print()
    print("="*80)

    # Visualization
    visualize_bayesian_approaches(scorer)


def visualize_bayesian_approaches(scorer):
    """Visualize different Bayesian scoring approaches."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bayesian Scoring Approaches Comparison', fontsize=16, fontweight='bold')

    answers = np.linspace(1, 4, 100)

    scenarios = [
        ([3], "Single expert: GT=3"),
        ([3, 3, 3], "Three experts agree: GT=[3,3,3]"),
        ([2, 3, 4], "Three experts disagree: GT=[2,3,4]"),
        ([3]*10, "Ten experts agree: GT=[3,...,3]")
    ]

    for idx, (expert_ratings, title) in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]

        rbf_scores = [scorer.rbf_baseline(a, expert_ratings) for a in answers]
        post_pred_scores = [scorer.posterior_predictive_score(a, expert_ratings) for a in answers]
        exp_loss_scores = [scorer.expected_loss_score(a, expert_ratings) for a in answers]
        cred_int_scores = [scorer.credible_interval_score(a, expert_ratings) for a in answers]

        ax.plot(answers, rbf_scores, 'b-', linewidth=2, label='RBF (baseline)', alpha=0.8)
        ax.plot(answers, post_pred_scores, 'r-', linewidth=2, label='Posterior Predictive', alpha=0.8)
        ax.plot(answers, exp_loss_scores, 'g-', linewidth=2, label='Expected Loss', alpha=0.8)
        ax.plot(answers, cred_int_scores, 'm-', linewidth=2, label='Credible Interval', alpha=0.8)

        # Mark expert ratings
        for rating in expert_ratings:
            ax.axvline(rating, color='gray', linestyle='--', alpha=0.3)

        ax.set_xlabel('Examinee Answer', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Add posterior info
        K = len(expert_ratings)
        post_mean = np.mean(expert_ratings)
        post_std = scorer.sigma / np.sqrt(K)
        ax.text(0.02, 0.98, f'Posterior: N({post_mean:.2f}, {post_std:.3f}²)',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/user/ShinkaEvolve/rbfsim/bayesian_scoring_comparison.png',
                dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: rbfsim/bayesian_scoring_comparison.png")
    print()


if __name__ == "__main__":
    demonstrate_bayesian_scoring()
