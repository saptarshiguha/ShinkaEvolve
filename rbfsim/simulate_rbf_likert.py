"""
Simulation of RBF (Gaussian Similarity) scoring for Likert scale assessments.

This script implements a principled scoring approach that accounts for uncertainty in
the ground truth itself. Sigma (σ) represents the standard deviation of inter-expert
disagreement, not a "leniency parameter."

The RBF score is the unnormalized Gaussian likelihood: exp(-(answer - GT)² / (2σ²))

Key principle: If experts disagree on ground truth (high σ), examinees shouldn't be
heavily penalized for answers within the range of expert disagreement.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict
import pandas as pd


class LikertRBFSimulator:
    """Simulator for Likert scale assessment with RBF scoring."""

    def __init__(self, n_questions: int = 100,
                 scale_probs: list = [0.10, 0.25, 0.40, 0.25],
                 sigma: float = 0.5,
                 n_simulations: int = 1000):
        """
        Initialize the simulator.

        Args:
            n_questions: Number of questions in the assessment
            scale_probs: Probability distribution for scale values 1-4
            sigma: Standard deviation of ground truth uncertainty (inter-expert disagreement)
                   Should be empirically estimated from inter-rater reliability studies.
                   Example: If 3 experts rate as [3,3,4], SD≈0.47≈0.5
            n_simulations: Number of simulation runs
        """
        self.n_questions = n_questions
        self.scale_values = np.array([1, 2, 3, 4])
        self.scale_probs = np.array(scale_probs)
        self.sigma = sigma
        self.n_simulations = n_simulations

    def generate_ground_truth(self) -> np.ndarray:
        """Generate ground truth answers based on the distribution."""
        return np.random.choice(self.scale_values,
                               size=self.n_questions,
                               p=self.scale_probs)

    def rbf_score(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calculate RBF (Gaussian similarity) score.

        Score = sum(exp(-|pred - gt|^2 / (2 * sigma^2)))

        This is the unnormalized Gaussian likelihood of the predictions under N(gt, sigma).
        Interpretation: How likely are the examinee's answers given that the ground truth
        itself has uncertainty (variance sigma^2) due to inter-expert disagreement?

        Args:
            predictions: Examinee's answers
            ground_truth: Ground truth answers (may have inherent uncertainty)

        Returns:
            Total RBF score across all questions (max = n_questions for perfect match)
        """
        diff_squared = (predictions - ground_truth) ** 2
        similarities = np.exp(-diff_squared / (2 * self.sigma ** 2))
        return np.sum(similarities)

    def accuracy(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate exact match accuracy."""
        return np.mean(predictions == ground_truth)

    def spearman_correlation(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate Spearman rank correlation."""
        return stats.spearmanr(predictions, ground_truth)[0]

    def generate_accuracy_based_responses(self, ground_truth: np.ndarray,
                                         target_accuracy: float) -> np.ndarray:
        """
        Generate responses with a target accuracy.

        Strategy: For target_accuracy fraction of questions, copy ground truth.
        For remaining questions, sample randomly from the distribution.

        Args:
            ground_truth: True answers
            target_accuracy: Desired accuracy (0-1)

        Returns:
            Simulated responses
        """
        responses = np.zeros(self.n_questions, dtype=int)

        # Randomly select which questions to answer correctly
        n_correct = int(target_accuracy * self.n_questions)
        correct_indices = np.random.choice(self.n_questions, size=n_correct, replace=False)

        # Copy ground truth for correct answers
        responses[correct_indices] = ground_truth[correct_indices]

        # Random answers for incorrect ones
        incorrect_indices = np.setdiff1d(np.arange(self.n_questions), correct_indices)
        responses[incorrect_indices] = np.random.choice(
            self.scale_values,
            size=len(incorrect_indices),
            p=self.scale_probs
        )

        return responses

    def generate_correlation_based_responses(self, ground_truth: np.ndarray,
                                            target_correlation: float,
                                            max_iterations: int = 100) -> np.ndarray:
        """
        Generate responses with a target Spearman correlation.

        Uses Gaussian copula approach:
        1. Generate correlated normal variables
        2. Map to uniform via CDF
        3. Map to ordinal scale via inverse CDF

        Args:
            ground_truth: True answers
            target_correlation: Desired Spearman correlation
            max_iterations: Maximum attempts to achieve target correlation

        Returns:
            Simulated responses
        """
        # Convert ground truth to ranks
        gt_ranks = stats.rankdata(ground_truth)

        # Generate correlated normal variables using Cholesky decomposition
        # For bivariate normal with correlation rho:
        # X1 ~ N(0,1), X2 = rho*X1 + sqrt(1-rho^2)*Z where Z ~ N(0,1)

        rho = target_correlation
        z1 = stats.norm.ppf(gt_ranks / (len(ground_truth) + 1))
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn(self.n_questions)

        # Convert to uniform
        u = stats.norm.cdf(z2)

        # Map to ordinal scale
        cumulative_probs = np.cumsum(self.scale_probs)
        responses = np.searchsorted(cumulative_probs, u) + 1
        responses = np.clip(responses, 1, 5)

        return responses

    def run_single_simulation(self) -> Dict[str, Dict[str, float]]:
        """
        Run a single simulation for all grader types.

        Returns:
            Dictionary with metrics for each grader
        """
        # Generate ground truth
        ground_truth = self.generate_ground_truth()

        results = {}

        # Accuracy-based graders
        for acc in [0.80, 0.70, 0.60]:
            responses = self.generate_accuracy_based_responses(ground_truth, acc)
            results[f'accuracy_{int(acc*100)}pct'] = {
                'rbf_score': self.rbf_score(responses, ground_truth),
                'accuracy': self.accuracy(responses, ground_truth),
                'spearman': self.spearman_correlation(responses, ground_truth)
            }

        # Correlation-based graders
        for corr in [0.70, 0.60]:
            responses = self.generate_correlation_based_responses(ground_truth, corr)
            results[f'correlation_{int(corr*100)}pct'] = {
                'rbf_score': self.rbf_score(responses, ground_truth),
                'accuracy': self.accuracy(responses, ground_truth),
                'spearman': self.spearman_correlation(responses, ground_truth)
            }

        return results

    def run_simulations(self) -> pd.DataFrame:
        """
        Run multiple simulations and aggregate results.

        Returns:
            DataFrame with summary statistics
        """
        all_results = []

        for i in range(self.n_simulations):
            sim_results = self.run_single_simulation()
            for grader_type, metrics in sim_results.items():
                all_results.append({
                    'simulation': i,
                    'grader_type': grader_type,
                    **metrics
                })

        return pd.DataFrame(all_results)

    def summarize_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize simulation results with mean and std.

        Args:
            results_df: Raw simulation results

        Returns:
            Summary statistics
        """
        summary = results_df.groupby('grader_type').agg({
            'rbf_score': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'spearman': ['mean', 'std']
        }).round(4)

        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

        return summary


def main():
    """Run the simulation and display results."""
    print("=" * 80)
    print("RBF (Gaussian Similarity) Scoring Simulation for Likert Scale Assessment")
    print("=" * 80)
    print()

    # Initialize simulator
    simulator = LikertRBFSimulator(
        n_questions=100,
        scale_probs=[0.10, 0.25, 0.40, 0.25],  # 4-point scale distribution
        sigma=0.5,
        n_simulations=1000
    )

    print("Simulation Parameters:")
    print(f"  Number of questions: {simulator.n_questions}")
    print(f"  Scale: 1-4 (4-point Likert)")
    print(f"  Ground truth distribution: {dict(zip(simulator.scale_values, simulator.scale_probs))}")
    print(f"  RBF sigma: {simulator.sigma}")
    print(f"  Number of simulations: {simulator.n_simulations}")
    print()

    print("Running simulations...")
    results_df = simulator.run_simulations()

    print("Simulations complete!")
    print()

    # Summarize results
    summary = simulator.summarize_results(results_df)

    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Display formatted results
    print("Grader Performance Comparison:")
    print("-" * 80)
    print(f"{'Grader Type':<25} {'RBF Score':<20} {'Accuracy':<20} {'Spearman Corr':<20}")
    print("-" * 80)

    for grader in summary.index:
        rbf_mean = summary.loc[grader, 'rbf_score_mean']
        rbf_std = summary.loc[grader, 'rbf_score_std']
        acc_mean = summary.loc[grader, 'accuracy_mean']
        acc_std = summary.loc[grader, 'accuracy_std']
        spear_mean = summary.loc[grader, 'spearman_mean']
        spear_std = summary.loc[grader, 'spearman_std']

        print(f"{grader:<25} {rbf_mean:>6.2f} ± {rbf_std:<6.2f}  "
              f"{acc_mean:>4.1%} ± {acc_std:<5.1%}  "
              f"{spear_mean:>5.3f} ± {spear_std:<5.3f}")

    print("-" * 80)
    print()

    # Additional insights
    print("Key Insights:")
    print("-" * 80)

    # Maximum possible RBF score
    max_rbf = simulator.n_questions  # When all answers are exactly correct
    print(f"Maximum possible RBF score: {max_rbf:.2f} (100% accuracy)")
    print()

    # Normalized scores
    print("Normalized RBF Scores (as percentage of maximum):")
    for grader in summary.index:
        rbf_mean = summary.loc[grader, 'rbf_score_mean']
        normalized = (rbf_mean / max_rbf) * 100
        print(f"  {grader:<25} {normalized:>5.2f}%")
    print()

    # Correlation vs Accuracy comparison
    print("Correlation-based graders - achieved accuracy:")
    for grader in summary.index:
        if 'correlation' in grader:
            acc_mean = summary.loc[grader, 'accuracy_mean']
            spear_mean = summary.loc[grader, 'spearman_mean']
            print(f"  {grader}: {acc_mean:.1%} accuracy, {spear_mean:.3f} correlation")
    print()

    # Save results
    results_df.to_csv('/home/user/ShinkaEvolve/rbfsim/simulation_results.csv', index=False)
    summary.to_csv('/home/user/ShinkaEvolve/rbfsim/summary_statistics.csv')

    print("Results saved to:")
    print("  - rbfsim/simulation_results.csv (raw data)")
    print("  - rbfsim/summary_statistics.csv (summary)")
    print()


if __name__ == "__main__":
    main()
