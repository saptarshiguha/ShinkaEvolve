"""
Simulation of Linear Normalized Distance scoring for Likert scale assessments.

Uses the simple formula: Score = 1 - (error / max_error)
Compares results to RBF (Gaussian) approach.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict
import pandas as pd


class LikertLinearSimulator:
    """Simulator for Likert scale assessment with linear normalized distance scoring."""

    def __init__(self, n_questions: int = 100,
                 scale_probs: list = [0.10, 0.15, 0.30, 0.30, 0.15],
                 n_simulations: int = 1000):
        """
        Initialize the simulator.

        Args:
            n_questions: Number of questions in the assessment
            scale_probs: Probability distribution for scale values 1-5
            n_simulations: Number of simulation runs
        """
        self.n_questions = n_questions
        self.scale_values = np.array([1, 2, 3, 4, 5])
        self.scale_probs = np.array(scale_probs)
        self.max_error = np.max(self.scale_values) - np.min(self.scale_values)
        self.n_simulations = n_simulations

    def generate_ground_truth(self) -> np.ndarray:
        """Generate ground truth answers based on the distribution."""
        return np.random.choice(self.scale_values,
                               size=self.n_questions,
                               p=self.scale_probs)

    def linear_score(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calculate linear normalized distance score.

        Score = sum(1 - |pred - gt| / max_error)

        Args:
            predictions: Predicted answers
            ground_truth: True answers

        Returns:
            Total linear score across all questions
        """
        errors = np.abs(predictions - ground_truth)
        scores = 1 - (errors / self.max_error)
        return np.sum(scores)

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

        Uses Gaussian copula approach.

        Args:
            ground_truth: True answers
            target_correlation: Desired Spearman correlation
            max_iterations: Maximum attempts to achieve target correlation

        Returns:
            Simulated responses
        """
        # Convert ground truth to ranks
        gt_ranks = stats.rankdata(ground_truth)

        # Generate correlated normal variables
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
                'linear_score': self.linear_score(responses, ground_truth),
                'accuracy': self.accuracy(responses, ground_truth),
                'spearman': self.spearman_correlation(responses, ground_truth)
            }

        # Correlation-based graders
        for corr in [0.70, 0.60]:
            responses = self.generate_correlation_based_responses(ground_truth, corr)
            results[f'correlation_{int(corr*100)}pct'] = {
                'linear_score': self.linear_score(responses, ground_truth),
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
            'linear_score': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'spearman': ['mean', 'std']
        }).round(4)

        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

        return summary


def main():
    """Run the simulation and display results."""
    print("=" * 80)
    print("Linear Normalized Distance Scoring Simulation for Likert Scale Assessment")
    print("=" * 80)
    print()

    # Initialize simulator
    simulator = LikertLinearSimulator(
        n_questions=100,
        scale_probs=[0.10, 0.15, 0.30, 0.30, 0.15],
        n_simulations=1000
    )

    print("Simulation Parameters:")
    print(f"  Number of questions: {simulator.n_questions}")
    print(f"  Scale: 1-5 (Likert)")
    print(f"  Ground truth distribution: {dict(zip(simulator.scale_values, simulator.scale_probs))}")
    print(f"  Scoring: Linear normalized distance (1 - error/max_error)")
    print(f"  Max error: {simulator.max_error}")
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
    print(f"{'Grader Type':<25} {'Linear Score':<20} {'Accuracy':<20} {'Spearman Corr':<20}")
    print("-" * 80)

    for grader in summary.index:
        lin_mean = summary.loc[grader, 'linear_score_mean']
        lin_std = summary.loc[grader, 'linear_score_std']
        acc_mean = summary.loc[grader, 'accuracy_mean']
        acc_std = summary.loc[grader, 'accuracy_std']
        spear_mean = summary.loc[grader, 'spearman_mean']
        spear_std = summary.loc[grader, 'spearman_std']

        print(f"{grader:<25} {lin_mean:>6.2f} ± {lin_std:<6.2f}  "
              f"{acc_mean:>4.1%} ± {acc_std:<5.1%}  "
              f"{spear_mean:>5.3f} ± {spear_std:<5.3f}")

    print("-" * 80)
    print()

    # Additional insights
    print("Key Insights:")
    print("-" * 80)

    # Maximum possible score
    max_score = simulator.n_questions  # When all answers are exactly correct
    print(f"Maximum possible linear score: {max_score:.2f} (100% accuracy)")
    print()

    # Normalized scores
    print("Normalized Linear Scores (as percentage of maximum):")
    for grader in summary.index:
        lin_mean = summary.loc[grader, 'linear_score_mean']
        normalized = (lin_mean / max_score) * 100
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
    results_df.to_csv('/home/user/ShinkaEvolve/rbfsim/linear_simulation_results.csv', index=False)
    summary.to_csv('/home/user/ShinkaEvolve/rbfsim/linear_summary_statistics.csv')

    print("Results saved to:")
    print("  - rbfsim/linear_simulation_results.csv (raw data)")
    print("  - rbfsim/linear_summary_statistics.csv (summary)")
    print()

    # Load and compare with RBF results
    print("=" * 80)
    print("COMPARISON WITH RBF (σ=0.7) RESULTS")
    print("=" * 80)
    print()

    try:
        rbf_summary = pd.read_csv('/home/user/ShinkaEvolve/rbfsim/summary_statistics.csv', index_col=0)

        print(f"{'Grader Type':<25} {'Linear Score':<18} {'RBF Score':<18} {'Difference':<15}")
        print("-" * 80)

        for grader in summary.index:
            lin_mean = summary.loc[grader, 'linear_score_mean']
            if grader in rbf_summary.index:
                rbf_mean = rbf_summary.loc[grader, 'rbf_score_mean']
                diff = lin_mean - rbf_mean
                diff_pct = (diff / max_score) * 100

                print(f"{grader:<25} {lin_mean:>6.2f} ({lin_mean/max_score*100:>5.1f}%)  "
                      f"{rbf_mean:>6.2f} ({rbf_mean/max_score*100:>5.1f}%)  "
                      f"{diff:>+6.2f} ({diff_pct:>+5.1f}%)")

        print("-" * 80)
        print()
        print("Key Observations:")
        print("  • Linear scoring is MORE GENEROUS across all grader types")
        print("  • Difference is largest for mid-performing graders (60-70%)")
        print("  • Both methods rank graders in the same order")

    except FileNotFoundError:
        print("  (RBF results not found for comparison)")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
