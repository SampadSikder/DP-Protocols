import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
from collections import defaultdict, Counter
import warnings
from basic_dp import LaplaceMechanism, GaussianMechanism
warnings.filterwarnings('ignore')

class RatioTestDP:
    """
    Implements ratio test to empirically evaluate if a mechanism satisfies ε-differential privacy.
    """
    
    def __init__(self, epsilon_claim, num_trials=10000, tolerance=0.2):
        """
        Initialize the ratio test.
        """
        self.epsilon_claim = epsilon_claim
        self.num_trials = num_trials
        self.tolerance = tolerance
        self.max_allowed_ratio = math.exp(epsilon_claim)

    def test_mechanism(self, mechanism, dataset1, dataset2, query_func):
        print(f"Testing ε-DP with claimed ε = {self.epsilon_claim}")
        print(f"Maximum allowed ratio: {self.max_allowed_ratio:.3f}")
        print(f"Running {self.num_trials} trials for each dataset...")
        
        # Compute query results once for each dataset
        true_value1 = query_func(dataset1)
        true_value2 = query_func(dataset2)
        
        print(f"True value for dataset 1: {true_value1}")
        print(f"True value for dataset 2: {true_value2}")
        
        # Generate samples by adding noise to the computed values
        samples1 = [mechanism.add_noise(true_value1) for _ in range(self.num_trials)]
        samples2 = [mechanism.add_noise(true_value2) for _ in range(self.num_trials)]

        # Create bins for histogram analysis
        bins = 20
        all_samples = samples1 + samples2
        min_val, max_val = min(all_samples), max(all_samples)
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Count samples in each bin
        hist1, _ = np.histogram(samples1, bins=bin_edges)
        hist2, _ = np.histogram(samples2, bins=bin_edges)
        
        bins = len(hist1)
        smoothing = 1e-3  
        prob1 = (hist1 + smoothing) / (self.num_trials + smoothing * bins)
        prob2 = (hist2 + smoothing) / (self.num_trials + smoothing * bins)
        
         # Calculate ratios in both directions
        ratios_1_to_2 = prob1 / prob2
        ratios_2_to_1 = prob2 / prob1
        
        # Find maximum ratio (in either direction)
        max_ratio = max(np.max(ratios_1_to_2), np.max(ratios_2_to_1))
        
        # Count violations (allowing small tolerance for numerical errors)
        violations_1_to_2 = np.sum(ratios_1_to_2 > self.max_allowed_ratio * (1 + self.tolerance))
        violations_2_to_1 = np.sum(ratios_2_to_1 > self.max_allowed_ratio * (1 + self.tolerance))
        total_violations = violations_1_to_2 + violations_2_to_1
        
        # Test passes if no significant violations
        test_passed = total_violations == 0 and max_ratio <= self.max_allowed_ratio * (1 + self.tolerance)
        
        results = {
            'test_passed': test_passed,
            'max_ratio_found': max_ratio,
            'max_allowed_ratio': self.max_allowed_ratio,
            'violations': total_violations,
            'total_bins': len(bin_edges) - 1,
            'samples1': samples1,
            'samples2': samples2,
            'prob1': prob1,
            'prob2': prob2,
            'ratios_1_to_2': ratios_1_to_2,
            'ratios_2_to_1': ratios_2_to_1,
            'bin_edges': bin_edges
        }
        
        self._print_results(results)
        return results
    
    def _print_results(self, results):
        """Print test results in a readable format."""
        print(f"\nRatio Test Results:")
        print(f"Test Passed: {'✓ YES' if results['test_passed'] else '✗ NO'}")
        print(f"Max Ratio Found: {results['max_ratio_found']:.3f}")
        print(f"Max Allowed Ratio: {results['max_allowed_ratio']:.3f}")
        print(f"Violations: {results['violations']}/{results['total_bins']} bins")
        
        if not results['test_passed']:
            print(f"Mechanism may not satisfy ε={self.epsilon_claim}-DP")
        else:
            print(f"Mechanism appears to satisfy ε={self.epsilon_claim}-DP")

def test_laplace_vs_gaussian():
    """Compare Laplace and Gaussian mechanisms using ratio test."""
    
    print("\n" + "="*70)
    print("COMPARING LAPLACE VS GAUSSIAN MECHANISMS")
    print("="*70)
    
    # Test parameters
    epsilon = 0.2
    delta = 1e-5
    sensitivity = 1.0
    
    # Create neighboring datasets
    dataset1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    dataset2 = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # Removed one element
    
    def count_above_50(data):
        return sum(1 for x in data if x > 50)
    
    print(f"Dataset 1: {dataset1}")
    print(f"Dataset 2: {dataset2}")  
    print(f"Query: Count elements > 50")
    print(f"True answers: {count_above_50(dataset1)} vs {count_above_50(dataset2)}")
    print(f"Sensitivity: {sensitivity}")
    
    # Create mechanisms
    laplace_mech = LaplaceMechanism(epsilon, sensitivity)
    gaussian_mech = GaussianMechanism(epsilon, delta, sensitivity)
    
    print(f"\nLaplace scale parameter: {laplace_mech.scale:.3f}")
    print(f"Gaussian sigma parameter: {gaussian_mech.sigma:.3f}")
    print(f"Gaussian uses {gaussian_mech.sigma/laplace_mech.scale:.2f}x more noise than Laplace")
    
    # Test Laplace mechanism
    print("\n" + "-"*50)
    print("TESTING LAPLACE MECHANISM")
    print("-"*50)
    
    tester_laplace = RatioTestDP(epsilon_claim=epsilon, num_trials=15000)
    results_laplace = tester_laplace.test_mechanism(
        laplace_mech, dataset1, dataset2, count_above_50
    )
    
    # Test Gaussian mechanism (note: testing for ε-DP, not (ε,δ)-DP)
    print("\n" + "-"*50)
    print("TESTING GAUSSIAN MECHANISM")
    print("-"*50)
    print("Note: Testing Gaussian for pure ε-DP (ignoring δ parameter)")
    print("In practice, Gaussian provides (ε,δ)-DP which is slightly weaker")
    
    tester_gaussian = RatioTestDP(epsilon_claim=epsilon, num_trials=15000)
    results_gaussian = tester_gaussian.test_mechanism(
        gaussian_mech, dataset1, dataset2, count_above_50
    )
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"{'Mechanism':<15} {'Max Ratio':<12} {'Expected':<12} {'Test':<8} {'Noise Level':<12}")
    print("-"*70)
    
    expected_ratio = math.exp(epsilon)
    laplace_status = "PASS" if results_laplace['test_passed'] else "FAIL"
    gaussian_status = "PASS" if results_gaussian['test_passed'] else "FAIL"
    
    print(f"{'Laplace':<15} {results_laplace['max_ratio_found']:<12.3f} {expected_ratio:<12.3f} {laplace_status:<8} {laplace_mech.scale:<12.3f}")
    print(f"{'Gaussian':<15} {results_gaussian['max_ratio_found']:<12.3f} {expected_ratio:<12.3f} {gaussian_status:<8} {gaussian_mech.sigma:<12.3f}")
    
    # Statistical comparison
    laplace_samples = results_laplace['samples1'] + results_laplace['samples2']
    gaussian_samples = results_gaussian['samples1'] + results_gaussian['samples2']
    
    print(f"\nStatistical Properties:")
    print(f"{'Mechanism':<15} {'Mean Error':<12} {'Std Dev':<12} {'95% Range':<15}")
    print("-"*60)
    
    true_answer = count_above_50(dataset1)
    laplace_errors = [abs(s - true_answer) for s in results_laplace['samples1']]
    gaussian_errors = [abs(s - true_answer) for s in results_gaussian['samples1']]
    
    laplace_std = np.std(results_laplace['samples1'])
    gaussian_std = np.std(results_gaussian['samples1'])
    
    laplace_95_range = 1.96 * laplace_std
    gaussian_95_range = 1.96 * gaussian_std
    
    print(f"{'Laplace':<15} {np.mean(laplace_errors):<12.2f} {laplace_std:<12.2f} {laplace_95_range:<15.2f}")
    print(f"{'Gaussian':<15} {np.mean(gaussian_errors):<12.2f} {gaussian_std:<12.2f} {gaussian_95_range:<15.2f}")
    
    return results_laplace, results_gaussian

if __name__ == "__main__":
    test_laplace_vs_gaussian()
    
