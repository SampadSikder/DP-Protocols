import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt



class GeneralizedRandomizedResponse:
    """
    Implements Generalized Randomized Response (GRR) for differential privacy.
    
    Args:
        epsilon: Privacy budget
        k: domain size
    """
    
    def __init__(self, epsilon: float, k: int):
        self.epsilon = epsilon
        self.k = k
        
        self.p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)  # Prob of reporting true value
        self.q = 1 / (np.exp(epsilon) + k - 1)                # Prob of reporting each other value
        
        print(f"GRR initialized with k={k}, ε={epsilon}")
        print(f"Probability of reporting true value: {self.p:.4f}")
        print(f"Probability of reporting each other value: {self.q:.4f}")
    
    def privatize(self, true_value: int) -> int:
        """
        Apply GRR mechanism to a single value.
        """
        if not (0 <= true_value < self.k):
            raise ValueError(f"Value must be between 0 and {self.k-1}")
        
        # Create probability distribution
        probs = np.full(self.k, self.q)  
        probs[true_value] = self.p      
        
        # Sample from this distribution
        return np.random.choice(self.k, p=probs)
    
    
    def privatize_dataset(self, data: List[int]) -> List[int]:
        perturbed_vector = [self.privatize_value(val) for val in data]
        print(f"Privatized dataset: {perturbed_vector[:10]}... (showing first 10 samples)")
        return [self.privatize_value(val) for val in data]

    
    def estimate_distribution(self, privatized_data: List[int]) -> np.ndarray:
        """
        Estimate the true distribution from privatized data using unbiased estimator.
        """
        n = len(privatized_data)
        
        # Count frequencies in privatized data
        privatized_counts = np.bincount(privatized_data, minlength=self.k)
        privatized_freq = privatized_counts / n
        
        # Unbiased estimator: (observed_freq - q) / (p - q)
        estimated_freq = (privatized_freq - self.q) / (self.p - self.q)
        
        # Ensure non-negative (can be slightly negative due to noise)
        estimated_freq = np.maximum(estimated_freq, 0)
        
        # Normalize to ensure probabilities sum to 1
        estimated_freq = estimated_freq / np.sum(estimated_freq)
        
        return estimated_freq
    
    def calculate_variance(self, n: int) -> float:
        """
        Calculate the variance of the frequency estimator.
        """
        return (self.p * (1 - self.p) + (self.k - 1) * self.q * (1 - self.q)) / (n * (self.p - self.q)**2)
    
    def privacy_analysis(self):
        """Print privacy analysis of the mechanism."""
        print("\n=== Privacy Analysis ===")
        print(f"Privacy parameter ε = {self.epsilon}")
        print(f"For any two values v1, v2 in domain:")
        print(f"This mechanism satisfies {self.epsilon}-local differential privacy.")
        

def demonstrate_grr():

    np.random.seed(42) 
    
    k = 5  # Domain size (e.g., ratings 0-4)
    epsilon = 1.0  # Privacy parameter
    n_samples = 10000
    
    # Create GRR mechanism
    grr = GeneralizedRandomizedResponse(epsilon, k)
    grr.privacy_analysis()
    
    # Generate synthetic true data (skewed distribution)
    true_distribution = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  
    true_data = np.random.choice(k, size=n_samples, p=true_distribution)
    
    print(f"True Data (n={n_samples}): {true_data[:10]}... (showing first 10 samples)")
    
    # Apply GRR privatization
    privatized_data = grr.privatize_dataset(true_data.tolist())
    
    # Estimate original distribution
    estimated_distribution = grr.estimate_distribution(privatized_data)
    
    # Calculate empirical distribution from privatized data (biased)
    empirical_distribution = np.bincount(privatized_data, minlength=k) / len(privatized_data)
    
    # Results
    print(f"\n=== Results (n={n_samples}) ===")
    print("Value | True   | Privatized | Estimated | Error")
    print("------|--------|------------|-----------|-------")
    for i in range(k):
        error = abs(estimated_distribution[i] - true_distribution[i])
        print(f"  {i}   | {true_distribution[i]:.3f}  |   {empirical_distribution[i]:.3f}    |   {estimated_distribution[i]:.3f}   | {error:.3f}")
    
    # Calculate theoretical variance
    variance = grr.calculate_variance(n_samples)
    print(f"\nTheoretical standard error per estimate: {np.sqrt(variance):.4f}")
    print(f"Mean absolute error: {np.mean(np.abs(estimated_distribution - true_distribution)):.4f}")
    
    return true_distribution, empirical_distribution, estimated_distribution


if __name__ == "__main__":
    
    demonstrate_grr()