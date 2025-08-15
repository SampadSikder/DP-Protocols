
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class OUE:
    """
    Optimized Unary Encoding (OUE) for Local Differential Privacy
    
    This implementation focuses on minimizing the variance of the count function c(v)
    by using optimal perturbation probabilities.
    """
    
    def __init__(self, d: int, epsilon: float):
        """
        Initialize OUE mechanism
        
        Args:
            d: Domain size (number of possible items)
            epsilon: Privacy budget (differential privacy parameter)
        """
        self.d = d
        self.epsilon = epsilon
        
        # Calculate optimal perturbation probabilities to minimize variance
        self.p, self.q = self._calculate_optimal_probabilities()
        
        print(f"Domain size: {d}")
        print(f"Privacy budget (ε): {epsilon}")
        print(f"Optimal probabilities: p={self.p:.4f}, q={self.q:.4f}")

    def _calculate_optimal_probabilities(self) -> Tuple[float, float]:
        """
        Calculate optimal perturbation probabilities for OUE to minimize variance
        
        Returns:
            Tuple of (p, q) probabilities
        """
        exp_eps = np.exp(self.epsilon)
        p = 1.0 / (1.0 + exp_eps)
        q = 1.0 / (1.0 + exp_eps)
        return p, q
    
    
    def encode(self, item: int) -> np.ndarray:
        """
        Args:
            item: Item to encode (0 to d-1)
            
        Returns:
            One-hot encoded vector of length d
        """
        if item < 0 or item >= self.d:
            raise ValueError(f"Item {item} out of range [0, {self.d-1}]")
        
        # Create one-hot encoding
        vector = np.zeros(self.d, dtype=int)
        vector[item] = 1
        return vector
    
    def perturb(self, vector: np.ndarray) -> np.ndarray:
        """
        Args:
            vector: One-hot encoded vector
            
        Returns:
            Perturbed vector
        """
        perturbed = vector.copy()
        
        for i in range(len(vector)):
            if vector[i] == 1:
                # Flip from 1 to 0 with probability p
                if np.random.random() < self.p:
                    perturbed[i] = 0
            else:  # vector[i] == 0
                # Flip from 0 to 1 with probability q
                if np.random.random() < self.q:
                    perturbed[i] = 1
        
        return perturbed
    
    def privatize(self, item: int) -> np.ndarray:
        """
        Complete OUE privatization: encode then perturb
        
        Args:
            item: Item to privatize
            
        Returns:
            Privatized vector
        """
        encoded = self.encode(item)
        return self.perturb(encoded)
    
    
    def estimate_counts(self, privatized_data: List[np.ndarray]) -> np.ndarray:
        """
        Args:
            privatized_data: List of privatized vectors
            
        Returns:
            Estimated counts for each item
        """
        n = len(privatized_data)
        if n == 0:
            return np.zeros(self.d)
        
        # Sum all privatized vectors
        sum_vectors = np.sum(privatized_data, axis=0)
        # Calculate estimated counts using unbiased estimator
        denominator = 1 - self.p - self.q
        
        if abs(denominator) < 1e-10:
            print("Warning: denominator close to zero, estimates may be unstable")
            return np.zeros(self.d)
        
        estimated_counts = (sum_vectors - n * self.q) / denominator
        
        # Ensure non-negative counts
        estimated_counts = np.maximum(estimated_counts, 0)
        
        return estimated_counts
    
    def calculate_variance(self) -> float:
        """
        Calculate theoretical variance of the count estimator
        
        Returns:
            Variance of c(v) for OUE
        """
        # Theoretical variance for OUE: Var[c(v)] = (p + q - p*q) / (1 - p - q)²
        numerator = self.p * (1 - self.p) + self.q * (1 - self.q)
        denominator = (1 - self.p - self.q) ** 2
        
        if abs(denominator) < 1e-10:
            return float('inf')
        
        return numerator / denominator
    
def demonstrate_oue():
    """Demonstrate OUE mechanism with variance analysis"""
    
    # Parameters
    d = 10  # Domain size
    epsilon = 1.0  # Privacy budget
    n_users = 1000  # Number of users
    
    # Initialize OUE
    oue = OUE(d, epsilon)
    
    # Generate synthetic data (items 0 to d-1 with different frequencies)
    true_items = np.random.choice(d, n_users, p=np.random.dirichlet(np.ones(d)))
    true_counts = np.bincount(true_items, minlength=d)
    
    print(f"\nTrue counts: {true_counts}")
    
    # Privatize data - create matrix of shape (n_users, d)
    privatized_matrix = np.array([oue.privatize(item) for item in true_items])
    
    # Estimate counts
    estimated_counts = oue.estimate_counts(privatized_matrix)
    print(f"Estimated counts: {estimated_counts.astype(int)}")
    
    # Calculate theoretical and empirical variance
    theoretical_variance = oue.calculate_variance()
    
    # Run multiple experiments to estimate empirical variance
    n_experiments = 100
    all_estimates = []
    
    for _ in range(n_experiments):
        # Create privatized matrix for this experiment
        exp_privatized_matrix = np.array([oue.privatize(item) for item in true_items])
        exp_estimates = oue.estimate_counts(exp_privatized_matrix)
        all_estimates.append(exp_estimates)
    
    all_estimates = np.array(all_estimates)
    empirical_variance = np.var(all_estimates, axis=0)
    
    print(f"\nVariance Analysis:")
    print(f"Theoretical variance per item: {theoretical_variance:.4f}")
    print(f"Empirical variance (avg): {np.mean(empirical_variance):.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot counts comparison
    x = range(d)
    ax1.bar([i-0.2 for i in x], true_counts, width=0.4, label='True counts', alpha=0.7)
    ax1.bar([i+0.2 for i in x], estimated_counts, width=0.4, label='Estimated counts', alpha=0.7)
    ax1.set_xlabel('Item')
    ax1.set_ylabel('Count')
    ax1.set_title('True vs Estimated Counts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot variance comparison
    ax2.bar(x, empirical_variance, alpha=0.7, label='Empirical variance')
    ax2.axhline(y=theoretical_variance, color='red', linestyle='--', 
                label=f'Theoretical variance: {theoretical_variance:.2f}')
    ax2.set_xlabel('Item')
    ax2.set_ylabel('Variance')
    ax2.set_title('Variance per Item')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return oue, true_counts, estimated_counts


if __name__ == "__main__":
    print("=== Optimized Unary Encoding (OUE) Demonstration ===")
    
    # Main demonstration
    oue, true_counts, estimated_counts = demonstrate_oue()