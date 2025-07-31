
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

class LaplaceMechanism:
    
    def __init__(self, sensitivity, epsilon):
        """
        Initialize Laplace with sensitivity and epsilon.
        """
        
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")
            
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon 

    def add_noise(self, value):
        """
        Add Laplace noise to a given value
        Scale to control how spread out the noise is.
        """
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def get_privacy_params(self):
        """Return the privacy parameters this mechanism satisfies."""
        return {"epsilon": self.epsilon, "delta": 0, "type": "pure_DP"}
    
class GaussianMechanism:
    """
    Implements the Gaussian Mechanism for (ε, δ)-differential privacy.
    
    The Gaussian mechanism adds noise drawn from Gaussian distribution
    with variance σ² where σ ≥ √(2 ln(1.25/δ)) * sensitivity / ε.
    """
    
    def __init__(self, epsilon, delta, sensitivity):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta < 0 or delta >= 1:
            raise ValueError("Delta must be in the range [0, 1)")
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")
        
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # Standard deviation for Gaussian noise
        self.sigma = math.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    
    def add_noise(self, true_value):
        """
        Add Gaussian noise to a true value.
        """
        noise = np.random.normal(0, self.sigma, size=np.array(true_value).shape)
        return true_value + noise
    
    def get_privacy_params(self):
        """Return the privacy parameters this mechanism satisfies."""
        return {
            "epsilon": self.epsilon, 
            "delta": self.delta, 
            "sigma": self.sigma,
            "type": "approximate_DP"
        }
        
def demonstrate_mechanisms():
    """
    Demonstrate the Laplace and Gaussian mechanisms.
    """
    sensitivity = 1.0
    epsilon = 0.5
    delta = 0.01
    
    laplace_mechanism = LaplaceMechanism(sensitivity, epsilon)
    gaussian_mechanism = GaussianMechanism(epsilon, delta, sensitivity)
    
    ## An example count query where we count how many > 5
    
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    true_count = sum(1 for x in data if x > 5)
    
    print(f"\nLaplace Mechanism (ε=0.5): Sequential Composition")
    for i in range(5):
        noisy_count = laplace_mechanism.add_noise(true_count)
        print(f"  Run {i+1}: {noisy_count:.2f}")
        
    print(f"\nGaussian Mechanism (ε=1.0, δ=1e-5):")
    print(f"  Required σ: {gaussian_mechanism.sigma:.3f}")
    for i in range(5):
        noisy_count = gaussian_mechanism.add_noise(true_count)
        print(f"  Run {i+1}: {noisy_count:.2f}")


def privacy_utility_tradeoff():
    """Demonstrate privacy-utility tradeoff for different epsilon values."""
    
    true_value = 100
    sensitivity = 1.0
    delta = 1e-5
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("Privacy-Utility Tradeoff Analysis")
    print("True value: 100")
    print(f"{'Epsilon':<8} {'Laplace':<15} {'Gaussian':<15} {'L_Error':<10} {'G_Error':<10}")
    print("-" * 65)
    
    for eps in epsilon_values:
        laplace_mech = LaplaceMechanism(eps, sensitivity)
        gaussian_mech = GaussianMechanism(eps, delta, sensitivity)
        
        # Generate multiple samples to estimate average error
        n_runs = 1000
        laplace_results = [laplace_mech.add_noise(true_value) for _ in range(n_runs)]
        gaussian_results = [gaussian_mech.add_noise(true_value) for _ in range(n_runs)]
        
        laplace_error = np.mean([abs(r - true_value) for r in laplace_results])
        gaussian_error = np.mean([abs(r - true_value) for r in gaussian_results])
        
        laplace_sample = laplace_mech.add_noise(true_value)
        gaussian_sample = gaussian_mech.add_noise(true_value)
        
        print(f"{eps:<8} {laplace_sample:<15.2f} {gaussian_sample:<15.2f} {laplace_error:<10.2f} {gaussian_error:<10.2f}")

def compare_noise_distributions():
    """Compare the noise distributions of both mechanisms."""
    
    epsilon = 1.0
    delta = 1e-5
    sensitivity = 1.0
    
    laplace_mech = LaplaceMechanism(epsilon, sensitivity)
    gaussian_mech = GaussianMechanism(epsilon, delta, sensitivity)
    
    # Generate noise samples
    n_samples = 10000
    laplace_noise = [np.random.laplace(0, laplace_mech.scale) for _ in range(n_samples)]
    gaussian_noise = [np.random.normal(0, gaussian_mech.sigma) for _ in range(n_samples)]
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(laplace_noise, bins=50, alpha=0.7, density=True, label='Laplace')
    plt.hist(gaussian_noise, bins=50, alpha=0.7, density=True, label='Gaussian')
    plt.xlabel('Noise Value')
    plt.ylabel('Density')
    plt.title('Noise Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    x = np.linspace(-5, 5, 1000)
    laplace_pdf = stats.laplace.pdf(x, scale=laplace_mech.scale)
    gaussian_pdf = stats.norm.pdf(x, scale=gaussian_mech.sigma)
    
    plt.plot(x, laplace_pdf, label=f'Laplace (scale={laplace_mech.scale:.2f})', linewidth=2)
    plt.plot(x, gaussian_pdf, label=f'Gaussian (σ={gaussian_mech.sigma:.2f})', linewidth=2)
    plt.xlabel('Noise Value')
    plt.ylabel('Probability Density')
    plt.title('Theoretical PDFs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nNoise Statistics:")
    print(f"Laplace - Mean: {np.mean(laplace_noise):.3f}, Std: {np.std(laplace_noise):.3f}")
    print(f"Gaussian - Mean: {np.mean(gaussian_noise):.3f}, Std: {np.std(gaussian_noise):.3f}")
    print(f"Laplace theoretical std: {laplace_mech.scale * math.sqrt(2):.3f}")
    print(f"Gaussian theoretical std: {gaussian_mech.sigma:.3f}")      
        
if __name__ == "__main__":
    demonstrate_mechanisms()
    print("\n"+"-" * 65)
    privacy_utility_tradeoff()
    
    # Optionally, you can plot the results or further analyze the noise distributions.
    #compare_noise_distributions()