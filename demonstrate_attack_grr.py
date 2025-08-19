import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import math
from scipy.stats import binom, chisquare
from cfo.grr import GeneralizedRandomizedResponse
from attacks.MGA import MaximalGainAttack



def simulate_attack_with_grr(attack: MaximalGainAttack, grr: GeneralizedRandomizedResponse,
                           honest_data: List[int], adversarial_data: List[int]) -> Dict:
    """
    Simulate the attack by using GRR methods for privatization.
    Gets the bias and gain from MGA class, then uses GRR for actual privatization.
    """
    print(f"\n=== Simulating Attack with GRR Privatization ===")
    print(f"Honest users: {len(honest_data)}")
    print(f"Adversarial users: {len(adversarial_data)}")
    
    # Get theoretical predictions from MGA
    true_distribution = np.bincount(honest_data + adversarial_data, minlength=grr.k) / (len(honest_data) + len(adversarial_data))
    theoretical_bias = attack.compute_frequency_bias(len(adversarial_data), 
                                                   len(honest_data) + len(adversarial_data),
                                                   true_distribution)
    theoretical_gain = attack.calculate_gain_with_distribution(true_distribution)
    
    print(f"Theoretical gain per adversarial user: {theoretical_gain:+.4f}")
    print(f"Theoretical target bias: {sum(theoretical_bias[t] for t in attack.target_items):+.4f}")
    
    # Privatize honest users using GRR
    honest_privatized = grr.privatize_dataset(honest_data)
    
    # Privatize adversarial users using attack strategy + GRR structure
    adversarial_privatized = []
    for true_val in adversarial_data:
        # Get optimal report from attack strategy
        strategic_report = attack.adversarial_privatize(true_val)
        adversarial_privatized.append(strategic_report)
    
    # Combine datasets
    all_privatized = honest_privatized + adversarial_privatized
    
    # Estimate distributions using GRR
    honest_estimated = grr.estimate_distribution(honest_privatized)
    attacked_estimated = grr.estimate_distribution(all_privatized)
    
    # Calculate empirical bias
    empirical_bias = attacked_estimated - honest_estimated
    empirical_target_bias = sum(empirical_bias[t] for t in attack.target_items)
    
    print(f"\nEmpirical vs Theoretical Results:")
    print("Item | True   | Honest Est | Attack Est | Emp. Bias | Theo. Bias")
    print("-----|--------|------------|------------|-----------|------------")
    
    for i in range(grr.k):
        marker = " *" if i in attack.target_items else ""
        print(f"  {i}  | {true_distribution[i]:.3f}  |   {honest_estimated[i]:.3f}    |   {attacked_estimated[i]:.3f}    |  {empirical_bias[i]:+.4f}   |   {theoretical_bias[i]:+.4f}{marker}")
    
    print(f"\nTarget bias - Empirical: {empirical_target_bias:+.4f}, Theoretical: {sum(theoretical_bias[t] for t in attack.target_items):+.4f}")
    
    return {
        'true_distribution': true_distribution,
        'honest_estimated': honest_estimated,
        'attacked_estimated': attacked_estimated,
        'empirical_bias': empirical_bias,
        'theoretical_bias': theoretical_bias,
        'empirical_target_bias': empirical_target_bias,
        'theoretical_target_bias': sum(theoretical_bias[t] for t in attack.target_items),
        'theoretical_gain': theoretical_gain
    }


def demonstrate_optimization_based_attack():
    """
    Demonstrate the optimization-based Maximal Gain Attack with GRR simulation.
    """
    np.random.seed(42)
    
    # Setup parameters
    k = 5  # Domain size
    epsilon = 1.0  # Privacy parameter
    target_items = {0, 1}  # Items the adversary wants to boost
    
    print("=== Optimization-Based Maximal Gain Attack with GRR ===")
    
    # Create attack (solves optimization problem internally)
    attack = MaximalGainAttack(epsilon, k, target_items)
    
    # Create GRR mechanism for privatization
    grr = GeneralizedRandomizedResponse(epsilon, k)
    
    # Generate data for simulation
    true_distribution = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    
    scenarios = [
        (1000, 9000),   
        (2000, 8000),    
        (3000, 7000),   
    ]
    
    for n_adv, n_honest in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {n_adv} adversarial, {n_honest} honest users")
        
        # Generate synthetic data
        honest_data = np.random.choice(k, size=n_honest, p=true_distribution).tolist()
        adversarial_data = np.random.choice(k, size=n_adv, p=true_distribution).tolist()

        print(f"Generated {len(honest_data)} honest and {len(adversarial_data)} adversarial users.")
        print("Honest data frequencies:", np.bincount(honest_data, minlength=k))
        print("Adversarial data frequencies:", np.bincount(adversarial_data, minlength=k))
        
        # Simulate attack using GRR methods
        results = simulate_attack_with_grr(attack, grr, honest_data, adversarial_data)


if __name__ == "__main__":
    # Run optimization-based attack demonstration with GRR simulation
    demonstrate_optimization_based_attack()