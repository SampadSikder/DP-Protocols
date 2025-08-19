import numpy as np
from typing import List, Union, Set
import matplotlib.pyplot as plt
from collections import Counter


class MaximalGainAttack:
    """
    Implements a Maximal Gain Attack against GRR by solving the optimization problem:
    
    max_Y ∑_{t∈T} E[Δf̂_t]
    
    where the attacker crafts perturbed values Ŷ to maximize the overall gain of target items.
    """

    def __init__(self, epsilon: float, k: int, target_items: Set[int]):
        """
        Initialize the attack by storing GRR parameters and target items.
        
        Args:
            epsilon: Privacy parameter from GRR
            k: Domain size from GRR
            target_items: Set of items the adversary wants to boost (T)
        """
        self.epsilon = epsilon
        self.k = k
        self.target_items = target_items

        self.p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        self.q = 1 / (np.exp(epsilon) + k - 1)
      
        self.optimal_strategy = {}
        self.expected_gain = 0.0
        
        print(f"\n=== Maximal Gain Attack Initialization ===")
        print(f"Domain size k: {k}")
        print(f"Privacy parameter ε: {epsilon}")
        print(f"GRR probabilities: p = {self.p:.4f}, q = {self.q:.4f}")
        print(f"Target items T: {sorted(target_items)}")
     
        self.solve_optimization_problem()

    def solve_optimization_problem(self):
        """
        Solve the optimization problem: max_Y ∑_{t∈T} E[Δf̂_t]
        """
        
        # For each true value, determine optimal reporting strategy
        for true_value in range(self.k):
            optimal_report = self.solve_for_true_value(true_value)
            self.optimal_strategy[true_value] = optimal_report
        
        # Calculate expected gain
        self.calculate_expected_gain()
        
        print(f"\n=== Optimal Attack Strategy ===")
        print("True Value | Optimal Report | Reasoning")
        print("-----------|----------------|----------")
        for y in range(self.k):
            report = self.optimal_strategy[y]
            if y in self.target_items:
                reason = "True value is target - report truthfully for max gain"
            else:
                reason = f"True value not target - report target item {report}"
            print(f"    {y}      |       {report}        | {reason}")

    def solve_for_true_value(self, true_value: int) -> int:
        """
        For a given true value, determine the optimal report to maximize
        the expected gain for target items.
        """
        if true_value in self.target_items:
            return true_value
        else:
            # If true value is not a target, always report a target item
            return np.random.choice(list(self.target_items))
        

    def calculate_expected_gain(self):
        """
        Calculate the expected gain in frequency estimation for target items
        under the optimal attack strategy.
        
        Expected gain = ∑_{y=0}^{k-1} P(Y=y) * [Gain from optimal strategy - Gain from honest GRR]
        """
        print(f"\n=== Calculating Expected Gain ===")
        
        total_gain = 0.0
        
        for true_value in range(self.k):
            # Gain from optimal attack strategy
            optimal_report = self.optimal_strategy[true_value]
            if optimal_report in self.target_items:
                attack_gain = 1.0  
            else:
                attack_gain = 0.0
            
            if true_value in self.target_items:
                honest_gain = self.p
            else:
                honest_gain = len(self.target_items) * self.q
            
            # Net gain per user with this true value
            net_gain = attack_gain - honest_gain
            
            print(f"True value {true_value}: Attack gain = {attack_gain:.3f}, "
                  f"Honest gain = {honest_gain:.3f}, Net gain = {net_gain:+.3f}")
            
            total_gain += net_gain
        
        self.expected_gain = total_gain / self.k
        print(f"\nExpected gain per adversarial user (uniform prior): {self.expected_gain:+.4f}")
        
        return self.expected_gain
    
    def calculate_gain_with_distribution(self, true_distribution: np.ndarray) -> float:
        """
        Calculate expected gain given a specific true data distribution.
        """
        total_gain = 0.0
        
        for true_value in range(self.k):
            prob_true_value = true_distribution[true_value]
            
            # Gain from optimal attack strategy
            optimal_report = self.optimal_strategy[true_value]
            if optimal_report in self.target_items:
                attack_gain = 1.0
            else:
                attack_gain = 0.0
            
            # Expected gain from honest GRR
            if true_value in self.target_items:
                honest_gain = self.p
            else:
                honest_gain = len(self.target_items) * self.q
            
            net_gain = attack_gain - honest_gain
            total_gain += prob_true_value * net_gain
        
        return total_gain
    
    def adversarial_privatize(self, true_value: int) -> int:
        return self.optimal_strategy[true_value]
    
    def compute_frequency_bias(self, n_adversarial: int, n_total: int, 
                             true_distribution: np.ndarray) -> np.ndarray:
        """
        Compute the expected bias in frequency estimation due to the attack.
        """
        adversarial_fraction = n_adversarial / n_total
        bias = np.zeros(self.k)
        
        # For each true value, compute contribution to bias
        for true_value in range(self.k):
            prob_true_value = true_distribution[true_value]
            optimal_report = self.optimal_strategy[true_value]
            
            # Expected reports under honest GRR
            honest_reports = np.full(self.k, self.q)
            honest_reports[true_value] = self.p
            
            # Reports under attack (deterministic)
            attack_reports = np.zeros(self.k)
            attack_reports[optimal_report] = 1.0
            
            # Bias contribution from users with this true value
            bias_contribution = adversarial_fraction * prob_true_value * (attack_reports - honest_reports)
            bias += bias_contribution
        
        # Convert to frequency estimation bias (accounting for GRR unbiasing)
        frequency_bias = bias / (self.p - self.q)
        
        return frequency_bias
    
    def analyze_attack_impact(self, n_adversarial: int, n_total: int, 
                            true_distribution: np.ndarray):
        """
        Analyze the theoretical impact of the attack.
        
        Args:
            n_adversarial: Number of adversarial users
            n_total: Total number of users  
            true_distribution: True distribution of values
        """
        print(f"\n=== Attack Impact Analysis ===")
        print(f"Total users: {n_total}")
        print(f"Adversarial users: {n_adversarial} ({100*n_adversarial/n_total:.1f}%)")
        print(f"True distribution: {true_distribution}")
        
        # Calculate expected gain per adversarial user
        gain_per_user = self.calculate_gain_with_distribution(true_distribution)
        print(f"Expected gain per adversarial user: {gain_per_user:+.4f}")
        
        # Calculate frequency bias
        frequency_bias = self.compute_frequency_bias(n_adversarial, n_total, true_distribution)
        
        print(f"\nExpected frequency bias:")
        print("Item | True Freq | Expected Bias | Final Freq")
        print("-----|-----------|---------------|------------")
        
        total_target_bias = 0.0
        for i in range(self.k):
            final_freq = true_distribution[i] + frequency_bias[i]
            if i in self.target_items:
                total_target_bias += frequency_bias[i]
                marker = " *"
            else:
                marker = ""
            print(f"  {i}  |   {true_distribution[i]:.3f}   |    {frequency_bias[i]:+.4f}    |   {final_freq:.3f}{marker}")
        
        print(f"\nTotal bias for target items: {total_target_bias:+.4f}")
        
        return {
            'frequency_bias': frequency_bias,
            'total_target_bias': total_target_bias,
            'gain_per_user': gain_per_user
        }

    