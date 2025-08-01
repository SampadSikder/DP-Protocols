import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Any
import random

class ExponentialMechanism:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
    
    def select(self, dataset: Any, candidates: List[Any], 
               score_function: Callable, sensitivity: float) -> Any:
        """
        Select an output using the exponential mechanism
            dataset: The input dataset
            candidates: List of possible outputs to choose from
            score_function: Function that scores each candidate (higher = better)
            sensitivity: Maximum change in score function across neighboring datasets
            
        Returns:
            Selected candidate from the list
        """
        scores = [score_function(dataset, candidate) for candidate in candidates]
        
        exp_scores = [np.exp(self.epsilon * score / (2 * sensitivity)) 
                     for score in scores]
        
        total = sum(exp_scores)
        probabilities = [exp_score / total for exp_score in exp_scores]
        
        return np.random.choice(candidates, p=probabilities)


def histogram_example():
    """Example: Count query that should recover Laplace mechanism"""
    print("\n=== Histogram/Count Query Example ===")
    
    # Dataset: ages of people
    dataset = [25, 30, 28, 35, 22, 31, 29, 26, 33, 27]
    
    # Query: count people aged 25-30
    def count_query(data):
        return sum(1 for age in data if 25 <= age <= 30)
    
    true_count = count_query(dataset)
    print(f"True count: {true_count}")
    
    # Candidate outputs (possible counts)
    candidates = list(range(0, 15))  # 0 to 14
    
    def count_score(data, candidate):
        true_val = count_query(data)
        return -abs(true_val - candidate)  
    
    sensitivity = 1.0 
    
    # Test with different epsilon values
    for epsilon in [0.5, 1.0, 2.0]:
        em = ExponentialMechanism(epsilon)
        
        # Run multiple times
        results = []
        for _ in range(1000):
            result = em.select(dataset, candidates, count_score, sensitivity)
            results.append(result)
        
        mean_result = np.mean(results)
        std_result = np.std(results)
        
        print(f"\nε = {epsilon}:")
        print(f"  Mean: {mean_result:.2f} (true: {true_count})")
        print(f"  Std:  {std_result:.2f} (Laplace std should be {np.sqrt(2) * sensitivity/epsilon:.2f})")


def topk_example():
    """Example: Select top-k items privately"""
    print("\n=== Top-k Selection Example ===")
    
    # Dataset: item ratings
    dataset = {
        'item_A': [5, 4, 5, 3, 4],
        'item_B': [3, 3, 2, 4, 3], 
        'item_C': [4, 5, 4, 5, 4],
        'item_D': [2, 2, 3, 2, 1],
        'item_E': [5, 5, 5, 4, 5]
    }
    
    # Calculate true average ratings
    true_averages = {item: np.mean(ratings) for item, ratings in dataset.items()}
    print("True averages:", {k: f"{v:.2f}" for k, v in true_averages.items()})
    
    candidates = list(dataset.keys())
    
    # Scoring function: average rating
    def rating_score(data, item):
        return np.mean(data[item])
    
    sensitivity = 1.0  # Rating scale difference
    
    epsilon = 1.0
    em = ExponentialMechanism(epsilon)
    
    # Select top item multiple times
    selections = []
    for _ in range(1000):
        selected = em.select(dataset, candidates, rating_score, sensitivity)
        selections.append(selected)
    
    from collections import Counter
    counts = Counter(selections)
    print(f"\nSelection frequencies (ε = {epsilon}):")
    for item, count in counts.most_common():
        print(f"  {item}: {count/1000:.3f}")
        
        
        
if __name__ == "__main__":
    
    np.random.seed(42)
    random.seed(42)
    
    histogram_example()
    topk_example()