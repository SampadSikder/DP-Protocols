from noise.basic_dp import LaplaceMechanism
import numpy as np
import pandas as pd

def parallel_composition_top_k(df, k, epsilon):
    """
    Find top-k items using parallel composition.
    Each query gets epsilon/k privacy budget.
    """
    # Sort pages by true traffic (this doesn't consume privacy budget)
    sorted_df = df.sort_values('traffic', ascending=False)
    
    # Take top k pages
    top_k_pages = sorted_df.head(k)
    
    # Privacy budget per query (parallel composition)
    epsilon_per_query = epsilon / k
    
    results = []
    
    for _, row in top_k_pages.iterrows():
        page = row['page']
        true_traffic = row['traffic']
        
        # Add Laplace noise to each traffic count
        mechanism = LaplaceMechanism(sensitivity=1, epsilon=epsilon_per_query)
        noisy_traffic = mechanism.add_noise(true_traffic)
        
        # Ensure non-negative traffic
        noisy_traffic = max(0, noisy_traffic)
        
        results.append((page, noisy_traffic))
    
    return results