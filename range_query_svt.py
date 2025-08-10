from noise.basic_dp import LaplaceMechanism
import numpy as np
import pandas as pd
from parallel_composition import parallel_composition_top_k

def age_range_query(df, lower, upper):
    df1 = df[df['Age'] > lower]
    return len(df1[df1['Age'] < upper])
def create_age_range_query(): 
    lower = np.random.randint(30, 50)
    upper = np.random.randint(lower, 70)
    query_func = lambda df: age_range_query(df, lower, upper)
    return query_func, (lower, upper)

def sparse_vector(queries, df, c, T, epsilon):
    """
    Sparse Vector Technique implementation.
    
    Args:
        queries: List of query functions
        df: DataFrame to query
        c: Count of queries to return (how many queries above threshold)
        T: Threshold value
        epsilon: Privacy budget
    
    Returns:
        List of indices of queries that are above threshold
    """
    if len(queries) == 0:
        return []
    
    # Add noise to threshold
    threshold_mechanism = LaplaceMechanism(sensitivity=1, epsilon=epsilon/2)
    noisy_threshold = threshold_mechanism.add_noise(T)
    
    # Privacy budget for each query
    query_epsilon = epsilon / (2 * len(queries))
    
    indices = []
    
    for i, query in enumerate(queries):
        # Get true query result
        true_result = query(df)
        
        # Add noise to query result
        query_mechanism = LaplaceMechanism(sensitivity=1, epsilon=query_epsilon)
        noisy_result = query_mechanism.add_noise(true_result)
        
        # Check if above threshold
        if noisy_result >= noisy_threshold:
            indices.append(i)
            
            # Stop if we've found c queries above threshold
            if len(indices) >= c:
                break
    
    return indices

def range_query_svt(queries, query_ranges, df, c, T, epsilon):
    """
    Use SVT to find 'good' range queries, then return noisy results for those queries.
    
    Args:
        queries: List of query functions
        query_ranges: List of (lower, upper) tuples corresponding to each query
        df: DataFrame to query
        c: Number of queries to return
        T: Threshold for considering a query "good"
        epsilon: Total privacy budget
    
    Returns:
        List of tuples: (age_range, noisy_count) for the selected queries
    """
    # First, run Sparse Vector to get the indices of the "good" queries
    sparse_epsilon = epsilon / 2
    indices = sparse_vector(queries, df, c, T, sparse_epsilon)
    
    # Then, run the Laplace mechanism on each "good" query
    laplace_epsilon = epsilon / (2 * len(indices)) if len(indices) > 0 else epsilon / 2
    results = []
    
    for i in indices:
        true_result = queries[i](df)
        laplace_mechanism = LaplaceMechanism(sensitivity=1, epsilon=laplace_epsilon)
        noisy_result = laplace_mechanism.add_noise(true_result)
        age_range = query_ranges[i]
        results.append((age_range, noisy_result))
    
    return results

def heavy_hitters_example():
    """
    Heavy Hitters Example: Find top web pages using parallel composition.
    Given 10,000 web pages, find the noisy top 10 using parallel composition.
    """
    print("\n" + "="*60)
    print("HEAVY HITTERS EXAMPLE: Top Web Pages")
    print("="*60)
    
    # Generate synthetic web traffic data
    np.random.seed(42)
    
    # Create 10,000 web pages with different popularity distributions
    web_pages = []
    true_traffic = []
    
    # Top 50 pages: Very popular (Zipf-like distribution)
    for i in range(50):
        page_name = f"popular-site-{i+1}.com"
        traffic = int(np.random.exponential(10000) + 5000)  # High traffic
        web_pages.append(page_name)
        true_traffic.append(traffic)
    
    # Next 950 pages: Moderately popular
    for i in range(950):
        page_name = f"medium-site-{i+1}.com"
        traffic = int(np.random.exponential(2000) + 500)  # Medium traffic
        web_pages.append(page_name)
        true_traffic.append(traffic)
    
    # Remaining 9000 pages: Low traffic
    for i in range(9000):
        page_name = f"small-site-{i+1}.com"
        traffic = int(np.random.exponential(100) + 10)  # Low traffic
        web_pages.append(page_name)
        true_traffic.append(traffic)
    
    # Create DataFrame
    web_data = pd.DataFrame({
        'page': web_pages,
        'traffic': true_traffic
    })
    
    # Sort by true traffic to see actual top 10
    web_data_sorted = web_data.sort_values('traffic', ascending=False)
    
    print("True Top 10 Web Pages (by traffic):")
    for i, row in web_data_sorted.head(10).iterrows():
        print(f"{row['page']}: {row['traffic']:,} visits")
    
    # Use parallel composition to get noisy top 10
    print(f"\nUsing Parallel Composition for Private Top 10 (epsilon=1.0):")
    top_10_private = parallel_composition_top_k(web_data, k=10, epsilon=1.0)
    
    print("Private Top 10 Web Pages (with noise):")
    for i, (page, noisy_traffic) in enumerate(top_10_private, 1):
        print(f"{i}. {page}: {noisy_traffic:.0f} visits")

# Example usage and test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)  # For reproducible results
    
    # Generate sample adult dataset with ages
    n_samples = 10000
    ages = np.random.normal(40, 15, n_samples)  # Normal distribution around age 40
    ages = np.clip(ages, 18, 80)  # Clip to reasonable age range
    adult = pd.DataFrame({'Age': ages})
    
    # Generate 10 random range queries
    np.random.seed(123)  # Different seed for query generation
    query_data = [create_age_range_query() for i in range(10)]
    range_queries = [query_func for query_func, _ in query_data]
    query_ranges = [age_range for _, age_range in query_data]
    
    # Print the generated age ranges for reference
    print("Generated age ranges:")
    for i, (lower, upper) in enumerate(query_ranges):
        print(f"Query {i}: Ages {lower} to {upper-1} (range [{lower}, {upper}))")
    
    # Test the queries without privacy first
    print("\nTrue query results:")
    true_results = [q(adult) for q in range_queries]
    for i, (result, (lower, upper)) in enumerate(zip(true_results, query_ranges)):
        print(f"Query {i} [Ages {lower}-{upper-1}]: {result} people")
    
    # Run range_query_svt
    print("\nRunning SVT with parameters: c=5, T=10000, epsilon=1")
    svt_results = range_query_svt(range_queries, query_ranges, adult, c=5, T=10000, epsilon=1)
    print(f"SVT returned {len(svt_results)} results:")
    for age_range, noisy_count in svt_results:
        lower, upper = age_range
        print(f"Ages {lower}-{upper-1}: {noisy_count:.2f} people")
    
    # Test with different threshold
    print("\nRunning SVT with lower threshold: c=5, T=1000, epsilon=1")
    svt_results_2 = range_query_svt(range_queries, query_ranges, adult, c=5, T=1000, epsilon=1)
    print(f"SVT returned {len(svt_results_2)} results:")
    for age_range, noisy_count in svt_results_2:
        lower, upper = age_range
        print(f"Ages {lower}-{upper-1}: {noisy_count:.2f} people")
        
    heavy_hitters_example()
