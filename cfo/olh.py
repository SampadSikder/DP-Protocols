from typing import List, Tuple, Dict, Optional
import numpy as np

class OLH:
    """
    Optimal Local Hashing (OLH) for Local Differential Privacy (frequency estimation).

    Each report contains a (hash seed) and a perturbed bucket in {0, ..., g-1}.
    Aggregation re-hashes each candidate item with the provided seed and
    accumulates matches to build an unbiased count estimate.
    """

    def __init__(self, d: int, epsilon: float, g: Optional[int] = None, prime: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        """
        Args:
            d: Domain size (number of possible items, items are 0..d-1).
            epsilon: Privacy budget (ε-LDP).
            g: (Optional) Hash range size. If None, uses optimal g = floor(e^ε) + 1.
            prime: (Optional) A prime > d used for universal hashing. If None, a default prime is chosen.
            rng: (Optional) numpy Generator for reproducibility.
        """
        if d <= 1:
            raise ValueError("d must be >= 2")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")

        self.d = d
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()

        # Choose prime > d for universal hashing: h(x) = ((a*x + b) mod P) mod g
        if prime is None:
            # A convenient large prime (2^31 - 1) works for typical d, or pick next prime above d.
            prime = 2147483647
        if prime <= d:
            raise ValueError("prime must be > d")
        self.prime = prime

        # Calculate optimal parameters
        self.g, self.p, self.q = self._calculate_optimal_params(g)

        print(f"Domain size: {d}")
        print(f"Privacy budget (ε): {epsilon}")
        print(f"Optimal parameters: g={self.g}, p={self.p:.6f}, q={self.q:.6f}")

    # ---------- Core math ----------
    def _calculate_optimal_params(self, g_override: Optional[int]) -> Tuple[int, float, float]:
        """
        Optimal g and GRR probabilities for OLH.
        g* = floor(e^ε) + 1 minimizes estimator variance (Wang et al., 2017).
        Inside the hashed domain of size g, apply k-ary randomized response:
            p = e^ε / (e^ε + g - 1),  q = 1 / (e^ε + g - 1).
        """
        exp_eps = np.exp(self.epsilon)
        if g_override is not None:
            g = int(g_override)
            if g < 2:
                raise ValueError("g must be >= 2")
        else:
            g = int(np.floor(exp_eps)) + 1
            g = max(g, 2)  # safety

        p = exp_eps / (exp_eps + g - 1.0)
        q = 1.0 / (exp_eps + g - 1.0)
        return g, p, q

    def _sample_hash(self) -> Tuple[int, int]:
        """
        Sample a universal hash h(x) = ((a*x + b) mod prime) mod g
        with a in {1..prime-1}, b in {0..prime-1}.
        """
        a = int(self.rng.integers(1, self.prime))  # nonzero
        b = int(self.rng.integers(0, self.prime))
        return a, b

    def _hash(self, x: int, a: int, b: int) -> int:
        return ((a * x + b) % self.prime) % self.g

    def encode(self, item: int, a: int, b: int) -> int:
        """
        Hash item into {0..g-1} with the provided (a,b).
        """
        if not (0 <= item < self.d):
            raise ValueError(f"Item {item} out of range [0, {self.d-1}]")
        return self._hash(item, a, b)

    def perturb(self, bucket: int) -> int:
        """
        k-ary RR in hashed domain:
            report true bucket with prob p, otherwise a random different bucket.
        """
        if not (0 <= bucket < self.g):
            raise ValueError("bucket out of range")

        if self.rng.random() < self.p:
            return bucket
        else:
            # choose uniformly among the other g-1 buckets
            r = int(self.rng.integers(0, self.g - 1))
            return r if r < bucket else r + 1 #ensure in no way we return the same bucket

    def privatize(self, item: int) -> Dict[str, int]:
        a, b = self._sample_hash()
        t = self.encode(item, a, b)  # true hashed bucket
        y = self.perturb(t)
        return {'a': a, 'b': b, 'y': y}

    # ---------- Aggregation ----------
    def estimate_counts(self, reports: List[Dict[str, int]]) -> np.ndarray:
        """
        Unbiased estimator for item counts.

        For each report (a,b,y), and for each candidate item v:
            I_v = 1{ y == h(v) }.
        Let S_v = sum_reports I_v.
        Then E[I_v] = f_v * p + (1 - f_v) * [ q + (p - q)/g ]  (due to hashing collision)
        Solve for f_v (count n_v):
            n_v = (S_v - n * [ q + (p - q)/g ]) / ( (p - q) * (g-1)/g )
        """
        n = len(reports)
        if n == 0:
            return np.zeros(self.d, dtype=float)

        # Precompute constants
        adj_const = (self.q + (self.p - self.q) / self.g)    # = q + (p - q)/g
        denom = (self.p - self.q) * (self.g - 1.0) / self.g  # = (p - q)*(1 - 1/g)

        if abs(denom) < 1e-12:
            # extremely small ε or pathological g
            return np.zeros(self.d, dtype=float)

        # Vectorized: for each report, compute matches for all v
        estimated = np.zeros(self.d, dtype=float)

        # To avoid O(n*d) in very large domains, you would shard or sketch; here we keep it simple.
        for rep in reports:
            a, b, y = rep['a'], rep['b'], rep['y']
            # Compute h(v) for all v with this (a,b), compare to y
            v = np.arange(self.d, dtype=np.int64)
            hv = ( (a * v + b) % self.prime ) % self.g
            matches = (hv == y).astype(np.float64)
            estimated += matches

        # Debias
        estimated = (estimated - n * adj_const) / denom

        # Clamp to [0, n] to avoid tiny negative noise
        estimated = np.clip(estimated, 0.0, float(n))
        return estimated

    def calculate_variance(self) -> float:
        """
        Returns the per-report indicator variance term Var[I_v] needed
        for analytic variance; full closed-form for count variance depends on n and d.
        We expose the basic ingredients users often want to log/compare.
        """
        # For an arbitrary report and fixed item v, the probability a report matches h(v) when v is NOT the true item:
        p_bg = self.q + (self.p - self.q) / self.g
        # When v IS the true item: prob of match is p.
        # Aggregate variance of the unbiased estimator over n users would combine these.
        # Here we return (p, p_bg, g) for downstream analysis or logging.
        return {
            "g": self.g,
            "p_true_match": self.p,
            "p_background_match": p_bg
        }
        
        
def demonstrate_olh():
    # Step 1: Parameters
    d = 10              # Domain size: items are 0..9
    epsilon = 1.0       # Privacy budget
    n_users = 1000      # Number of simulated users

    # Step 2: Simulate a "true" dataset
    rng = np.random.default_rng(42)
    true_items = rng.integers(0, d, size=n_users)  # random user choices

    # Step 3: Create OLH mechanism
    olh = OLH(d=d, epsilon=epsilon, rng=rng)

    # Step 4: Privatize each user's item
    reports = [olh.privatize(item) for item in true_items]

    # Step 5: Estimate counts
    est_counts = olh.estimate_counts(reports)

    # Step 6: Show results
    print("\n=== OLH Demonstration ===")
    print(f"True counts:      {np.bincount(true_items, minlength=d)}")
    print(f"Estimated counts: {np.round(est_counts, 2)}")

    # Step 7: Variance-related parameters
    print("\nVariance info:", olh.calculate_variance())
    
    
if __name__ == "__main__":
    demonstrate_olh()    