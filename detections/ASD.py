import numpy as np
import math
from typing import List, Set, Dict
from scipy.stats import norm


from sympy import symbols, sqrt, solve

def norms(est, n):
    while (np.fabs(sum(est) - n) > 1):
        sum_estimate = np.sum(est)
        diff_pre = (n - sum_estimate) / len(est)
        est += diff_pre
        est = np.array([0 if x > n else x for x in est], dtype=float)
    return np.array(est, dtype=float)

def lower_point(n, epsilon, domain, est_dist, perturb_method):
    if perturb_method == 'OUE':
        q_OUE = 1 / (math.exp(epsilon) + 1)
        p = 1 / 2
        q = q_OUE
    elif perturb_method in ('OLH_User', 'OLH_Server'):
        g = int(round(math.exp(epsilon))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1 / g
    elif perturb_method in ('HST_Server', 'HST_User'):
        g = 2
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1 / g
    elif perturb_method == 'GRR':
        p = np.exp(epsilon) / (np.exp(epsilon) + domain - 1)
        q = 1.0 / (np.exp(epsilon) + domain - 1)
    else:
        p = 0
        q = 0

    r_h = np.max(est_dist) / sum(est_dist)

    max_std = sqrt(n*q*(1-q))
    max_std = max_std / (p - q)

    x = symbols('x')
    y = solve(2 * sqrt(n*(x * p * (1 - p) + (1 - x) * q * (1 - q))) - n * (p - q) * x, x)
    y1 = y.pop()  # symbolic root (unused downstream)
    y_high = 3 * sqrt(n * q * (1 - q)) / (p - q)
    y3 = y_high
    y2 = (1 * sqrt(n*(y1 * p * (1 - p) + (1 - y1) * q * (1 - q))) + n * (p - q) * y1) / (p - q)
    y4 = (1 * sqrt(n*(y1 * p * (1 - p) + (1 - y1) * q * (1 - q))) / (p - q) + n * y1)
    return y1, y4, float(max_std)  # cast max_std to float

def find_confidence_level(n, d, s, est, confidence_start=0.90, confidence_end=0.9999, step=0.0001):
    lambdas = 0.02
    n_percent = lambdas * n
    for confidence in np.arange(confidence_start, confidence_end + step, step):
        z_score = norm.ppf((1 + confidence) / 2)
        x = z_score * s
        est_min = np.min(est)
        l = len([num for num in est if num < abs(est_min)])
        lhs = x * l * (1 - confidence) * 0.5
        if lhs < n_percent:
            return confidence, lambdas, z_score
    return None, None, 3.8906  # very high z if nothing found

def _pq(epsilon: float, k: int, method: str):
        if method == 'GRR':
            p = math.exp(epsilon) / (math.exp(epsilon) + k - 1)
            q = 1.0 / (math.exp(epsilon) + k - 1)
        elif method == 'OUE':
            p, q = 0.5, 1.0 / (math.exp(epsilon) + 1.0)
        elif method in ('OLH_User','OLH_Server'):
            g = int(round(math.exp(epsilon))) + 1
            p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
            q = 1.0 / g
        elif method in ('HST_Server','HST_User'):
            g = 2
            p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
            q = 1.0 / g
        else:
            raise ValueError(f"Unsupported perturb_method: {method}")
        return p, q

def _sigma_counts(n: int, epsilon: float, k: int, method: str) -> float:
        p, q = _pq(epsilon, k, method)
        return math.sqrt(n) * math.sqrt(p*(1-p) + (k-1)*q*(1-q)) / (p - q)

def ASD_detect(est_dist_counts, n, epsilon, domain,
                   perturb_method='GRR', alpha=0.01, lambda_frac=0.02, verbose=True):
        est = norms(np.array(est_dist_counts, dtype=float), n)

        sigma = _sigma_counts(n, epsilon, domain, perturb_method)
        # Family-wise control across k bins:
        z = norm.ppf(1 - alpha / (2 * domain))  # Bonferroni
        # Robust baseline across bins (avoid punishing skewed truths too aggressively)
        baseline = float(np.median(est))
        cap = baseline + z * sigma

        over = np.maximum(0.0, est - cap)
        overflow_mass = float(over.sum())
        suspicious_idx = np.where(over > 0)[0]

        flag = 1 if overflow_mass > (lambda_frac * n) else 0

        if verbose:
            print(f"sigma_counts: {sigma:.3f}, z: {z:.3f}, baseline: {baseline:.1f}")
            print(f"cap: {cap:.1f}, overflow_mass: {overflow_mass:.1f}")
            print(f"suspicious_idx: {list(suspicious_idx)}")
            print("Attacking!" if flag else "No attack detected.")

        return {
            'flag': flag,
            'est_counts': est,
            'z_score': float(z),
            'sigma_counts': float(sigma),
            'baseline': baseline,
            'threshold': float(cap),
            'overflow_mass': overflow_mass,
            'suspicious_idx': suspicious_idx
        }

def asd_correct(est_counts: np.ndarray, n: int, cap: float, suspicious_idx: np.ndarray) -> np.ndarray:
    est = est_counts.astype(float).copy()
    if len(suspicious_idx) == 0:
        return est

    over = np.maximum(0.0, est[suspicious_idx] - cap)
    total_over = float(over.sum())
    est[suspicious_idx] -= over

    # redistribute to non-suspicious bins proportionally to their current mass
    non_idx = np.array([i for i in range(len(est)) if i not in set(suspicious_idx)])
    if len(non_idx) > 0 and total_over > 0:
        w = est[non_idx].copy()
        w = (w + 1e-9) / (w.sum() + 1e-9)  # stable
        est[non_idx] += total_over * w

    # final normalization
    est *= n / est.sum()
    return est