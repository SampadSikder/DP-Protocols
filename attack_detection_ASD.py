import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, chisquare

from attacks import AdaptivePatternAttack
from cfo.grr import GeneralizedRandomizedResponse
from attacks.MGA import MaximalGainAttack
from detections.ASD import ASD_detect, asd_correct

def run_experiment_with_asd(seed=42):
    np.random.seed(seed)

    # Problem setup
    k = 8
    epsilon = 1.0
    n_total = 20000

    # True distribution (skewed, but sums to 1)
    true_dist = np.array([0.02, 0.08, 0.10, 0.35, 0.20, 0.15, 0.07, 0.03], dtype=float)
    true_dist = true_dist / true_dist.sum()

    # Attack settings
    target_items = {3}           # attacker boosts item 3 (already popular)
    frac_adv = 0.15             # 15% adversarial users
    n_adv = int(frac_adv * n_total)
    n_honest = n_total - n_adv

    # Mechanism + attack
    grr = GeneralizedRandomizedResponse(epsilon, k)
    mga = MaximalGainAttack(epsilon, k, target_items)

    # Sample true data
    true_data = np.random.choice(k, size=n_total, p=true_dist)

    # Split users
    honest_true = true_data[:n_honest]
    adv_true = true_data[n_honest:]

    # Honest users use GRR
    honest_reports = [grr.privatize(v) for v in honest_true]

    # Adversaries report using MGA's optimal (deterministic) strategy
    adv_reports = [mga.adversarial_privatize(v) for v in adv_true]

    # Combine reports
    all_reports = honest_reports + adv_reports

    # Estimation under GRR's unbiased estimator
    est_probs = grr.estimate_distribution(all_reports)      # probabilities
    est_counts = est_probs * n_total                         # convert to counts for ASD

    # Baseline errors
    def l1(a, b): return float(np.abs(a - b).sum())
    def l2(a, b): return float(np.sqrt(((a - b) ** 2).sum()))
    base_L1 = l1(est_probs, true_dist)
    base_L2 = l2(est_probs, true_dist)
    base_max = float(np.max(np.abs(est_probs - true_dist)))

    print("\n=== Baseline (under attack, no correction) ===")
    print(f"L1 error:  {base_L1:.4f}")
    print(f"L2 error:  {base_L2:.4f}")
    print(f"L∞ error:  {base_max:.4f}")

    # ASD detection
    asd_info = ASD_detect(est_counts, n_total, epsilon, k, 'GRR', verbose=True)

    # ASD correction (cap & redistribute)
    corrected_counts = asd_correct(
        asd_info['est_counts'],
        n_total,
        asd_info['threshold'],
        asd_info['suspicious_idx']
    )
    corrected_probs = corrected_counts / corrected_counts.sum()

    # Post-correction errors
    corr_L1 = l1(corrected_probs, true_dist)
    corr_L2 = l2(corrected_probs, true_dist)
    corr_max = float(np.max(np.abs(corrected_probs - true_dist)))

    print("\n=== After ASD correction ===")
    print(f"L1 error:  {corr_L1:.4f}  (recovery: {(base_L1 - corr_L1):+.4f})")
    print(f"L2 error:  {corr_L2:.4f}  (recovery: {(base_L2 - corr_L2):+.4f})")
    print(f"L∞ error:  {corr_max:.4f}  (recovery: {(base_max - corr_max):+.4f})")
    print(f"Attack detected? {'YES' if asd_info['flag']==1 else 'NO'}")
    print(f"Suspicious bins: {list(asd_info['suspicious_idx'])}")
    print(f"ASD threshold (z*σ_max): {asd_info['threshold']:.3f}")

    # Return artifacts in case you want to plot
    return {
        'true_dist': true_dist,
        'est_probs': est_probs,
        'corrected_probs': corrected_probs,
        'suspicious_idx': asd_info['suspicious_idx'],
        'threshold': asd_info['threshold'],
        'errors': {
            'baseline': {'L1': base_L1, 'L2': base_L2, 'Linf': base_max},
            'corrected': {'L1': corr_L1, 'L2': corr_L2, 'Linf': corr_max}
        }
    }

if __name__ == "__main__":
    run_experiment_with_asd()