import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import math
from scipy.stats import binom, chisquare
from cfo.oue import OUE
from cfo.olh import OLH
from attacks.APA import AdaptivePatternAttack
from detections.diffstats import DiffStats
from cfo.oue import OUE 
from detections.improved import ImprovedDiffStats


def demonstrate_diffstats_with_apa_oue(
    domain_size: int = 10,
    epsilon: float = 1.0,
    n_honest_users: int = 200,
    n_fake_users: int = 50,
    target_item: int = 8,
    seed: int = 42,
):
    """Demonstration of Diffstats vs APA under OUE using the *new* API.

    Returns: (diffstats, detection_result, accuracy_metrics)
    """
    print("=== DiffStats Detection against APA on OUE (new API) ===")
    print(f"Domain size={domain_size}, ε={epsilon}, honest={n_honest_users}, fake={n_fake_users}, target={target_item}")

    np.random.seed(seed)
    oue = OUE(domain_size, epsilon)

    # Honest users
    honest_reports = []
    for _ in range(n_honest_users):
        true_item = np.random.randint(0, domain_size)
        report_vec = oue.privatize(true_item)   # np.ndarray length D
        honest_reports.append([report_vec])     # exactly one report per user

    # APA fake users
    apa = AdaptivePatternAttack(oue, target_item, n_fake_users, "OUE")
    fake_reports = [[r] for r in apa.generate_fake_reports()]  # wrap each as [report]

    # Combine & truth set of fake IDs
    all_reports = honest_reports + fake_reports
    true_fake_users = set(range(n_honest_users, n_honest_users + n_fake_users))

    # New Diffstats constructor/signature
    diffstats = DiffStats(
        mechanism_type="oue",
        mechanism_params={"epsilon": epsilon},
        domain_size=domain_size,
        top_L=5,
        max_combo_size=3,
        max_outer_iterations=8,
    )

    detection_result = diffstats.detect_fake_users(all_reports)
    accuracy_metrics = DiffStats.evaluate_detection_accuracy(detection_result, true_fake_users)
    return diffstats, detection_result, accuracy_metrics


def demonstrate_diffstats_with_apa_olh(
    domain_size: int = 10,
    epsilon: float = 1.0,
    n_honest_users: int = 200,
    n_fake_users: int = 50,
    target_item: int = 8,
    seed: int = 43,
):
    """Demonstration of Diffstats vs APA under OLH using the *new* API."""
    print("=== DiffStats Detection against APA on OLH (new API) ===")
    print(f"Domain size={domain_size}, ε={epsilon}, honest={n_honest_users}, fake={n_fake_users}, target={target_item}")

    np.random.seed(seed)
    olh = OLH(domain_size, epsilon)

    # Honest users
    honest_reports = []
    for _ in range(n_honest_users):
        true_item = np.random.randint(0, domain_size)
        report_dict = olh.privatize(true_item)  # expects {'a','b','y'}
        honest_reports.append([report_dict])    # exactly one report per user

    # APA fake users
    apa = AdaptivePatternAttack(olh, target_item, n_fake_users, "OLH")
    fake_reports = [[r] for r in apa.generate_fake_reports()]

    # Combine & truth set of fake IDs
    all_reports = honest_reports + fake_reports
    true_fake_users = set(range(n_honest_users, n_honest_users + n_fake_users))

    diffstats = DiffStats(
        mechanism_type="olh",
        mechanism_params={"epsilon": epsilon, "g": getattr(olh, "g", int(np.floor(np.exp(epsilon))) + 1), "prime": getattr(olh, "prime", 2147483647)},
        domain_size=domain_size,
        top_L=5,
        max_combo_size=3,
        max_outer_iterations=8,
    )

    detection_result = diffstats.detect_fake_users(all_reports)
    accuracy_metrics = DiffStats.evaluate_detection_accuracy(detection_result, true_fake_users)
    return diffstats, detection_result, accuracy_metrics


    """Demonstrate the enhanced DiffStats against APA"""
    
    # Parameters
    domain_size = 20
    epsilon = 1.0
    n_honest = 300
    n_fake = 80
    target_item = 15
    
    print(f"Demo: domain={domain_size}, honest={n_honest}, fake={n_fake}, target={target_item}")
    
    # Setup
    np.random.seed(42)
    oue = OUE(domain_size, epsilon)
    
    # Generate honest users
    honest_reports = []
    for _ in range(n_honest):
        true_item = np.random.randint(0, domain_size)
        report = oue.privatize(true_item)
        honest_reports.append([report])
    
    # Generate APA fake users
    apa = AdaptivePatternAttack(oue, target_item, n_fake, "OUE")
    fake_reports = [[r] for r in apa.generate_fake_reports()]
    
    # Combine data
    all_reports = honest_reports + fake_reports
    true_fake_users = set(range(n_honest, n_honest + n_fake))
    
    # Enhanced detection
    enhanced_diffstats = ImprovedDiffStats(
        mechanism_type="oue",
        mechanism_params={"epsilon": epsilon},
        domain_size=domain_size,
        top_L=8,
        max_combo_size=3,
        statistical_threshold=0.05,
    )
    
    result = enhanced_diffstats.detect_fake_users(all_reports)
    
    # Evaluate
    detected = result['fake_users']
    tp = len(detected & true_fake_users)
    fp = len(detected - true_fake_users)
    fn = len(true_fake_users - detected)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n=== Enhanced DiffStats Results ===")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    return result, {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }

if __name__ == "__main__":
    # Example usage for OUE
    diffstats_oue, result_oue, metrics_oue = demonstrate_diffstats_with_apa_oue()
    print("OUE Detection Result:", result_oue)
    print("OUE Accuracy Metrics:", metrics_oue)


   