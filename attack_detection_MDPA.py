import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, chisquare

from attacks import AdaptivePatternAttack
from cfo.oue import OUE
from cfo.olh import OLH
from detections.MDPA import DiffStats

def reports_to_support_matrix_oue(oue, reports: List[np.ndarray]) -> np.ndarray:
   
    support = np.asarray(reports, dtype=int)
    if support.ndim != 2 or support.shape[1] != oue.d:
        raise ValueError(f"OUE support shape mismatch: expected (*,{oue.d}), got {support.shape}")
    return (support > 0).astype(int)


def olh_hash_all_v(olh, a: int, b: int) -> np.ndarray:
    v = np.arange(olh.d, dtype=np.int64)
    return ((a * v + b) % olh.prime) % olh.g


@dataclass
class OLHDerived:
   
    one_count: np.ndarray
    matches_target: np.ndarray
    support_full: Optional[np.ndarray] = None


def derive_features_olh(
    olh, reports: List[Dict[str, int]],
    target_item: int,
    need_full_support: bool = False
) -> OLHDerived:
    n = len(reports)
    one_count = np.zeros(n, dtype=int)
    matches_target = np.zeros(n, dtype=int)
    support_full = None
    if need_full_support:
        support_full = np.zeros((n, olh.d), dtype=int)

    for i, rep in enumerate(reports):
        a, b, y = rep['a'], rep['b'], rep['y']
        hv = olh_hash_all_v(olh, a, b)
        mask = (hv == y)
        one_count[i] = int(mask.sum())

        # target bucket match?
        target_bucket = hv[target_item]
        matches_target[i] = int(y == target_bucket)

        if need_full_support:
            support_full[i, mask] = 1

    return OLHDerived(one_count=one_count, matches_target=matches_target, support_full=support_full)


def simulate_legitimate_reports(protocol: str, d: int, epsilon: float, n_legit: int, seed: int = 123):
    protocol = protocol.upper()
    rng = np.random.default_rng(seed)
    items = rng.integers(0, d, size=n_legit)

    if protocol == "OUE":
        mech = OUE(d=d, epsilon=epsilon)
        reports = [mech.privatize(x) for x in items]
        return mech, reports

    elif protocol == "OLH": 
        mech = OLH(d=d, epsilon=epsilon, rng=rng)
        reports = [mech.privatize(x) for x in items]
        return mech, reports

    else:
        raise ValueError("protocol must be 'OUE' or 'OLH'")


def run_apa_attack(mech, protocol: str, target_item: int, m_fake: int, legitimate_reports):
 
    apa = AdaptivePatternAttack(ldp_mechanism=mech,
                                target_item=target_item,
                                m_fake_users=m_fake,
                                protocol_type=protocol)
    attacked = apa.execute_attack(legitimate_reports)
    return attacked, apa


def evaluate_indices(n_legit: int, m_fake: int, suspects: np.ndarray) -> Dict[str, float]:
    
    n = n_legit + m_fake
    if len(suspects) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0, "fnr": 1.0}

    suspects = np.asarray(suspects, dtype=int)
    fake_range = np.arange(n_legit, n_legit + m_fake)
    fake_set = set(fake_range.tolist())
    sus_set = set(suspects.tolist())

    tp = len(fake_set & sus_set)
    fp = len(sus_set - fake_set)
    fn = m_fake - tp
    tn = n_legit - fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / max(n_legit, 1)
    fnr = fn / max(m_fake, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "fnr": fnr}


def simulate_apa_and_detect(
    protocol: str = "OUE",
    d: int = 32,
    epsilon: float = 1.0,
    n_legit: int = 2000,
    m_fake: int = 200,
    target_item: int = 7,
    seed: int = 123,
    show_plots: bool = True
) -> Dict[str, Union[float, np.ndarray]]:
   
    mech, legit = simulate_legitimate_reports(protocol, d, epsilon, n_legit, seed=seed)
    attacked, apa = run_apa_attack(mech, protocol, target_item, m_fake, legit)

    ds = DiffStats(protocol=protocol, d=d, epsilon=epsilon, target_item=target_item)
    ds.prepare(mech, attacked)
    out = ds.detect()
    suspects = out["suspects"]

    metrics = evaluate_indices(n_legit, m_fake, suspects)

    print(f"\n[{protocol}]  d={d}, ε={epsilon}, n_legit={n_legit}, m_fake={m_fake}, target={target_item}")
    print(f"Found suspects: {len(suspects)}")
    print(f"Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}  F1={metrics['f1']:.3f} "
          f"FPR={metrics['fpr']:.3f}  FNR={metrics['fnr']:.3f}")

    if show_plots:
        n_total = n_legit + m_fake

        if protocol == "OUE":
            support = reports_to_support_matrix_oue(mech, attacked)
            ones = support.sum(axis=1)
            mask_keep = np.ones(n_total, dtype=bool)
            mask_keep[suspects] = False

            plt.figure(figsize=(10, 4))
            # Before
            plt.subplot(1, 2, 1)
            plt.hist(ones, bins=30)
            plt.title("OUE: #ones per user (attacked)")
            plt.xlabel("#ones")
            plt.ylabel("Count")
            # After
            plt.subplot(1, 2, 2)
            plt.hist(ones[mask_keep], bins=30)
            plt.title("OUE: #ones per user (cleaned)")
            plt.xlabel("#ones")
            plt.tight_layout()
            plt.show()

        elif protocol == "OLH":
            derived = derive_features_olh(mech, attacked, target_item, need_full_support=False)
            matches = derived.matches_target
            mask_keep = np.ones(n_total, dtype=bool)
            mask_keep[suspects] = False

            def frac(x): return x.mean() if len(x) else 0.0

            print(f"Target-match rate before: {frac(matches):.3f}")
            print(f"Target-match rate after : {frac(matches[mask_keep]):.3f}")

            plt.figure(figsize=(6, 4))
            bars = [frac(matches), frac(matches[mask_keep])]
            plt.bar([0, 1], bars)
            plt.xticks([0, 1], ["Before", "After"])
            plt.title("OLH: P(y == h(target))")
            plt.ylabel("Rate")
            plt.show()

    return {
        "suspects": suspects,
        **metrics
    }

if __name__ == "__main__":
    print("=== Adaptive Pattern Attack Detection Demonstration ===")
    print("=====================OUE=====================")
    simulate_apa_and_detect(protocol="OUE", d=32, epsilon=1.0,
                            n_legit=2000, m_fake=200, target_item=7, seed=42)
    print("=====================OLH=====================")
    simulate_apa_and_detect(protocol="OLH", d=32, epsilon=1.0,
                            n_legit=2000, m_fake=200, target_item=7, seed=42)