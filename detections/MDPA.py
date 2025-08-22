
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, chisquare, binomtest


def reports_to_support_matrix_oue(oue, reports: List[np.ndarray]) -> np.ndarray:
   
    support = np.asarray(reports, dtype=int)
    if support.ndim != 2 or support.shape[1] != oue.d:
        raise ValueError(f"OUE support shape mismatch: expected (*,{oue.d}), got {support.shape}")
    return (support > 0).astype(int)
def reports_to_support_matrix_oue(oue, reports: List[np.ndarray]) -> np.ndarray:
   
    support = np.asarray(reports, dtype=int)
    if support.ndim != 2 or support.shape[1] != oue.d:
        raise ValueError(f"OUE support shape mismatch: expected (*,{oue.d}), got {support.shape}")
    return (support > 0).astype(int)


def olh_hash_all_v(olh, a: int, b: int) -> np.ndarray:
    v = np.arange(olh.d, dtype=np.int64)
    return ((a * v + b) % olh.prime) % olh.g

def _oue_exact_pmf_K(d: int, p: float, q: float) -> np.ndarray:
    s = np.arange(d)                   # 0..(d-1)
    pmf_S = binom.pmf(s, d-1, q)      # S ~ Binom(d-1, q)
    pmf = np.zeros(d + 1)
    pmf[0]   = p * pmf_S[0]                               # K=0
    pmf[1:d] = (1 - p) * pmf_S[0:d-1] + p * pmf_S[1:d]    # 1..d-1
    pmf[d]   = (1 - p) * pmf_S[d - 1]                     # K=d
    return pmf


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



class DiffStats:
    """
    DiffStats detection mechanism for identifying fake users in OUE and OLH.
    """

    def __init__(self, protocol: str, d: int, epsilon: float, target_item: int):
        self.protocol = protocol.upper()
        self.d = d
        self.epsilon = epsilon
        self.target_item = target_item

        # OUE features
        self.support_oue: Optional[np.ndarray] = None
        self.ones_oue: Optional[np.ndarray] = None

        # OLH features
        self.olh_one_count: Optional[np.ndarray] = None
        self.olh_matches_target: Optional[np.ndarray] = None

        # Params
        self.params: Dict[str, float] = {}

        # For OLH calibration
        self._mech = None
        self._reports = None

    def _set_params_oue(self, oue) -> None:
        p, q = oue.p, oue.q
        p_bit = ((1.0 - p) + (self.d - 1.0) * q) / self.d

        Var_exact = p * (1 - p) + (self.d - 1) * q * (1 - q)
        self.params.update({
            "p": p,
            "q": q,
            "p_bit": p_bit,
            "E_ones": self.d * p_bit,
            "Var_ones": Var_exact
        })

    def _set_params_olh(self, olh) -> None:
        # prob a non-target report matches the target’s bucket
        r_bg = olh.q + (olh.p - olh.q) / olh.g
        self.params.update({
            "p": olh.p,
            "q": olh.q,
            "g": olh.g,
            "r_bg": r_bg
        })

    def prepare(self, ldp_mechanism, reports: List[Union[np.ndarray, Dict[str, int]]]) -> None:
        self._mech = ldp_mechanism
        self._reports = reports

        if self.protocol == "OUE":
            self._set_params_oue(ldp_mechanism)
            self.support_oue = reports_to_support_matrix_oue(ldp_mechanism, reports)
            self.ones_oue = self.support_oue.sum(axis=1)

        elif self.protocol == "OLH":
            self._set_params_olh(ldp_mechanism)
            derived = derive_features_olh(ldp_mechanism, reports, self.target_item, need_full_support=False)
            self.olh_one_count = derived.one_count
            self.olh_matches_target = derived.matches_target

        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")

    def detect(self) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        if self.protocol == "OUE":
            return self._detect_oue()
        elif self.protocol == "OLH":
            return self._detect_olh()
        else:
            raise ValueError("prepare() not called or protocol not supported")

    # -------- OUE detection: per-K target residual + stratified χ² --------
    def _detect_oue(self) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        ones = self.ones_oue              # (n,)
        support = self.support_oue        # (n,d)
        n = len(ones)
        d = self.d
        p = float(self.params["p"])
        q = float(self.params["q"])

        # Exact clean pmf for K = #ones
        pmf_K = _oue_exact_pmf_K(d, p, q)
        k_values, counts = np.unique(ones, return_counts=True)
        expected_counts_full = np.maximum(n * pmf_K[k_values], 1e-12)

        # Per-user conditional residual wrt clean P(target=1 | K) = K/d
        target_col = support[:, self.target_item].astype(int)
        k_over_d = ones / d
        var_k = k_over_d * (1.0 - k_over_d) + 1e-9
        z_target = (target_col - k_over_d) / np.sqrt(var_k)

        # Rank: largest positive residuals first (APA bias)
        order = np.argsort(-z_target)

        # Track per-K-bin target counts to do stratified χ²
        pos = {int(k): i for i, k in enumerate(k_values)}
        target_counts = np.zeros_like(counts, dtype=float)
        for i in range(n):
            kk = int(ones[i])
            if kk in pos:
                target_counts[pos[kk]] += target_col[i]

        # Greedy removal with objective:
        # metric = chi2(K-hist) + chi2(target|K stratified) + small penalty
        best_k = 0
        best_metric = np.inf
        removed = np.zeros(n, dtype=bool)
        curr_counts = counts.astype(float).copy()
        curr_target_counts = target_counts.astype(float).copy()

        for k in range(min(n, 400)):
            idx = order[k]
            removed[idx] = True

            # decrement K-bin and its target count
            kk = int(ones[idx])
            if kk in pos:
                j = pos[kk]
                curr_counts[j] -= 1.0
                curr_target_counts[j] -= target_col[idx]

            n_kept = n - (k + 1)
            if n_kept <= max(20, d):
                break

            # χ² on K histogram
            expected_K_kept = expected_counts_full * (n_kept / n)
            chi_ones = np.sum((curr_counts - expected_K_kept) ** 2 / np.maximum(expected_K_kept, 1e-9))
            df_ones = max(len(k_values) - 1, 1)
            chi_ones /= df_ones

            # Stratified χ² for target|K
            k_over_d_bins = k_values / d
            exp_targets = curr_counts * k_over_d_bins
            var_targets = np.maximum(curr_counts * k_over_d_bins * (1.0 - k_over_d_bins), 1e-9)
            chi_tk = np.sum((curr_target_counts - exp_targets) ** 2 / var_targets)

            # Small penalty to discourage over-removal
            metric = chi_ones + chi_tk + 0.05 * ((k + 1) / n)

            if metric < best_metric:
                best_metric = metric
                best_k = k + 1

        suspects = order[:best_k]
        return {
            "suspects": np.sort(suspects),
            "scores": z_target  # for debugging/inspection
        }

    # -------- OLH detection: data-calibrated baseline + binomial guard --------
    def _estimate_r0_from_data_olh(self) -> float:
        """
        Estimate clean baseline target-match rate r0 = f_t*p + (1-f_t)*r_bg
        using OLH aggregation to get f_t (fraction of target item).
        """
        mech = self._mech
        reports = self._reports
        n = len(reports)
        if n == 0:
            return self.params["r_bg"]

        est_counts = mech.estimate_counts(reports)  # unbiased OLH estimate
        f_t_hat = float(np.clip(est_counts[self.target_item], 0.0, n)) / max(n, 1)

        r_bg = self.params["r_bg"]
        r0_hat = f_t_hat * mech.p + (1.0 - f_t_hat) * r_bg
        return float(np.clip(r0_hat, 1e-9, 1 - 1e-9))

    def _detect_olh(self) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        matches = self.olh_matches_target.astype(int)
        n_total = len(matches)

        # Calibrate baseline from data (fallback to r_bg if needed)
        try:
            r0 = self._estimate_r0_from_data_olh()
        except Exception:
            r0 = self.params["r_bg"]

        # Candidate pool: reports that matched the target bucket
        pool = np.where(matches == 1)[0]

        # Tie-breaker: prefer "typical" one_count (APA tends to look typical)
        if self.olh_one_count is not None and len(pool) > 0:
            med = np.median(self.olh_one_count)
            pool = pool[np.argsort(np.abs(self.olh_one_count[pool] - med))]

        kept = np.ones(n_total, dtype=bool)
        k_success = int(matches.sum())
        n_kept = int(n_total)
        alpha = 0.05

        # Remove as few as needed so kept target-match rate is not > r0
        removed_ct = 0
        for idx in pool:
            pval = binomtest(k_success, n_kept, r0, alternative='greater').pvalue
            if pval >= alpha:
                break  # baseline satisfied

            kept[idx] = False
            removed_ct += 1
            k_success -= 1
            n_kept -= 1

            if n_kept < 50:
                break

        # Final p-value after loop
        pval_final = binomtest(k_success, n_kept, r0, alternative='greater').pvalue

        suspects = np.where(~kept)[0]
        return {
            "suspects": np.sort(suspects),
            "scores": matches.astype(float),
            "r0": r0,
            "p_value_final": float(pval_final),
            "removed": int(removed_ct),
            "kept": int(n_kept)
        }