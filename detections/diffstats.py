import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Union
from collections import Counter, defaultdict
import itertools
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class DetectionResult:
    """Results from fake user detection"""
    fake_users: Set[int]
    support_sets: Dict[frozenset, Set[int]]
    error_frequencies: Dict[int, float]
    total_iterations: int
    convergence_achieved: bool
    E_min: float = float("inf") # best objective value

class DiffStats:
    """
    Implementation of the DiffStats algorithm for detecting fake users in LDP mechanisms.
    
    The algorithm works by:
    1. Finding differentiating subsets of items that have different expected frequencies
       under honest vs adversarial reporting
    2. Iteratively identifying users whose reports are inconsistent with honest behavior
    3. Using statistical tests to determine if users are likely fake based on their
       reporting patterns on differentiating subsets
    """
    def __init__(
        self,
        mechanism_type: str,
        mechanism_params: Dict,
        domain_size: int,
        *,
        top_L: int = 5,
        max_combo_size: int = 3,
        max_outer_iterations: int = 8,
        assume_uniform_prior: bool = True,
        ):
            self.mech = mechanism_type.lower()
            if self.mech not in {"grr", "olh", "oue"}:
                raise ValueError("mechanism_type must be one of {'grr','olh','oue'}")


            self.params = dict(mechanism_params or {})
            self.D = int(domain_size)
            if self.D <= 1:
                raise ValueError("domain_size must be >= 2")

            self.epsilon = float(self.params.get("epsilon", 1.0))
            # Mechanism-specific knobs
            if self.mech == "grr":
        
                self.k_sym = self.params.get("k", self.D)
                e = np.exp(self.epsilon)
                self.p = e / (e + self.k_sym - 1)
                self.q = 1.0 / (e + self.k_sym - 1)
            elif self.mech == "olh":
                self.g = int(self.params.get("g", int(np.floor(np.exp(self.epsilon))) + 1))
                self.prime = int(self.params.get("prime", 2147483647))
                e = np.exp(self.epsilon)
                self.p = e / (e + self.g - 1)
            elif self.mech == "oue":
                self.p = 0.5
                self.q = 1.0 / (1.0 + np.exp(self.epsilon))
            
            
            
            self.top_L = int(max(1, min(top_L, self.D)))
            self.max_combo_size = int(max(1, min(max_combo_size, self.top_L)))
            self.max_outer_iterations = int(max_outer_iterations)


            self.uniform_prior = bool(assume_uniform_prior)
            self.pi = 1.0 / self.D if self.uniform_prior else self.params.get("pi", 1.0 / self.D)


            print(
            f"Diffstats initialized: mech={self.mech.upper()}, D={self.D}, ε={self.epsilon}, "
            f"top_L={self.top_L}, max_combo_size={self.max_combo_size}"
            )

    def _support_set_from_report(self, report: Union[int, Dict, np.ndarray]) -> Set[int]:
        if self.mech == "grr":
            if isinstance(report, (int, np.integer)):
                r = int(report)
                if r < 0 or r >= self.D:
                    raise ValueError("GRR report out of range [0,D)")
                return {r}
            raise ValueError("GRR expects an integer symbol per user")

        if self.mech == "olh":
            if not (isinstance(report, dict) and all(k in report for k in ("a", "b", "y"))):
                raise ValueError("OLH expects a dict with keys {'a','b','y'} per user")
            a, b, y = int(report["a"]), int(report["b"]), int(report["y"])
            S = set()
            for k in range(self.D):
                if ((a * k + b) % self.prime) % self.g == y:
                    S.add(k)
            return S

        # OUE
        if isinstance(report, np.ndarray) and report.ndim == 1 and report.size == self.D:
            # treat non-binary as binary by thresholding to {0,1}
            return {i for i, v in enumerate(report.tolist()) if int(v) == 1}
        raise ValueError("OUE expects a binary vector of length D per user")
    
    def _expected_prob_bit_one(self) -> float:
        """
        Compute P(bit k == 1) under *honest* reporting
        true items (π = 1/D). This probability is identical for all k by symmetry.
        """
        pi = self.pi
        if self.mech == "grr":
            # P = π * p + (1-π) * q
            e = np.exp(self.epsilon)
            p = e / (e + self.D - 1)
            q = 1.0 / (e + self.D - 1)
            return pi * p + (1 - pi) * q
        if self.mech == "olh":
            return pi * self.p + (1 - pi) * (1.0 / self.g)
        p = 0.5
        q = 1.0 / (1.0 + np.exp(self.epsilon))
        return pi * p + (1 - pi) * q
  
    def _build_support_maps(self, users_reports: List[List[Union[int, Dict, np.ndarray]]]):
        user_supports: List[Set[int]] = []
        item_to_users: List[Set[int]] = [set() for _ in range(self.D)]

        for uid, reports in enumerate(users_reports):
            if len(reports) != 1:
                raise ValueError("Each user must provide exactly one report for Algorithm 1.")
            S = self._support_set_from_report(reports[0])
            user_supports.append(S)
            for k in S:
                item_to_users[k].add(uid)
        return user_supports, item_to_users

    def _Ok_counts(self, item_to_users: List[Set[int]]) -> np.ndarray:
        """Observed counts O^k = number of users with bit k == 1."""
        return np.array([len(u) for u in item_to_users], dtype=float)

    def _yk_expected(self, n_users: int) -> np.ndarray:
        p1 = self._expected_prob_bit_one()
        return np.full(self.D, n_users * p1, dtype=float)

    def _candidate_pool(self, K: Set[int], item_to_users: List[Set[int]]) -> Set[int]:
        pool: Set[int] = set()
        for k in K:
            pool |= item_to_users[k]
        return pool

    def _topL_items_in_pool(self, pool: Set[int], item_to_users: List[Set[int]], L: int) -> List[int]:
        if not pool:
            return []
        # score items by how many users in pool support them
        scores = [(k, len(item_to_users[k] & pool)) for k in range(self.D)]
        scores.sort(key=lambda kv: kv[1], reverse=True)
        # keep items with positive support
        filtered = [k for k, c in scores if c > 0]
        return filtered[:L]

    def _users_supporting_combo(self, combo: Tuple[int, ...], pool: Set[int], item_to_users: List[Set[int]]) -> Set[int]:
        if not combo:
            return set()
        users = pool.copy()
        for k in combo:
            users &= item_to_users[k]
            if not users:
                break
        return users

    def _remove_users_and_score(self, Ok: np.ndarray, yk: np.ndarray, users_to_remove: Set[int], user_supports: List[Set[int]]) -> float:
        if not users_to_remove:
            return float(np.sum((Ok - yk) ** 2))
        Ok_prime = Ok.copy()
        for u in users_to_remove:
            for k in user_supports[u]:
                Ok_prime[k] -= 1
        return float(np.sum((Ok_prime - yk) ** 2))


    def detect_fake_users(self, users_reports: List[List[Union[int, Dict, np.ndarray]]]) -> DetectionResult:
        """
        Run Algorithm 1 across the chosen mechanism. Expects exactly one report per user.
        Returns the subset U_f that minimizes the squared frequency error if removed.
        """
        print("\n=== Starting Diffstats (Algorithm 1) ===")

        user_supports, item_to_users = self._build_support_maps(users_reports)
        n = len(user_supports)
        Ok = self._Ok_counts(item_to_users)            # length D
        yk = self._yk_expected(n)                      # length D (constant by symmetry)
        K: Set[int] = {k for k in range(self.D) if Ok[k] > 0}

        E_min = float("inf")
        Uf: Set[int] = set()
        support_sets: Dict[frozenset, Set[int]] = {}
        iteration = 0

        while K and iteration < self.max_outer_iterations:
            iteration += 1
            # E_sq(k) per item; choose δ̂ = argmin over current K
            E_sq = (Ok - yk) ** 2
            k_hat = min(K, key=lambda kk: E_sq[kk])
            print(f"Iteration {iteration}: δ̂={k_hat}, E_sq(δ̂)={E_sq[k_hat]:.3f}")

            # remove δ̂ from K
            K.remove(k_hat)

            # U_s: users that support any k in remaining K
            Us = self._candidate_pool(K, item_to_users)
            print(f"  |U_s|={len(Us)}")
            if not Us:
                break

            # Top-L items inside the pool
            SL = self._topL_items_in_pool(Us, item_to_users, self.top_L)
            if not SL:
                print("  No supported items in candidate pool; stopping early.")
                break
            print(f"  Top-L items: {SL}")

            # Enumerate combos up to size max_combo_size
            combos: List[Tuple[int, ...]] = []
            for r in range(1, min(self.max_combo_size, len(SL)) + 1):
                combos.extend(list(itertools.combinations(SL, r)))

            best_score = float("inf")
            best_combo: Optional[Tuple[int, ...]] = None
            best_subset: Set[int] = set()

            for combo in combos:
                U_sc = self._users_supporting_combo(combo, Us, item_to_users)
                if not U_sc:
                    continue
                score = self._remove_users_and_score(Ok, yk, U_sc, user_supports)
                if score < best_score:
                    best_score = score
                    best_combo = combo
                    best_subset = U_sc

            if best_score < E_min and best_subset:
                E_min = best_score
                Uf = best_subset
                support_sets[frozenset(best_combo)] = set(best_subset)
                print(f"  Updated best: combo={best_combo}, |U_f|={len(Uf)}, E_min={E_min:.3f}")
            else:
                print("  No improving subset found in this iteration.")

        # Simple per-user proxy: fraction of supported items
        freq_proxy: Dict[int, float] = {}
        for u in Uf:
            denom = max(1, len(user_supports[u]))
            freq_proxy[u] = float(len(user_supports[u])) / float(self.D)

        converged = (iteration < self.max_outer_iterations)
        print("\n=== Diffstats complete ===")
        print(f"Iterations: {iteration}, |U_f|={len(Uf)}, converged={converged}")

        return DetectionResult(
            fake_users=Uf,
            support_sets=support_sets,
            error_frequencies=freq_proxy,
            total_iterations=iteration,
            convergence_achieved=converged,
            E_min=E_min,
        )


    @staticmethod
    def evaluate_detection_accuracy(detection_result: DetectionResult, true_fake_users: Set[int]) -> Dict[str, float]:
        detected_fake = detection_result.fake_users
        tp = len(detected_fake & true_fake_users)
        fp = len(detected_fake - true_fake_users)
        fn = len(true_fake_users - detected_fake)
        # We don't know TN exactly without n_total; approximate as 0 if not provided
        tn = 0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        acc = 0.0  # unknown without TN
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": acc,
        }
