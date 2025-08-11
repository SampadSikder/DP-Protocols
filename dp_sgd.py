import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import math


from basic_dp import GaussianMechanism


def compute_gradient(theta, X, y):
    n = len(y)
    predictions = X.dot(theta)
    errors = y - predictions
    gradient = -X.T.dot(errors) / n
    return gradient

def gaussian_mech_vec(grad, sensitivity, epsilon, delta):
    """
    Apply Gaussian mechanism to a vector-valued function with L2-sensitivity.
    """
    sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
    return grad + np.random.normal(0.0, sigma, size=np.asarray(grad).shape)

def noisy_gradient_descent(X_train, y_train, iterations, epsilon, sensitivity, delta=1e-5):
    theta = np.zeros(X_train.shape[1])
    for _ in range(iterations):
        grad = compute_gradient(theta, X_train, y_train)
        noisy_grad = gaussian_mech_vec(grad, sensitivity, epsilon, delta)
        theta = theta - 0.01 * noisy_grad  # learning rate = 0.01
    return theta


# ---------------------------
# DP-SGD (mini-batch, Gaussian)
# ---------------------------
class DPSGDGaussian:
    """
    DP-SGD with per-example clipping + Gaussian mechanism.
    """

    def __init__(self, epsilon_total, delta=1e-5, learning_rate=0.01,
                 gradient_clip_bound=1.0, batch_size=32):
        self.epsilon_total = float(epsilon_total)
        self.delta = float(delta)
        self.learning_rate = float(learning_rate)
        self.gradient_clip_bound = float(gradient_clip_bound)
        self.batch_size = int(batch_size)

    def clip_gradient(self, v, b):
        norm = np.linalg.norm(v)
        if norm > b:
            return v * (b / norm)
        return v.copy()

    def compute_batch_gradient(self, theta, X_batch, y_batch):
        # Per-example gradients, then clip each, then average
        grads = []
        for i in range(len(X_batch)):
            grad_i = compute_gradient(theta, X_batch[i:i+1], y_batch[i:i+1])
            grads.append(self.clip_gradient(grad_i, self.gradient_clip_bound))
        return np.mean(grads, axis=0)

    def fit(self, X, y, iterations):
        theta = np.zeros(X.shape[1])
        n = len(y)

        epsilon_per_iter = self.epsilon_total / iterations

    
        sensitivity = 2.0 * self.gradient_clip_bound / self.batch_size

        print("Gaussian DP-SGD Configuration:")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Gradient clip bound: {self.gradient_clip_bound}")
        print(f"- Sensitivity: {sensitivity:.6f}")
        print(f"- Epsilon per iteration: {epsilon_per_iter:.6f}")
        print(f"- Delta (total): {self.delta}")

        for it in range(iterations):
            batch_idx = np.random.choice(n, self.batch_size, replace=False)
            Xb, yb = X[batch_idx], y[batch_idx]

            noise_multiplier = 1.2  # try 0.8–2.0
            clipped_grad = self.compute_batch_gradient(theta, Xb, yb)
            sigma = (noise_multiplier * self.gradient_clip_bound) / self.batch_size
            noisy_grad = clipped_grad + np.random.normal(0.0, sigma, size=clipped_grad.shape)
            theta = theta - self.learning_rate * noisy_grad

            if (it + 1) % 20 == 0:
                loss = np.mean((y - X.dot(theta)) ** 2)
                print(f"Iteration {it + 1}/{iterations}, Loss: {loss:.6f}")

        return theta
    

# ---------------------------
# Sample-and-Aggregate (Gaussian)
# ---------------------------
class DPSGDSampleAggregate:
    """
    DP-SGD using sample-and-aggregate with clipping + Gaussian mechanism.
    """

    def __init__(self, epsilon_total, delta=1e-5, num_partitions=10,
                 learning_rate=0.01, gradient_clip_bound=None, auto_clip=True):
        self.epsilon_total = float(epsilon_total)
        self.delta = float(delta)
        self.num_partitions = int(num_partitions)
        self.learning_rate = float(learning_rate)
        self.gradient_clip_bound = None if gradient_clip_bound is None else float(gradient_clip_bound)
        self.auto_clip = bool(auto_clip)
        self.epsilon_per_iteration = self.epsilon_total  # set in fit()

    def estimate_gradient_bounds(self, X, y, theta_init, num_samples=100):
        norms = []
        n = len(y)
        batch_size = min(50, max(1, n // 10))
        for _ in range(num_samples):
            idx = np.random.choice(n, batch_size, replace=False)
            grad = compute_gradient(theta_init, X[idx], y[idx])
            norms.append(np.linalg.norm(grad))
        return float(np.percentile(norms, 95)) # 95th percentile of gradient norms is the clipping bound

    def clip_gradient(self, v, b):
        norm = np.linalg.norm(v)
        if norm > b:
            return v * (b / norm)
        return v.copy()

    def compute_sensitivity_sample_aggregate(self, clip_bound, partition_size):
        return 2.0 * clip_bound / max(1, partition_size) 

    def partition_data(self, X, y):
        n = len(y)
        idx = np.random.permutation(n)
        base = n // self.num_partitions

        parts_X, parts_y, sizes = [], [], []
        for i in range(self.num_partitions):
            start = i * base
            end = n if i == self.num_partitions - 1 else (i + 1) * base
            sel = idx[start:end]
            parts_X.append(X[sel])
            parts_y.append(y[sel])
            sizes.append(len(sel))
        return parts_X, parts_y, sizes

    def compute_partition_gradient(self, theta, X_part, y_part, clip_bound):
        grad = compute_gradient(theta, X_part, y_part)
        return self.clip_gradient(grad, clip_bound)

    def aggregate_gradients_with_noise(self, gradients, sensitivity, epsilon_per_iter, delta):
        avg_grad = np.mean(gradients, axis=0)
        mech = GaussianMechanism(epsilon=epsilon_per_iter, delta=delta, sensitivity=sensitivity)
        return mech.add_noise(avg_grad)

    def fit(self, X, y, iterations, sensitivity=None):
        theta = np.zeros(X.shape[1])

        # Determine clipping bound
        if self.auto_clip and self.gradient_clip_bound is None:
            self.gradient_clip_bound = self.estimate_gradient_bounds(X, y, theta)
            print(f"Auto-computed gradient clipping bound: {self.gradient_clip_bound:.4f}")
        elif self.gradient_clip_bound is None:
            self.gradient_clip_bound = 1.0
            print(f"Using default gradient clipping bound: {self.gradient_clip_bound}")
        else:
            print(f"Using manual gradient clipping bound: {self.gradient_clip_bound}")

        parts_X, parts_y, sizes = self.partition_data(X, y)
        typical_partition_size = int(np.median(sizes)) if sizes else 1

        if sensitivity is None:
            sensitivity = self.compute_sensitivity_sample_aggregate(
                self.gradient_clip_bound, typical_partition_size
            )
            print(f"Computed sensitivity: {sensitivity:.6f}")
        else:
            print(f"Using provided sensitivity: {sensitivity}")

        # Split ε across iterations (simple accountant)
        epsilon_per_iter = self.epsilon_total / iterations
        self.epsilon_per_iteration = epsilon_per_iter

        info = {
            "clip_bound": self.gradient_clip_bound,
            "sensitivity": sensitivity,
            "epsilon_per_iter": epsilon_per_iter,
            "partition_size": typical_partition_size,
            "delta": self.delta,
        }

        for it in range(iterations):
            parts_X, parts_y, _ = self.partition_data(X, y)

            grads = []
            for Xp, yp in zip(parts_X, parts_y):
                if len(yp) == 0:
                    continue
                grads.append(self.compute_partition_gradient(
                    theta, Xp, yp, self.gradient_clip_bound
                ))

            if not grads:
                raise RuntimeError("No non-empty partitions; increase dataset size or reduce num_partitions.")

            noisy_avg_grad = self.aggregate_gradients_with_noise(
                grads, sensitivity, epsilon_per_iter, self.delta
            )
            theta = theta - self.learning_rate * noisy_avg_grad

            if (it + 1) % 20 == 0:
                loss = np.mean((y - X.dot(theta)) ** 2)
                print(f"Iteration {it + 1}/{iterations}, Loss: {loss:.6f}")

        return theta, info
    

def generate_synthetic_data(n_samples=1000, n_features=5, noise_std=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    X = np.column_stack([np.ones(n_samples), X])  # bias term
    true_theta = rng.standard_normal(n_features + 1)
    y = X.dot(true_theta) + rng.normal(0.0, noise_std, n_samples)
    return X, y, true_theta


def evaluate_model(theta, X_test, y_test):
    preds = X_test.dot(theta)
    return float(np.mean((y_test - preds) ** 2))


if __name__ == "__main__":
    # Data
    X, y, true_theta = generate_synthetic_data(n_samples=1000, n_features=3)
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("=" * 60)
    print("GAUSSIAN-ONLY DP TRAINING")
    print("=" * 60)
    print("True parameters:", true_theta)

    # Non-private baseline
    theta_np = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    mse_np = evaluate_model(theta_np, X_test, y_test)
    print(f"\nNon-private LSQ - MSE: {mse_np:.6f}")

    # 1) DP-SGD (mini-batch, Gaussian)
    print("\n" + "-" * 50)
    print("DP-SGD (Gaussian)")
    print("-" * 50)
    dp_sgd = DPSGDGaussian(
        epsilon_total=1.0,
        delta=1e-5,
        learning_rate=0.01,
        gradient_clip_bound=1.0,
        batch_size=32
    )
    theta_dp = dp_sgd.fit(X_train, y_train, iterations=100)
    mse_dp = evaluate_model(theta_dp, X_test, y_test)
    print(f"Gaussian DP-SGD - MSE: {mse_dp:.6f}")

    # 2) Sample-and-aggregate (Gaussian)
    print("\n" + "-" * 50)
    print("Sample-and-Aggregate (Gaussian)")
    print("-" * 50)
    dp_sa = DPSGDSampleAggregate(
        epsilon_total=1.0,
        delta=1e-5,
        num_partitions=10,
        learning_rate=0.01,
        auto_clip=True
    )
    theta_sa, info = dp_sa.fit(X_train, y_train, iterations=100)
    mse_sa = evaluate_model(theta_sa, X_test, y_test)
    print(f"Gaussian Sample-Aggregate - MSE: {mse_sa:.6f}")
    print("Training info:", info)

    # 3) Simple noisy GD (full-batch, Gaussian)
    print("\n" + "-" * 50)
    print("Simple Noisy GD (Gaussian)")
    print("-" * 50)
    theta_gd = noisy_gradient_descent(X_train, y_train, iterations=100,
                                      epsilon=1.0, sensitivity=1.0, delta=1e-5)
    mse_gd = evaluate_model(theta_gd, X_test, y_test)
    print(f"Simple Gaussian GD - MSE: {mse_gd:.6f}")

    print("\n" + "=" * 60)
    print("Done")