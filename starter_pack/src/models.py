"""Model implementations: Softmax Regression and One-Hidden-Layer Neural Network.

Both models expose the same interface:
    forward(X)  -> (scores, cache)
    backward(cache, Y_onehot, reg) -> grads dict
    compute_loss(S, Y_onehot, reg) -> scalar
    get_params() / set_params(p) for checkpointing
"""
import copy
import numpy as np
from utils import stable_softmax, cross_entropy_loss, l2_weight_penalty


class SoftmaxRegression:
    """Multiclass softmax regression: s(x) = Wx + b."""

    def __init__(self, input_dim, num_classes, seed=42):
        rng = np.random.default_rng(seed)
        self.params = {
            "W": rng.standard_normal((num_classes, input_dim)) * 0.01,
            "b": np.zeros(num_classes),
        }
        self.weight_keys = ["W"]

    def forward(self, X):
        """Compute logits S = X W^T + b.

        Returns (S, cache) where cache stores quantities needed for backward.
        """
        S = X @ self.params["W"].T + self.params["b"]
        cache = {"X": X, "S": S}
        return S, cache

    def backward(self, cache, Y_onehot, reg=1e-4):
        """Gradient of (CE + L2) w.r.t. W and b."""
        n = cache["X"].shape[0]
        P = stable_softmax(cache["S"])
        dS = (P - Y_onehot) / n

        grads = {
            "W": dS.T @ cache["X"] + reg * self.params["W"],
            "b": np.sum(dS, axis=0),
        }
        return grads

    def compute_loss(self, S, Y_onehot, reg=1e-4):
        return cross_entropy_loss(S, Y_onehot) + l2_weight_penalty(
            self.params, reg, self.weight_keys
        )

    def get_params(self):
        return copy.deepcopy(self.params)

    def set_params(self, params):
        self.params = copy.deepcopy(params)


class NeuralNetwork:
    """One-hidden-layer network: h = tanh(W1 x + b1), s = W2 h + b2."""

    def __init__(self, input_dim, hidden_dim, num_classes, seed=42):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.standard_normal((hidden_dim, input_dim))
            * np.sqrt(1.0 / input_dim),
            "b1": np.zeros(hidden_dim),
            "W2": rng.standard_normal((num_classes, hidden_dim))
            * np.sqrt(1.0 / hidden_dim),
            "b2": np.zeros(num_classes),
        }
        self.weight_keys = ["W1", "W2"]

    def forward(self, X):
        """Forward pass through the full network.

        Z1 = X W1^T + b1   (pre-activation)
        H  = tanh(Z1)       (hidden representation)
        S  = H W2^T + b2    (output logits)
        """
        Z1 = X @ self.params["W1"].T + self.params["b1"]
        H = np.tanh(Z1)
        S = H @ self.params["W2"].T + self.params["b2"]
        cache = {"X": X, "Z1": Z1, "H": H, "S": S}
        return S, cache

    def backward(self, cache, Y_onehot, reg=1e-4):
        """Backpropagation: chain-rule gradients for all four parameter arrays.

        dL/dS  = (1/n)(P - Y)
        dL/dW2 = dS^T H  + reg W2
        dL/db2 = 1^T dS
        dL/dH  = dS W2
        dL/dZ1 = dH * (1 - H^2)          [tanh derivative]
        dL/dW1 = dZ1^T X + reg W1
        dL/db1 = 1^T dZ1
        """
        n = cache["X"].shape[0]
        P = stable_softmax(cache["S"])
        dS = (P - Y_onehot) / n

        grads = {}
        grads["W2"] = dS.T @ cache["H"] + reg * self.params["W2"]
        grads["b2"] = np.sum(dS, axis=0)

        dH = dS @ self.params["W2"]
        dZ1 = dH * (1.0 - cache["H"] ** 2)

        grads["W1"] = dZ1.T @ cache["X"] + reg * self.params["W1"]
        grads["b1"] = np.sum(dZ1, axis=0)

        return grads

    def compute_loss(self, S, Y_onehot, reg=1e-4):
        return cross_entropy_loss(S, Y_onehot) + l2_weight_penalty(
            self.params, reg, self.weight_keys
        )

    def get_params(self):
        return copy.deepcopy(self.params)

    def set_params(self, params):
        self.params = copy.deepcopy(params)
