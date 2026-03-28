"""Optimizer implementations: SGD, Momentum, and Adam.

Each optimizer exposes a single method:
    step(params, grads) — updates params in-place.
"""
import numpy as np


class SGDOptimizer:
    """Vanilla stochastic gradient descent: θ ← θ − η ∇L."""

    def __init__(self, lr=0.05):
        self.lr = lr

    def step(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]


class MomentumOptimizer:
    """SGD with classical momentum.

    v  ← μ v − η ∇L
    θ  ← θ + v
    """

    def __init__(self, params, lr=0.05, mu=0.9):
        self.lr = lr
        self.mu = mu
        self.velocity = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        for key in params:
            self.velocity[key] = self.mu * self.velocity[key] - self.lr * grads[key]
            params[key] += self.velocity[key]


class AdamOptimizer:
    """Adam optimizer (Kingma & Ba, 2015).

    m  ← β1 m + (1−β1) g
    v  ← β2 v + (1−β2) g²
    m̂  = m / (1−β1^t)
    v̂  = v / (1−β2^t)
    θ  ← θ − η m̂ / (√v̂ + ε)
    """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
