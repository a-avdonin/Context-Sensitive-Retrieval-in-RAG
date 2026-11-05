"""Парные тесты значимости по per-query метрикам."""
import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Δ = mean(b) - mean(a) и 95% доверительный интервал bootstrap по запросам."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def randomization_test(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> float:
    """Двусторонний p-value парного рандомизационного теста (случайная перестановка a/b внутри пары)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if not d.any():
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


def holm(pvalues: list[float]) -> list[float]:
    """Поправка Холма–Бонферрони на множественные сравнения."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj
