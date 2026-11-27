"""Парные тесты значимости по per-query метрикам."""
import numpy as np


def _cluster_totals(d, groups):
    if groups is None:
        return d, np.ones(len(d), dtype=int)
    if len(groups) != len(d):
        raise ValueError("Для каждой пары нужен идентификатор диалога")
    _, indices, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return np.bincount(indices, weights=d), counts


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0, *, groups=None):
    """Парный bootstrap среднего по ходам с выборкой целых диалогов."""
    d = np.asarray(b, float) - np.asarray(a, float)
    totals, counts = _cluster_totals(d, groups)
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(totals), size=(n, len(totals)))
    means = totals[sample].sum(axis=1) / counts[sample].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def randomization_test(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0, *, groups=None) -> float:
    """Двусторонний парный тест с одним случайным знаком на диалог."""
    d = np.asarray(b, float) - np.asarray(a, float)
    totals, _ = _cluster_totals(d, groups)
    if not totals.any():
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(totals)))
    null = np.abs((signs * totals).sum(axis=1) / len(d))
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
