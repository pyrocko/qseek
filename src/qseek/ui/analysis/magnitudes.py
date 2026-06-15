from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
from scipy.stats import norm

LOG_10 = np.log(10)


def log_likelihood_func(magnitude: float, beta: float, mu: float, sigma: float):
    log_gr = np.log(beta) - beta * (magnitude - mu) - 0.5 * beta**2 * sigma**2
    log_qm = norm.logcdf((magnitude - mu) / sigma)
    return log_gr + log_qm


def neg_log_likelihood_func(args: tuple[float, float, float], magnitudes: np.ndarray):
    b, mu, sigma = np.square(args)
    beta = b * LOG_10
    return -np.sum(log_likelihood_func(magnitudes, beta, mu, sigma))


def calculate_entire_magnitude_fit(magnitudes: np.ndarray):
    x0 = [np.sqrt(1.0), np.min(magnitudes) + 1, np.sqrt(1.0)]
    res = minimize(neg_log_likelihood_func, x0, args=(magnitudes,), method="Powell")
    sqrtb, sqrtmu, sqrtsigma = res.x
    b = np.square(sqrtb)
    mu = np.square(sqrtmu)
    sigma = np.square(sqrtsigma)
    return b, mu, sigma


def prob_ogata_katsura(mbinvalues: float, b: float, mu: float, sigma: float) -> float:
    dum = (mbinvalues - mu) / (np.sqrt(2.0) * sigma)
    return 0.5 * (1.0 + erf(dum))


def ogata_katsura(mbinvalues: float, b: float, mu: float, sigma: float) -> float:
    beta = LOG_10 * b
    dum = (mbinvalues - mu) / (np.sqrt(2.0) * sigma)
    qm = 0.5 * (1.0 + erf(dum))
    gr = beta * np.exp(
        -beta * (mbinvalues - mu) - np.square(beta) * np.square(sigma) / 2.0
    )
    return gr * qm


def calculate_dmag_bpositive(times: np.ndarray, magnitudes: np.ndarray, d_mc: float):
    idx = np.argsort(times)
    times_sorted = times[idx]
    magnitudes_sorted = magnitudes[idx]
    times_diff = times_sorted[:-1]
    mag_diff = magnitudes_sorted[1:] - magnitudes_sorted[:-1]
    idx_pos = mag_diff >= d_mc
    return times_diff[idx_pos], mag_diff[idx_pos] - d_mc
