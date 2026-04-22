"""
Data generation for damped oscillator Bayesian inference project.

Physical model: x(t) = A * exp(-gamma * t) * cos(omega * t + phi)

Parameters:
    A     - initial amplitude
    gamma - damping coefficient (1/s)
    omega - angular frequency (rad/s)
    phi   - initial phase (rad)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# ── True parameters ────────────────────────────────────────────────────────────

TRUE_PARAMS: Dict[str, float] = {
    "A": 2.0,
    "gamma": 0.3,
    "omega": 2.5,
    "phi": 0.4,
}


# ── Signal ─────────────────────────────────────────────────────────────────────

def damped_oscillator(
    t: np.ndarray,
    A: float,
    gamma: float,
    omega: float,
    phi: float,
) -> np.ndarray:
    """Return the displacement of a damped harmonic oscillator."""
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi)


# ── Noise generators ───────────────────────────────────────────────────────────

def gaussian_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Additive white Gaussian noise.
    Models: thermal noise, quantisation error, uncorrelated sensor noise.
    """
    return rng.normal(0.0, sigma, n)


def pink_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Additive 1/f (pink) noise via power-spectrum shaping in the frequency domain.
    Models: mechanical vibration background, material/environmental fluctuations.
    """
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid division by zero at DC
    power = 1.0 / np.sqrt(freqs)
    power[0] = 0.0  # zero-mean
    shaped = np.fft.irfft(np.fft.rfft(white) * power, n=n)
    # normalise to requested sigma
    shaped *= sigma / shaped.std()
    return shaped


def heteroscedastic_noise(
    signal: np.ndarray,
    sigma_frac: float,
    sigma_floor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Multiplicative noise: std ∝ |signal| + floor.
    Models: sensor gain uncertainty, ADC nonlinearity, amplitude-dependent drag.

    sigma_frac  - fractional amplitude of noise relative to |signal|
    sigma_floor - minimum noise floor (prevents zero noise at nodes)
    """
    local_std = sigma_frac * np.abs(signal) + sigma_floor
    return rng.normal(0.0, local_std)


def ornstein_uhlenbeck_noise(
    n: int,
    sigma: float,
    theta: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck (mean-reverting) noise.
    Models: slowly-drifting ambient temperature/pressure, correlated sensor drift.

    theta - mean-reversion rate; high → decorrelates quickly (→ white noise)
    """
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = (
            noise[i - 1] * np.exp(-theta * dt)
            + sigma * np.sqrt(1 - np.exp(-2 * theta * dt)) * rng.standard_normal()
        )
    return noise


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_datasets(
    t: np.ndarray,
    params: Dict[str, float] = TRUE_PARAMS,
    gaussian_sigmas: List[float] = (0.05, 0.15, 0.30, 0.60),
    pink_sigmas: List[float] = (0.05, 0.15, 0.30, 0.60),
    hetero_fracs: List[float] = (0.05, 0.10, 0.20, 0.40),
    ou_thetas: List[float] = (0.5, 2.0, 8.0),
    ou_sigma: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two DataFrames:

    signals_df  - one column per noisy realisation, plus 't' and 'x_clean'
    metadata_df - one row per realisation with noise type & level identifiers
    """
    rng = np.random.default_rng(seed)
    x_clean = damped_oscillator(t, **params)
    dt = float(t[1] - t[0])
    n = len(t)

    signals: Dict[str, np.ndarray] = {"t": t, "x_clean": x_clean}
    meta_rows: List[Dict] = []

    def _add(label: str, noise_type: str, level_param: str, level_value: float, noise: np.ndarray):
        signals[label] = x_clean + noise
        meta_rows.append(
            {
                "label": label,
                "noise_type": noise_type,
                "level_param": level_param,
                "level_value": level_value,
                "empirical_snr_db": _snr_db(x_clean, noise),
            }
        )

    for s in gaussian_sigmas:
        label = f"gaussian_s{s:.2f}"
        _add(label, "gaussian", "sigma", s, gaussian_noise(n, s, rng))

    for s in pink_sigmas:
        label = f"pink_s{s:.2f}"
        _add(label, "pink", "sigma", s, pink_noise(n, s, rng))

    for frac in hetero_fracs:
        label = f"hetero_f{frac:.2f}"
        floor = 0.02
        noise = heteroscedastic_noise(x_clean, frac, floor, rng)
        _add(label, "heteroscedastic", "sigma_frac", frac, noise)

    for theta in ou_thetas:
        label = f"ou_theta{theta:.1f}"
        noise = ornstein_uhlenbeck_noise(n, ou_sigma, theta, dt, rng)
        _add(label, "ornstein_uhlenbeck", "theta", theta, noise)

    signals_df = pd.DataFrame(signals)
    metadata_df = pd.DataFrame(meta_rows)
    return signals_df, metadata_df


def _snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return np.inf
    return 10.0 * np.log10(signal_power / noise_power)


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = np.linspace(0, 15, 600)
    signals_df, metadata_df = build_datasets(t)
    signals_df.to_csv("signals.csv", index=False)
    metadata_df.to_csv("metadata.csv", index=False)
    print("Saved signals.csv and metadata.csv")
    print(metadata_df.to_string(index=False))
