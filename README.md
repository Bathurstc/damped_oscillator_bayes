# Bayesian Parameter Inference for a Damped Harmonic Oscillator

Bayesian inference of physical parameters from noisy time-series data using [PyMC](https://www.pymc.io/). The project recovers the damping coefficient, angular frequency, amplitude, and phase of a damped harmonic oscillator from synthetic observations, and systematically evaluates inference quality across four physically motivated noise regimes.

This work is motivated by parameter estimation problems in experimental physics — specifically low-SNR signal recovery from detector data — where correctly characterizing uncertainty is as important as the point estimate itself.

---

## Physical Model

The displacement of a damped harmonic oscillator is given by:

```
x(t) = A · exp(−γt) · cos(ωt + φ)
```

| Parameter | Symbol | True value used |
|-----------|--------|----------------|
| Amplitude | A | 2.0 |
| Damping coefficient | γ | 0.3 s⁻¹ |
| Angular frequency | ω | 2.5 rad/s |
| Initial phase | φ | 0.4 rad |

---

## Noise Models

A key feature of this project is that inference is evaluated not just under idealized Gaussian noise, but across four noise types that reflect real measurement environments:

| Noise Type | Physical Motivation | Parameters Swept |
|---|---|---|
| **Gaussian (white)** | Thermal noise, ADC quantization error, uncorrelated sensor noise | σ ∈ {0.05, 0.15, 0.30, 0.60} |
| **Pink (1/f)** | Mechanical vibration background, environmental fluctuations | σ ∈ {0.05, 0.15, 0.30, 0.60} |
| **Heteroscedastic** | Sensor gain uncertainty, ADC nonlinearity, amplitude-dependent drag | σ_frac ∈ {0.05, 0.10, 0.20, 0.40} |
| **Ornstein-Uhlenbeck** | Slowly drifting ambient temperature/pressure, correlated sensor drift | θ ∈ {0.5, 2.0, 8.0} |

Each noise realization is stored alongside its empirical SNR (dB), enabling direct comparison of posterior quality as a function of signal-to-noise ratio.

---

## Project Structure

```
damped_oscillator_bayes/
├── data_generation.py       # Synthetic data generation: physical model + noise models
├── bayesian_inference.ipynb # Full Bayesian inference workflow in PyMC
├── signals.csv              # Generated noisy time-series (one column per noise realization)
├── metadata.csv             # Noise type, level, and empirical SNR for each realization
├── results/                 # Output plots: posteriors, trace plots, predictive checks
└── requirements.txt         # Python dependencies
```

---

## Bayesian Workflow

The notebook `bayesian_inference.ipynb` walks through the full inference pipeline:

1. **Data loading** — reads from `signals.csv` and `metadata.csv`; selects noise realizations for analysis
2. **Prior specification** — weakly informative priors placed on A, γ, ω, φ, and the noise scale σ
3. **Model definition** — PyMC generative model using the damped oscillator likelihood
4. **MCMC sampling** — posterior sampling via NUTS (No-U-Turn Sampler)
5. **Posterior diagnostics** — trace plots, R-hat convergence checks, effective sample size
6. **Posterior predictive checks** — simulated observations drawn from the posterior vs. observed data
7. **Parameter recovery** — posterior means and 94% HDI intervals compared against true values across noise conditions

---

## Installation

```bash
git clone https://github.com/Bathurstc/damped_oscillator_bayes.git
cd damped_oscillator_bayes
pip install -r requirements.txt
```

**Regenerate synthetic data** (optional — `signals.csv` and `metadata.csv` are included):

```bash
python data_generation.py
```

**Run the inference notebook:**

```bash
jupyter notebook bayesian_inference.ipynb
```

---

## Requirements

See `requirements.txt`. Core dependencies:

- `pymc` — probabilistic programming and MCMC sampling
- `numpy` / `scipy` — numerical computing and signal generation
- `pandas` — dataset management
- `matplotlib` / `arviz` — visualization and posterior diagnostics

---

## Connection to Experimental Physics

The noise models implemented here directly parallel those encountered in cryogenic particle detector experiments:

- **Ornstein-Uhlenbeck noise** models the kind of slowly correlated environmental drift (temperature, vibration) that drives data quality cuts in long detector runs
- **Heteroscedastic noise** reflects amplitude-dependent detector response, common in sensors operating near threshold
- **Pink noise** appears in the frequency spectra of many physical backgrounds, including mechanical and electromagnetic interference

The inference approach — prior selection, MCMC-based posterior estimation, and posterior predictive checks — mirrors the statistical workflow used in dark matter direct detection analyses, where correctly propagating uncertainty into a final exclusion limit is essential.

---

## Author

Corey R. Bathurst, Ph.D.
[linkedin.com/in/corey-bathurst](https://www.linkedin.com/in/corey-bathurst) | [github.com/Bathurstc](https://github.com/Bathurstc)
