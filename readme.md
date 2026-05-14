# Antennas Surrogate Model

This repository contains a numerical simulator and preliminary surrogate-modeling workflow for a Gaussian communication channel between two harmonic oscillator detectors interacting through a scalar field.

The project is currently under active development. The numerical simulation pipeline is functional, while the first surrogate models are exploratory and not yet reliable enough for deployment or scientific use.

## Project goal

The goal is to approximate the energy-constrained classical capacity

$$
C_E(t)
$$

of the detector channel as a function of the physical input parameters:

$$
\gamma_A,\gamma_B,\omega_A,\omega_B
$$

where:

- $\gamma_A,\gamma_B$ are the detector-field coupling/damping parameters;
- $\omega_A,\omega_B$ are the oscillator frequencies;
- $t$ is the evolution time.

The final objective is to build a surrogate model able to rapidly approximate the full capacity curve $C_E(t)$, avoiding the expensive numerical simulation.

## Current status

The repository currently includes:

- a numerical simulation of the channel dynamics;
- computation of:
  - transmissivity $\tau(t)$;
  - noise determinant $W(t)$;
  - energy-constrained classical capacity $C_E(t)$;
- reliability diagnostics for simulated data;
- automatic training-data generation;
- preliminary scikit-learn and PyTorch surrogate models;
- a Streamlit interface for numerical simulation and model comparison.

The simulator is based on the Green-function formulation of the detector-channel model, where the transmissivity is computed from $G_{BA}$, $\dot{G}_{BA}$, and $\ddot{G}_{BA}$, while the noise is computed from the covariance/noise matrix determinant.

## Important limitation

The first surrogate models are **not yet accurate enough**.

The current models capture only a rough global trend of $C_E(t)$, but they can produce noisy or unreliable pointwise predictions, especially for non-trivial asymmetric inputs such as:

$$
\gamma_A \neq \gamma_B
$$

or

$$
\omega_A \neq \omega_B.
$$

Therefore, the trained models should currently be considered as **proof-of-concept models**, not final scientific surrogates.

## Why the first models are limited

The first approach used pointwise regression:

$$
(\gamma_A,\gamma_B,\omega_A,\omega_B,t) \rightarrow C_E(t).
$$

This does not explicitly enforce that the predicted values over time must form one smooth physical curve. As a result, the models may predict scattered time-point values instead of a coherent $C_E(t)$ function.

## Planned improvements

The next development step is to move from pointwise prediction to curve-level surrogate modeling.

Planned improvements include:

### 1. Curve-based surrogate model

Train models of the form:

$$
(\gamma_A,\gamma_B,\omega_A,\omega_B)
\rightarrow
[C_E(t_1),...,C_E(t_N)].
$$

### 2. PCA / basis-function surrogate

Decompose simulated capacity curves into smooth basis functions and train a regression model to predict the basis coefficients.

### 3. Physics-informed neural surrogate

Add physical constraints such as:

- $C_E(t) \geq 0$;
- $C_E(t<d)=0$;
- smoothness in time except around physical $|\tau|$-bounce regions;
- reliability-weighted loss using the numerical diagnostics.

### 4. Improved validation

Evaluate models at the full-curve level, including:

- pointwise MAE/RMSE;
- curve-level error;
- maximum-capacity error;
- time-of-maximum error.

## Repository structure

```text
.
├── app.py                         # Streamlit numerical simulation app
├── surrogate_app.py               # Streamlit app comparing numerical and surrogate outputs
├── generate_parameter_inputs.py   # Generates constrained input parameter samples
├── generate_training_data_full.py # Runs simulations and saves training batches
├── train_surrogate_ml.py          # scikit-learn baseline surrogate
├── train_surrogate_torch.py       # PyTorch neural-network surrogate
├── src/
│   ├── simulation.py              # Core numerical simulation code
│   ├── data_loading.py            # Batch loading and filtering utilities
│   ├── features.py                # Feature preprocessing utilities
│   ├── torch_models.py            # PyTorch model definitions
│   └── surrogate_predictors.py    # Model loading and prediction utilities
├── models/                        # Trained models, ignored except .gitkeep
├── outputs/                       # Generated data, ignored except .gitkeep
└── notebooks/                     # Experimental scripts/notebooks, ignored except .gitkeep