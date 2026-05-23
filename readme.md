# Quantum Antennas Communication — Gaussian Channel Surrogate Model

This project simulates the long-time interaction channel between two non-identical harmonic oscillator detectors and builds surrogate models for the energy-constrained classical capacity $C_E(t)$.

The workflow combines:

1. a numerical simulation of the physical channel;
2. automated generation of training data;
3. curve-level surrogate modelling;
4. a Streamlit app comparing numerical and surrogate predictions.

The main trainable input parameters are:

$$
\gamma_A,\quad \gamma_B,\quad \omega_A,\quad \omega_B.
$$

The remaining physical and numerical parameters are fixed to the values used during training.

---

## Project goal

The numerical simulation computes:

- the transmissivity $\tau(t)$;
- the noise determinant $W(t)$;
- the energy-constrained classical capacity $C_E(t)$.

The final surrogate model aims to approximate the full capacity curve:

$$
(\gamma_A,\gamma_B,\omega_A,\omega_B)
\longrightarrow
C_E(t_1),\ldots,C_E(t_{200}).
$$

This is useful because the full numerical simulation can be slow and may become unstable for some parameter configurations.

---

## Current model strategy

Earlier pointwise surrogates were tested:

$$
(\gamma_A,\gamma_B,\omega_A,\omega_B,t)\longrightarrow C_E(t).
$$

They captured the broad trend but produced noisy, time-incoherent curves. The current approach therefore uses **curve-level surrogates**.

The current main surrogate is:

- **PyTorch curve-output neural network**  
  Predicts the full $C_E(t)$ curve directly.

A PCA-based scikit-learn model is also supported architecturally, but the trained PCA-ML binary is not tracked if it exceeds GitHub’s standard file-size limit.

---

## Repository structure

```text
.
├── app.py
├── surrogate_app.py
├── readme.md
├── requirements.txt
│
├── generate_parameter_inputs.py
├── generate_training_data_test.py
├── generate_training_data_full.py
├── prepare_curve_dataset.py
│
├── train_surrogate_pca_ml.py
├── train_surrogate_curve_torch.py
│
├── src/
│   ├── __init__.py
│   ├── simulation.py
│   ├── bounce_detection.py
│   ├── surrogate_predictors.py
│   └── torch_curve_models.py
│
├── models/
│   ├── .gitkeep
│   ├── torch_curve_capacity_mlp.pt
│   ├── torch_curve_feature_scaler.joblib
│   ├── torch_curve_metrics.json
│   ├── torch_curve_loss_curve.png
│   ├── torch_curve_example_curves.png
│   ├── torch_curve_per_curve_rmse_hist.png
│   ├── torch_curve_worst_curves.png
│   │
│   ├── pca_ml_basis.joblib
│   ├── pca_ml_feature_scaler.joblib
│   ├── pca_ml_metrics.json
│   ├── pca_ml_example_curves.png
│   ├── pca_ml_explained_variance.png
│   ├── pca_ml_per_curve_rmse_hist.png
│   └── pca_ml_worst_curves.png
│
├── experiments/
│   └── pointwise/
│       ├── train_surrogate_ml.py
│       ├── train_surrogate_torch.py
│       ├── data_loading.py
│       ├── features.py
│       └── torch_models.py
│
├── outputs/
│   └── .gitkeep
│
└── notebooks/
    └── .gitkeep