# Antennas Surrogate Model

This repository contains an interactive numerical simulator for a physics-based communication channel between two harmonic-oscillator detectors interacting through a scalar field.

The project is the first step toward building a surrogate model for a complex physical simulation. The current app computes the channel quantities directly from the numerical model and provides visual diagnostics for parameter exploration.

## Current Features

- Streamlit interface for inserting physical and numerical parameters
- Numerical solution of the delayed Green-function equations
- Computation of:
  - transmissivity \(\tau(t)\)
  - noise determinant \(W(t)=\det N(t)\)
  - energy-bounded classical capacity \(C_E(t)\)
- Automatic calculation of:
  - \(\Sigma_A^2\)
  - \(\Sigma_B^2\)
  - maximum value of \(C_E(t)\)
  - time and delay at which the maximum occurs
- Inline plots for:
  - \(\tau(t)\)
  - \(W(t)\)
  - comparative log plot of \(|\tau(t)|\) and \(\sqrt{W(t)}\)
  - \(C_E(t)\)

## Physical Context

The model describes two non-identical harmonic oscillator detectors, labelled \(A\) and \(B\), coupled to a background scalar field.

The detectors act as localized quantum antennas. The communication channel is characterized by the transmissivity \(\tau(t)\), the noise determinant \(W(t)\), and the energy-constrained classical capacity:

\[
C_E =
h\left(
\frac{E}{\omega_A}|\tau|+\sqrt{W}
\right)
-
h\left(
\frac{|\tau|}{2}+\sqrt{W}
\right)
\]

where

\[
h(x)=
\left(x+\frac12\right)\log_2\left(x+\frac12\right)
-
\left(x-\frac12\right)\log_2\left(x-\frac12\right).
\]

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   └── simulation.py
└── outputs/
    └── .gitkeep