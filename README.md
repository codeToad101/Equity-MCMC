> **Bayesian HMC (NUTS) vs. GARCH(1,1)**

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research--complete-orange.svg)

## 📌 Project Overview
This repository contains a quantitative research framework comparing three distinct modeling paradigms for simulating asset log-returns. The goal is to evaluate how well different stochastic processes capture the statistical properties of equity indices, specifically focusing on **Tail Risk (Kurtosis)** and **Predictive Coverage**.

## 📊 Core Methodology

### 1. GARCH(1,1) Framework
The model assumes a constant mean with a time-varying variance process:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### 2. Bayesian HMC (NUTS)
Leveraging Hamiltonian Monte Carlo via the **No-U-Turn Sampler**, we explore the posterior distribution of volatility parameters. This approach allows for a probabilistic interpretation of uncertainty that traditional MLE-based GARCH models lack.

### 3. Sequential/Empirical Method
A particle-based approach used as a benchmark for capturing high-moment risks that parametric models often understate.

## 📈 Key Results
Below is a summary of performance across the **PREIX** (Equities) dataset:

| Metric | HMC (NUTS) | GARCH(1,1) | Empirical |
| :--- | :--- | :--- | :--- |
| **Observed Kurtosis** | 16.18 | 16.18 | 16.18 |
| **Simulated Kurtosis** | 3.91 | 4.72 | **11.59** |
| **95% Coverage** | 0.892 | **0.944** | N/A |

### Primary Findings
* **The "Kurtosis Gap":** Standard HMC and GARCH significantly underestimate tail risk compared to empirical sampling.
* **Coverage Accuracy:** GARCH provides superior interval calibration, while HMC offers tighter (more precise) but less accurate intervals.

## 🛠 Installation & Usage
1. Clone the repo: `git clone https://github.com/codeToad101/Equity-MCMC`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the simulation: `python run_simulations.py`

## 📝 Technical Paper
A full mathematical recap of the Hamiltonian dynamics and the PIT validation logic coming soon.