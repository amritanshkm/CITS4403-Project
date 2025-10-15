# Household Insolvency Simulation

**CITS4403 – Computational Modelling**
**Income Shock Propagation & Resilience Dynamics**

**Authors:**

* Sanchia Recson Lakkarvi — 24732787
* Amritansh Kaur Mamotra — 24703293

---

## Overview

This agent-based model simulates how macroeconomic shocks—specifically unemployment—impact household insolvency, inequality, and recovery dynamics over time. Households are placed on a 2D grid with heterogeneous income, resilience, and wealth attributes. The simulation integrates real quarterly Australian data and replicates insolvency trends with strong correlation to AFSA records.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run simulation

Use the main notebook:

```
src/simulation.ipynb
```

Outputs (plots and data files) are saved in `output/`.

---

## Model Summary

### Agents

Each household has:

* Wealth
* Employment status
* Income rate (decile-based)
* Resilience (recovery speed)
  Placed on a non-interacting 2D grid.

### Environment

* Time step = 1 quarter
* Exogenous unemployment rate drives shocks

### Behaviour

Each quarter, agents:

* Earn income (if employed)
* Consume wealth
* Can become unemployed or recover
* Declare insolvency below a threshold

### Scenarios

**Baseline** – stable unemployment trend
**Shock** – elevated unemployment shocks and variance

---

## Key Findings

* Unemployment shocks sharply increase insolvency rates
* Financially weak and low-resilience households are most affected
* Shock scenarios accelerate income & wealth inequality
* Recovery speed depends on employment rebound and liquidity
* Model aligns with AFSA data (r ≈ 0.8–0.84)
