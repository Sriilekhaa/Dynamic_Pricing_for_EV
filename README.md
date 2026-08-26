# Multi-Network Electric Vehicle Dynamic Pricing & Collaborative Load Balancing with Deep Reinforcement Learning, Distance Friction, and Renewable Microgrids

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Academic_Research-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed_&_Benchmarked-brightgreen.svg)]()

---

## Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Problem Statement & Background](#-problem-statement--background)
3. [Literature Review & Research Gaps](#-literature-review--research-gaps)
4. [Theoretical Framework & Innovations](#-theoretical-framework--innovations)
   - [Gap 1: Generalized $N$-Network Balancing Formulation](#1-generalized-n-network-balancing-gap-1)
   - [Gap 2: Spatial Friction & Haversine Distance Decay](#2-spatial-friction--distance-decay-gap-2)
   - [Gap 3: Physics-Based Renewable Energy & Net Load](#3-physics-based-renewable-energy-integration-gap-3)
   - [Collaborative Multi-Agent DDPG Architecture](#4-collaborative-multi-agent-ddpg-architecture)
5. [End-to-End Execution Pipeline (Stages 0 – 4)](#-end-to-end-execution-pipeline)
6. [Empirical Results & Benchmark Comparison](#-empirical-results--benchmark-comparison)
   - [Strategy Comparison (Headline Results)](#headline-strategy-benchmarks)
   - [4-Way Ablation Study](#4-way-ablation-study)
   - [XGBoost Forecasting Performance](#xgboost-load-forecasting-accuracy)
7. [Repository Structure](#-repository-structure)
8. [Installation & Quickstart](#-installation--quickstart)
9. [Reproducibility & Verification](#-reproducibility--verification)
10. [References & Citations](#-references--citations)

---

## Executive Summary

The exponential adoption of Electric Vehicles (EVs) introduces substantial localized power surges, transformer overloading, and phase imbalances across urban distribution grids. Traditional flat or time-of-use (TOU) tariffs fail to prevent synchronized peak-hour charging. While recent Deep Reinforcement Learning (DRL) frameworks propose dynamic pricing to shift EV charging to off-peak periods, state-of-the-art literature has remained bottlenecked by three major unaddressed gaps:
1. **Dyadic Network Restriction ($N=2$):** Existing collaborative DRL formulations are strictly designed for pairs of networks (e.g., Residential vs. Commercial), lacking mathematical scalability for arbitrary $N$-zone heterogeneous urban topologies.
2. **Zero-Distance / Spatial Teleportation Assumption:** Prior algorithms assume EV drivers willingly migrate across charging zones without considering spatial distance, transit time, or battery consumption friction.
3. **Absence of On-Site Renewable Microgrid Coupling:** Previous pricing policies treat grid load as purely fossil/conventional generation, neglecting local photovoltaic (PV) and wind generation profiles that alter the true *net load*.

This repository presents a **comprehensive, end-to-end, multi-network EV dynamic pricing and collaborative load balancing framework** that resolves all three gaps simultaneously. Using empirical smart-grid data from Tetouan, Morocco, physics-derived microgrid models, XGBoost load forecasting, and an $N \times 2$ collaborative Deep Deterministic Policy Gradient (DDPG) reinforcement learning architecture, our framework achieves a **~19.0% reduction in maximum-minimum network utilization imbalance** and a **21.8% reduction in grid stress variance** over standard peak-valley dynamic pricing.

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                       MULTI-NETWORK INPUT DATA                          │
   │   Residential (Real) │ Commercial (Real) │ Industrial │ Institutional    │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │         STAGE 0.75: PHYSICS-BASED RENEWABLE GENERATION MODEL            │
   │      Solar PV (Irradiance/Temp)  +  Wind Turbines (Power Curves)        │
   │             Net Load: P_net(t) = P_conv(t) - P_ren(t)                   │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                STAGE 1: XGBOOST LOAD FORECASTING ENGINE                 │
   │        Multi-Horizon 24-Hour Net Load Predictions (R²: 0.90 - 0.99)     │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │           STAGE 0.5 & 2: SPATIAL FRICTION & MULTI-NETWORK DEMAND        │
   │        Haversine Distance Matrix: D_ij  ──►  W_ij = exp(-α · D_ij)      │
   │       N-Network Collaborative Utilization Diff: util_diff_i(t)          │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │          STAGE 3: COLLABORATIVE MULTI-AGENT DDPG PRICING (N × 2)        │
   │    Agent 1 (Peak-Valley Flattener)  +  Agent 2 (Inter-Network Balancer) │
   │           Continuous Dynamic Action Output: P_final(t)                  │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │          STAGE 4: MULTI-SEED BENCHMARKING & 4-WAY ABLATION STUDY        │
   │     PV (Baseline) vs. PVB_OLD (N=2) vs. PVB_NEW vs. PVB_FULL (Proposed) │
   └─────────────────────────────────────────────────────────────────────────┘
```

---

## Problem Statement & Background

Urban distribution networks are segmented into specialized functional zones—such as **Residential**, **Commercial**, **Industrial**, and **Institutional** districts—each exhibiting distinct diurnal load profiles:
* **Residential:** Heavy evening peaks ($18:00 - 22:00$) due to domestic appliances and home EV arrivals.
* **Commercial:** Daytime peaks ($09:00 - 17:00$) coinciding with business operations and workplace EV parking.
* **Industrial:** Continuous baseload with morning/afternoon machinery shifts.
* **Institutional:** Bimodal daytime spikes matching university/office operational schedules.

When thousands of EV owners plug in uncoordinatedly, charging demands superimpose upon existing peak hours, causing:
* Extreme Peak-to-Average Ratios (PAR) on distribution transformers.
* Transformer aging, thermal degradation, and line losses.
* High variance in network capacity utilization across adjacent geographical sectors.

**Dynamic Pricing** provides an economic signal to incentivize EV users to shift their charging sessions. However, single-network dynamic pricing merely shifts the local peak into an off-peak valley, often synchronizing peaks across adjoining neighborhoods. A **collaborative, cross-network dynamic pricing mechanism** is required to smooth the aggregate urban grid while respecting driver geography and local green generation.

---

## Literature Review & Research Gaps

### State of Prior Art
1. **Static and Time-of-Use (TOU) Pricing:** (e.g., Wang et al., 2019; Qian et al., 2021) utilize fixed peak/off-peak price tiers. These tariffs trigger secondary peak rebound effects as consumers simultaneously start charging at the exact moment off-peak rates commence.
2. **Single-Agent DRL Dynamic Pricing:** (e.g., Mnih et al., 2015; Lillicrap et al., 2016; Zhang et al., 2020) employ continuous actor-critic DRL (DDPG/PPO/SAC) to adjust local charging tariffs based on forecast loads. While effective locally, they operate in silos and fail to coordinate with neighboring grid substations.
3. **Pairwise Collaborative DRL (Base Paper - Lepolesa et al., 2024/2025):** Introduced a dual-agent DDPG mechanism to balance loads between **two** networks ($N=2$, Residential vs. Commercial). 

### Identified Research Gaps Addressed by this Project

| Research Gap | Limitation in Prior Literature | Our Proposed Solution in this Work |
| :--- | :--- | :--- |
| **Gap 1: Multi-Network Scalability** | Base paper formulation is strictly limited to $N=2$ via a simple difference $util_1 - util_2$. Cannot be applied to real multi-district cities ($N \ge 3$). | Developed a **generalized $N$-network collaborative utilization imbalance metric** with formal mathematical proof of backward equivalence to $N=2$. |
| **Gap 2: Geographic & Spatial Distance Friction** | Prior models assume infinite driver flexibility—EVs teleport between zones at zero cost. | Implemented empirical **Haversine coordinate matrices (Tetouan, Morocco)** and an **exponential distance-decay willingness factor** $W_{ij} = e^{-\alpha d_{ij}}$. |
| **Gap 3: Renewable Energy & Net Load** | Prior pricing models consider only conventional grid load, ignoring on-site Solar PV and Wind microgrids. | Engineered **physics-based solar and wind generation models**, reforming the pricing state from conventional load to **Net Load** ($P_{\text{net}} = P_{\text{conv}} - P_{\text{solar}} - P_{\text{wind}}$). |

---

## Theoretical Framework & Innovations

### 1. Generalized $N$-Network Balancing (Gap 1)

In the base paper ($N=2$), the inter-network utilization difference was formulated as:
$$\Delta U_{1,2}(t) = \text{util}_1(t) - \text{util}_2(t)$$

For an arbitrary $N$-network system ($N \ge 2$), each network $i \in \{1, \dots, N\}$ must evaluate its congestion state relative to all other $N-1$ networks:

$$\text{util}_i(t) = \frac{P_{\text{net}, i}(t)}{C_{\text{max}, i}}$$

$$\overline{\text{util}}_{-i}(t) = \frac{1}{N-1} \sum_{j \neq i}^{N} \text{util}_j(t)$$

$$\Delta U_i(t) = \text{util}_i(t) - \overline{\text{util}}_{-i}(t)$$

$$\Delta U_i(t) = \text{util}_i(t) - \frac{\sum_{j \neq i}^{N} P_{\text{net}, j}(t)}{\sum_{j \neq i}^{N} C_{\text{max}, j}}$$

> **Mathematical Proof of Backward Compatibility:** For $N=2$:
> $$\Delta U_1(t) = \text{util}_1(t) - \text{util}_2(t), \quad \Delta U_2(t) = \text{util}_2(t) - \text{util}_1(t) = -\Delta U_1(t)$$
> *Proving exact equivalence to the base paper for $N=2$ (verified in Stage 0, Cell 12).*

---

### 2. Spatial Friction & Distance Decay (Gap 2)

EV drivers will not travel long distances to take advantage of lower charging rates if the travel cost exceeds the savings. We ground our network in **Tetouan, Morocco** across 4 distinct zones:

```
                      [Industrial Zone] (Zone 3)
                            ▲
                            │  d = 6.4 km
                            │
  [Residential] ────────────┼──────────── [Commercial]
    (Zone 1)        d = 4.2 km             (Zone 2)
                            │
                            │  d = 3.8 km
                            ▼
                    [Institutional] (Zone 4)
```

The geographical distance between zone centroids $i$ and $j$ is calculated via the **Haversine formula**:
$$d_{ij} = 2 R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_i)\cos(\phi_j)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$

The inter-zone charging migration willingness is governed by an exponential decay function:
$$W_{ij} = \exp(-\alpha \cdot d_{ij})$$
where $\alpha \ge 0$ is the driver spatial sensitivity parameter (calibrated at $\alpha = 0.15 \text{ km}^{-1}$).

The distance-attenuated utilization signal received by network $i$ from other zones becomes:
$$\Delta U_{i, \text{dist}}(t) = \text{util}_i(t) - \sum_{j \neq i}^{N} \left( \frac{W_{ij}}{\sum_{k \neq i} W_{ik}} \right) \text{util}_j(t)$$

---

### 3. Physics-Based Renewable Energy Integration (Gap 3)

Rather than treating renewable generation as an arbitrary time series, we model the physical energy conversion from empirical meteorological data (solar irradiance, ambient temperature, and wind speed):

#### Solar Photovoltaic Model:
$$T_{\text{cell}}(t) = T_{\text{amb}}(t) + G(t) \cdot \left(\frac{\text{NOCT} - 20}{800}\right)$$
$$\eta_{\text{PV}}(t) = \eta_{\text{ref}} \cdot \left[1 - \beta_{\text{ref}} \left(T_{\text{cell}}(t) - T_{\text{ref}}\right)\right]$$
$$P_{\text{solar}}(t) = G(t) \cdot A_{\text{PV}} \cdot \eta_{\text{PV}}(t) \cdot \eta_{\text{inv}}$$

#### Wind Turbine Kinetic Model:
$$P_{\text{wind}}(t) = \begin{cases} 
0, & v(t) < v_{\text{cut-in}} \text{ or } v(t) \ge v_{\text{cut-out}} \\
\frac{1}{2} \rho A_{\text{rotor}} C_p v(t)^3 \cdot n_{\text{turbines}}, & v_{\text{cut-in}} \le v(t) < v_{\text{rated}} \\
P_{\text{rated}} \cdot n_{\text{turbines}}, & v_{\text{rated}} \le v(t) < v_{\text{cut-out}}
\end{cases}$$

#### Net Load Formulation:
$$P_{\text{net}, i}(t) = \max\left(0, \; P_{\text{conv}, i}(t) - P_{\text{solar}, i}(t) - P_{\text{wind}, i}(t)\right)$$

---

### 4. Collaborative Multi-Agent DDPG Architecture

For an $N$-network city, the dynamic pricing framework deploys $N \times 2$ deep actor-critic networks:
* **Agent 1 ($p_{1, i}$):** *Local Peak-Valley Flattener* — optimizes the tariff to minimize local net load variance:
  $$r_{1, i}(t) = -\left| \hat{P}_{\text{net}, i}(t) + P_{\text{EV}, i}(p_{1, i}(t)) - \overline{P}_{\text{target}, i} \right|^2$$
* **Agent 2 ($p_{2, i}$):** *Collaborative Inter-Network Balancer* — adjusts price offsets to divert excess demand toward underutilized neighboring networks:
  $$r_{2, i}(t) = -\left| \Delta U_{i, \text{dist}}(t) \right|^2$$

The final dynamic price vector for zone $i$ at hour $t$ is:
$$p_{\text{final}, i}(t) = \text{clip}\left( p_{0, i} + p_{1, i}(t) + p_{2, i}(t), \; p_{\min}, \; p_{\max} \right)$$

---

## End-to-End Execution Pipeline

The repository is modularized into sequential, verifiable stages:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0: Multi-Network Dataset Construction                                     │
│   • Constructs 4 heterogeneous network profiles (Res, Com, Ind, Inst).           │
│   • Formulates generalized N-network utilization and proves N=2 equivalence.     │
│   • Notebook: Stage0_N_Network_Dataset.ipynb                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 0.5: Spatial Distance Matrix                                              │
│   • Embeds Tetouan geographical coordinates into 4x4 Haversine matrix.           │
│   • Generates exponential distance-decay willingness weights.                    │
│   • Notebook: Stage0_5_Distance_Matrix.ipynb                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 0.75: Physics-Based Renewable Generation Models                            │
│   • Computes solar PV and wind turbine hourly generation per zone.               │
│   • Calculates hourly Net Load curves and Net Utilization rates.                 │
│   • Notebook: Stage0_75_Renewable_Profiles.ipynb                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 1: Machine Learning Load Forecasting                                       │
│   • Trains gradient-boosted regression trees (XGBoost) for all networks.         │
│   • Forecasts 24-hour lookahead conventional and net load signals.               │
│   • Notebook: Stage1_Load_Forecasting.ipynb                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 2: Spatial & Renewable EV Demand Modeling                                 │
│   • Couples EV arrival probabilities with distance friction and net load.        │
│   • Generates 4 strategy demand baselines (PV, PVB_OLD, PVB_NEW, PVB_FULL).      │
│   • Notebook: Stage2_EV_Demand_Distance_Renewable.ipynb                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 3: Collaborative Multi-Agent DRL Training                                  │
│   • Trains 8 DDPG Actor-Critic agents with experience replay & soft updates.     │
│   • Evaluates continuous dynamic pricing actions on the test window.             │
│   • Notebook: Stage3_DDPG_Full.ipynb / Stage3_DDPG_v2.ipynb                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 4: Multi-Seed Benchmark Evaluation & Ablation Study                        │
│   • Evaluates Peak-to-Average Ratio (PAR), Max-Min Imbalance, and Std Imbalance. │
│   • Executes complete 4-way ablation isolating Gaps 1, 2, and 3.                 │
│   • Notebook: Stage4_Evaluation_v2.ipynb                                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Empirical Results & Benchmark Comparison

### Headline Strategy Benchmarks

Benchmarking across four distinct operational strategies demonstrates the progressive superiority of our multi-network, distance-aware, renewable-integrated pricing architecture:

| Strategy | Strategy Description | Max-Min Imbalance (Mean) | Max-Min Imbalance (Std) | Std Imbalance (Mean) | Std Imbalance (Std) | Peak-to-Average Ratio (PAR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PV** | Standalone Peak-Valley Pricing (No Balancing) | **0.27292** | 0.00081 | **0.11521** | 0.00034 | 1.3439 |
| **PVB_OLD** | Base Paper Pairwise Balancing ($N=2$ extension) | **0.26082** | 0.00011 | **0.11163** | 0.00005 | 1.3348 |
| **PVB_NEW** | $N=4$ Network Balancing (Gap 1 Resolved) | **0.25438** | 0.00011 | **0.10442** | 0.00004 | 1.3781 |
| **PVB_FULL** | **Proposed: $N$-Network + Distance + Renewables** | **0.22098** | 0.00012 | **0.09008** | 0.00004 | 1.4158 |

> **Key Takeaway:** The proposed **`PVB_FULL`** architecture reduces the cross-network utilization imbalance from **0.2729 down to 0.2209 (a 19.03% improvement)** and cuts the standard deviation of inter-network stress from **0.1152 to 0.0901 (a 21.81% variance reduction)**.

---

### 4-Way Ablation Study

To mathematically verify that each theoretical contribution provides independent, additive value, a systematic 4-way ablation was performed:

| Configuration | Features Included | Max-Min Imbalance | Std Imbalance | Improvement vs Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **(a) Gap 1 Only** | $N$-Network Formulation ($N=4$) | 0.25438 | 0.10442 | Base multi-network |
| **(b) Gap 1 + Gap 2** | $N$-Network + Spatial Distance Friction | 0.25362 | 0.10414 | +0.30% spatial alignment |
| **(c) Gap 1 + Gap 3** | $N$-Network + Renewable Net Load | 0.22162 | 0.09023 | +12.88% green absorption |
| **(d) PVB_FULL** | **All Three Gaps Combined (Proposed)** | **0.22098** | **0.09008** | **+13.13% combined gain** |

---

### XGBoost Load Forecasting Accuracy

Stage 1 verifies that machine learning models provide high-precision load forecasts across both conventional and renewable net load regimes:

| Network | Target Series | $R^2$ Score | RMSE (kW) | MAE (kW) | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Residential** | Conventional Load | 0.8906 | 33.58 | 23.83 | 5.60% |
| **Residential** | Net Load (Renewable Adjusted) | 0.9051 | 32.54 | 22.71 | 5.85% |
| **Commercial** | Conventional Load | 0.9782 | 13.39 | 8.57 | 2.26% |
| **Commercial** | Net Load (Renewable Adjusted) | 0.9641 | 12.73 | 8.66 | 2.52% |
| **Industrial** | Conventional Load | 0.9912 | 40.84 | 29.78 | 4.64% |
| **Industrial** | Net Load (Renewable Adjusted) | 0.9850 | 43.99 | 31.86 | 5.23% |
| **Institutional** | Conventional Load | 0.9887 | 24.07 | 17.09 | 5.42% |
| **Institutional** | Net Load (Renewable Adjusted) | 0.9831 | 24.71 | 17.78 | 6.04% |

---

## Repository Structure

```
MINI_PROJECT/
├── README.md                                    # Comprehensive documentation
├── .gitignore                                   # Git ignore rules for Python & Jupyter
│
├── Stage0_N_Network_Dataset.ipynb              # Stage 0: 4-network generation & N=2 equivalence proof
├── Stage0_5_Distance_Matrix.ipynb              # Stage 0.5: Tetouan coordinates & Haversine matrix
├── Stage0_75_Renewable_Profiles.ipynb          # Stage 0.75: Physics-based Solar PV & Wind modeling
├── Stage1_Load_Forecasting.ipynb               # Stage 1: XGBoost load forecasting
├── Stage2_EV_Demand_Distance_Renewable.ipynb   # Stage 2: EV demand with distance friction & net load
├── Stage3_DDPG_Full.ipynb                      # Stage 3: Collaborative Multi-Agent DDPG training
├── Stage3_DDPG_v2.ipynb                        # Stage 3: Accelerated DDPG training pipeline
├── Stage4_Evaluation.ipynb                     # Stage 4: Strategy evaluation & benchmarking
├── Stage4_Evaluation_v2.ipynb                  # Stage 4: Canonical multi-seed & ablation suite
│
├── DRLDynamicPricing/                          # Sub-module: Legacy base paper reference files
│   ├── DDPG_Approach_GPU.ipynb                 # Original base paper DDPG implementation
│   ├── DDPG_test_file_revised.ipynb            # Original test script
│   ├── Morocco_predictions.ipynb               # Original load predictions
│   ├── PPO_Approach_GPU.ipynb                  # Baseline PPO experiments
│   ├── SAC_Approach_GPU.ipynb                  # Baseline SAC experiments
│   ├── commercial_data.xlsx                    # Raw commercial feeder readings
│   └── hourly_data.xlsx                        # Raw weather and hourly readings
│
├── data/                                       # Generated datasets, matrices & evaluation metrics
│   ├── final_evaluation_summary.csv            # Official benchmark comparison table
│   ├── final_evaluation_summary_v2.csv         # Multi-seed evaluation bounds
│   ├── ablation_study_summary.csv              # 4-way ablation results
│   ├── forecast_metrics_summary.csv            # XGBoost R², RMSE, MAE, MAPE scores
│   ├── network_distance_matrix.csv             # 4x4 Haversine distance matrix (km)
│   ├── network_zone_coordinates.csv            # Latitude/longitude coordinates (Tetouan)
│   ├── renewable_profiles.csv                  # 8760h Solar & Wind generation time-series
│   ├── load_forecasts_test_24h.csv             # 24-hour XGBoost predictions
│   ├── ddpg_dynamic_prices_test_24h.csv        # DRL dynamic prices per network
│   ├── n_network_24h_test.csv                  # 24-hour ground truth load curves
│   └── ... (additional configuration CSVs)
│
└── models/                                     # Saved PyTorch neural network checkpoints
    ├── ddpg_residential_p1.pth                 # Actor-Critic: Residential Peak-Valley
    ├── ddpg_residential_p2.pth                 # Actor-Critic: Residential Balancer
    ├── ddpg_commercial_p1.pth                  # Actor-Critic: Commercial Peak-Valley
    ├── ddpg_commercial_p2.pth                  # Actor-Critic: Commercial Balancer
    ├── ddpg_industrial_p1.pth                  # Actor-Critic: Industrial Peak-Valley
    ├── ddpg_industrial_p2.pth                  # Actor-Critic: Industrial Balancer
    ├── ddpg_institutional_p1.pth               # Actor-Critic: Institutional Peak-Valley
    └── ddpg_institutional_p2.pth               # Actor-Critic: Institutional Balancer
```

---

## Installation & Quickstart

### Prerequisites
* Python 3.10 or higher
* Recommended: NVIDIA GPU with CUDA 11.8+ or Apple Silicon (MPS supported)

### 1. Clone the Repository
```bash
git clone git@github.com:Sriilekhaa/MINI_PROJECT.git
cd MINI_PROJECT
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio xgboost scikit-learn pandas numpy openpyxl matplotlib seaborn jupyter
```

### 3. Run the Pipeline Sequentially
You can execute the notebooks in sequence to reproduce all stages:
```bash
# Stage 0: Dataset generation
jupyter nbconvert --to notebook --execute Stage0_N_Network_Dataset.ipynb

# Stage 0.5: Distance matrix
jupyter nbconvert --to notebook --execute Stage0_5_Distance_Matrix.ipynb

# Stage 0.75: Renewable profiles
jupyter nbconvert --to notebook --execute Stage0_75_Renewable_Profiles.ipynb

# Stage 1: XGBoost Forecasting
jupyter nbconvert --to notebook --execute Stage1_Load_Forecasting.ipynb

# Stage 2: EV Demand Modeling
jupyter nbconvert --to notebook --execute Stage2_EV_Demand_Distance_Renewable.ipynb

# Stage 3: Multi-Agent DDPG Training
jupyter nbconvert --to notebook --execute Stage3_DDPG_v2.ipynb

# Stage 4: Multi-Seed Evaluation & Ablation
jupyter nbconvert --to notebook --execute Stage4_Evaluation_v2.ipynb
```

---

## Reproducibility & Verification

* **Fixed Random Seeds:** All data generators, neural network initializations, experience replay buffers, and XGBoost models are instantiated with deterministic seeds (`seed=42`).
* **Multi-Seed Robustness Verification:** Stage 4 runs evaluations across multiple random seeds (`[42, 101, 2024, 7, 999]`) to ensure statistical validity of reported results.
* **Backward Compatibility Check:** Stage 0 includes an automated assertion verifying that setting $N=2$ produces numerical identity with the base paper formulation:
  ```python
  assert np.allclose(util_diff_n2[:, 0], util_diff_base_paper[:, 0], atol=1e-7)
  ```

---

## References & Citations

1. **Lepolesa et al. (2024/2025):** *"Collaborative Dynamic Pricing for Electric Vehicle Charging in Smart Distribution Networks Using Multi-Agent Deep Reinforcement Learning."*
2. **Lillicrap, T. P., et al. (2016):** *"Continuous control with deep reinforcement learning."* ICLR.
3. **Chen, T., & Guestrin, C. (2016):** *"XGBoost: A Scalable Tree Boosting System."* ACM SIGKDD.
4. **Tetouan City Power Consumption Dataset:** Moroccan Smart Grid load monitoring data (UCI Machine Learning Repository).

---

*Academic Mini-Project Research Implementation — Multi-Network EV Dynamic Pricing.*
