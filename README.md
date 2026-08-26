# Multi-Network Electric Vehicle Dynamic Pricing & Collaborative Load Balancing with Deep Reinforcement Learning, Distance Friction, and Renewable Microgrids

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Academic_Research-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed_&_Benchmarked-brightgreen.svg)]()

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Problem Statement & Background](#-problem-statement--background)
3. [Literature Review & Research Gaps](#-literature-review--research-gaps)
4. [Theoretical Framework & Equations](#-theoretical-framework--equations)
   - [Gap 1: Generalized $N$-Network Balancing Formulation](#1-generalized-n-network-balancing-gap-1)
   - [Gap 2: Spatial Friction & Haversine Distance Decay](#2-spatial-friction--distance-decay-gap-2)
   - [Gap 3: Physics-Based Renewable Energy & Net Load](#3-physics-based-renewable-energy-integration-gap-3)
   - [Collaborative Multi-Agent DDPG Architecture & Exact Pricing Equation](#4-collaborative-multi-agent-ddpg-architecture--exact-pricing-equation)
5. [End-to-End Execution Pipeline (Stages 0 – 4)](#-end-to-end-execution-pipeline)
6. [Empirical Results & Benchmark Comparison](#-empirical-results--benchmark-comparison)
   - [Strategy Comparison (Headline Results)](#headline-strategy-benchmarks)
   - [4-Way Ablation Study](#4-way-ablation-study)
   - [Analysis of Peak-to-Average Ratio (PAR) Dynamics](#analysis-of-peak-to-average-ratio-par-dynamics)
   - [XGBoost Forecasting Performance](#xgboost-load-forecasting-accuracy)
7. [Repository Structure](#-repository-structure)
8. [Installation & Quickstart](#-installation--quickstart)
9. [Reproducibility & Verification Protocol](#-reproducibility--verification-protocol)
10. [References & Citations](#-references--citations)

---

## 🔭 Executive Summary

The exponential adoption of Electric Vehicles (EVs) introduces substantial localized power surges, transformer overloading, and phase imbalances across urban distribution grids. Traditional flat or time-of-use (TOU) tariffs fail to prevent synchronized peak-hour charging. While recent Deep Reinforcement Learning (DRL) frameworks propose dynamic pricing to shift EV charging to off-peak periods, state-of-the-art literature has remained bottlenecked by three major unaddressed gaps:
1. **Dyadic Network Restriction ($N=2$):** Existing collaborative DRL formulations are strictly designed for pairs of networks (e.g., Residential vs. Commercial), lacking mathematical scalability for arbitrary $N$-zone heterogeneous urban topologies.
2. **Zero-Distance / Spatial Teleportation Assumption:** Prior algorithms assume EV drivers willingly migrate across charging zones without considering spatial distance, transit time, or battery consumption friction.
3. **Absence of On-Site Renewable Microgrid Coupling:** Previous pricing policies treat grid load as purely fossil/conventional generation, neglecting local photovoltaic (PV) and wind generation profiles that alter the true *net load*.

This repository presents a **multi-network EV dynamic pricing and collaborative load balancing framework** extending the work of *Lepolesa et al. (IEEE Transactions on Smart Grid, 2025)* across all three gaps. Using empirical smart-grid data from Tetouan, Morocco, physics-derived microgrid models, XGBoost load forecasting, and an $N \times 2$ collaborative Deep Deterministic Policy Gradient (DDPG) reinforcement learning architecture, our framework achieves a **19.03% reduction in maximum-minimum network utilization imbalance** and a **21.81% reduction in cross-network utilization variance** over standard peak-valley dynamic pricing.

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                       MULTI-NETWORK INPUT DATA                          │
   │   Residential (Real) │ Commercial (Real) │ Industrial │ Institutional    │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │         STAGE 0.75: PHYSICS-BASED RENEWABLE GENERATION MODEL            │
   │      Solar PV (Irradiance/Temp)  +  Wind Turbines (Power Curves)        │
   │     Net Load: P_net(t) = max(0.05 * P_conv, P_conv - P_solar - P_wind)  │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                STAGE 1: XGBOOST LOAD FORECASTING ENGINE                 │
   │        Multi-Horizon 24-Hour Net Load Predictions (R²: 0.89 - 0.99)     │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │           STAGE 0.5 & 2: SPATIAL FRICTION & MULTI-NETWORK DEMAND        │
   │        Haversine Distance Matrix: D_ij  ──►  W_ij = exp(-α · D_ij)      │
   │     Distance-Weighted Relieved Util Diff: util_diff_dist_i(t)           │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │          STAGE 3: COLLABORATIVE MULTI-AGENT DDPG PRICING (N × 2)        │
   │    Agent 1 (Peak-Valley Flattener)  +  Agent 2 (Inter-Network Balancer) │
   │           Continuous Dynamic Action Output: P_final(t)                  │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │          STAGE 4: MULTI-SEED BENCHMARKING & 4-WAY ABLATION SUITE        │
   │     PV (Baseline) vs. PVB_OLD (N=2) vs. PVB_NEW vs. PVB_FULL (Proposed) │
   └─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Problem Statement & Background

Urban distribution networks are segmented into specialized functional zones—such as **Residential**, **Commercial**, **Industrial**, and **Institutional** districts—each exhibiting distinct diurnal load profiles:
* **Residential:** Heavy evening peaks ($18:00 - 22:00$) due to domestic appliances and home EV arrivals.
* **Commercial:** Daytime peaks ($09:00 - 17:00$) coinciding with business operations and workplace EV parking.
* **Industrial:** Continuous baseload with morning/afternoon machinery shifts ($06:00 - 18:00$).
* **Institutional:** Bimodal daytime spikes ($08:00 - 12:00$ and $14:00 - 17:00$) matching university/hospital operational schedules.

When thousands of EV owners plug in uncoordinatedly, charging demands superimpose upon existing peak hours, causing:
* Increased Peak-to-Average Ratios (PAR) on distribution transformers.
* Transformer aging, thermal degradation, and line losses.
* High variance in network capacity utilization across adjacent geographical sectors.

**Dynamic Pricing** provides an economic signal to incentivize EV users to shift their charging sessions. However, single-network dynamic pricing merely shifts the local peak into an off-peak valley, often synchronizing peaks across adjoining neighborhoods. A **collaborative, cross-network dynamic pricing mechanism** is required to smooth the aggregate urban grid while respecting driver geography and local green generation.

---

## 📚 Literature Review & Research Gaps

### State of Prior Art
1. **Static and Time-of-Use (TOU) Pricing:** (e.g., Wang et al., 2019; Qian et al., 2021) utilize fixed peak/off-peak price tiers. These tariffs trigger secondary peak rebound effects as consumers simultaneously start charging at the exact moment off-peak rates commence.
2. **Single-Agent DRL Dynamic Pricing:** (e.g., Mnih et al., 2015; Lillicrap et al., 2016; Zhang et al., 2020) employ continuous actor-critic DRL (DDPG/PPO/SAC) to adjust local charging tariffs based on forecast loads. While effective locally, they operate in silos and fail to coordinate with neighboring grid substations.
3. **Pairwise Collaborative DRL (Base Paper - Lepolesa et al., IEEE Trans. Smart Grid, 2025):** Introduced a dual-agent DDPG mechanism to balance loads between **two** networks ($N=2$, Residential vs. Commercial).

### Identified Research Gaps Addressed by this Project

| Research Gap | Limitation in Prior Literature | Our Proposed Solution in this Work |
| :--- | :--- | :--- |
| **Gap 1: Multi-Network Scalability** | Base paper formulation is strictly limited to $N=2$ via a simple difference $\text{util}_1 - \text{util}_2$. Cannot be applied to real multi-district cities ($N \ge 3$). | Developed a **generalized $N$-network collaborative utilization imbalance metric** with formal mathematical proof of backward equivalence to $N=2$. |
| **Gap 2: Geographic & Spatial Distance Friction** | Prior models assume infinite driver flexibility—EVs teleport between zones at zero cost. | Implemented empirical **Haversine coordinate matrices (Tetouan, Morocco)** and an **exponential distance-decay willingness factor** $W_{ij} = e^{-\alpha d_{ij}}$. |
| **Gap 3: Renewable Energy & Net Load** | Prior pricing models consider only conventional grid load, ignoring on-site Solar PV and Wind microgrids. | Engineered **physics-based solar and wind generation models**, reforming the pricing state from conventional load to **Net Load** ($P_{\text{net}} = \max(0.05 \cdot P_{\text{conv}}, P_{\text{conv}} - P_{\text{solar}} - P_{\text{wind}})$). |

---

## 🔬 Theoretical Framework & Equations

### 1. Generalized $N$-Network Balancing (Gap 1)

In the base paper ($N=2$), the inter-network utilization difference was formulated as:
$$\Delta U_{1,2}(t) = \text{util}_1(t) - \text{util}_2(t)$$

For an arbitrary $N$-network system ($N \ge 2$), each network $i \in \{1, \dots, N\}$ evaluates its congestion state relative to the unweighted mean utilization of all other $N-1$ networks:

$$\text{util}_i(t) = \frac{P_{\text{net}, i}(t)}{C_{\text{max}, i}}$$

$$\overline{\text{util}}_{-i}(t) = \frac{1}{N-1} \sum_{j \neq i}^{N} \text{util}_j(t)$$

$$\Delta U_i(t) = \max\left(\text{util}_i(t) - \overline{\text{util}}_{-i}(t), \; 0\right)$$

> **Mathematical Proof of Backward Compatibility:** For $N=2$:
> $$\Delta U_1(t) = \max(\text{util}_1(t) - \text{util}_2(t), 0), \quad \Delta U_2(t) = \max(\text{util}_2(t) - \text{util}_1(t), 0)$$
> *Proving exact mathematical equivalence to the base paper pairwise formulation when $N=2$ (verified in Stage 0, Cell 12).*

---

### 2. Spatial Friction & Distance Decay (Gap 2)

EV drivers will not travel long distances to take advantage of lower charging rates if the travel cost exceeds the savings. We ground our network in **Tetouan, Morocco** across 4 distinct zones:

```
                      [Industrial Zone] (Zone 3)
                            ▲
                            │  d = 5.52 km
                            │
  [Residential] ────────────┼──────────── [Commercial]
    (Zone 1)        d = 1.16 km             (Zone 2)
                            │
                            │  d = 1.54 km
                            ▼
                     [Institutional] (Zone 4)
```

The geographical distance between zone centroids $i$ and $j$ is calculated via the **Haversine formula**:
$$d_{ij} = 2 R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_i)\cos(\phi_j)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$

The inter-zone charging migration willingness is governed by an exponential decay function:
$$W_{ij} = \exp(-\alpha \cdot d_{ij}), \quad W_{ii} = 0$$
where $\alpha \ge 0$ is the driver spatial sensitivity parameter ($\alpha = 0.15 \text{ km}^{-1}$).

The distance-attenuated utilization difference signal received by network $i$ from other zones becomes:
$$\Delta U_{i, \text{dist}}(t) = \max\left(\text{util}_i(t) - \sum_{j \neq i}^{N} \left( \frac{W_{ij}}{\sum_{k \neq i} W_{ik}} \right) \text{util}_j(t), \; 0\right)$$

---

### 3. Physics-Based Renewable Energy Integration (Gap 3)

Rather than treating renewable generation as an arbitrary time series, we model energy conversion directly from empirical Tetouan meteorological readings:

#### Solar Photovoltaic Model:
$$P_{\text{solar}, i}(t) = G(t) \cdot A_{\text{PV}, i} \cdot \eta_{\text{PV}} \cdot \eta_{\text{inv}}$$
where $G(t) = \text{GeneralDiffuseFlows}(t) + \text{DiffuseFlows}(t)$ ($\text{W/m}^2$), with panel efficiency $\eta_{\text{PV}} = 18\%$ and inverter efficiency $\eta_{\text{inv}} = 95\%$.

#### Wind Turbine Kinetic Model:
$$P_{\text{wind}, i}(t) = \begin{cases} 
0, & v(t) < v_{\text{cut-in}} \text{ or } v(t) \ge v_{\text{cut-out}} \\
\frac{1}{2} \rho A_{\text{rotor}} C_p v(t)^3 \cdot n_{\text{turbines}, i}, & v_{\text{cut-in}} \le v(t) < v_{\text{rated}} \\
P_{\text{rated}} \cdot n_{\text{turbines}, i}, & v_{\text{rated}} \le v(t) < v_{\text{cut-out}}
\end{cases}$$

#### Net Load Formulation (with 5% Conventional Baseload Floor):
$$P_{\text{net}, i}(t) = \max\left(0.05 \cdot P_{\text{conv}, i}(t), \; P_{\text{conv}, i}(t) - P_{\text{solar}, i}(t) - P_{\text{wind}, i}(t)\right)$$

---

### 4. Collaborative Multi-Agent DDPG Architecture & Exact Pricing Equation

For an $N$-network city, the dynamic pricing framework deploys $N \times 2$ deep actor-critic agents (P1 and P2 per network). Following the codebase implementation:

* **Actor Architecture:** $24 \rightarrow 64 \rightarrow 64 \rightarrow 24$, activated via $\text{ReLU}$ and outputting continuous actions in $[0, 1]$ via $\frac{\tanh(z) + 1}{2}$.
* **Critic Architecture:** $48 \rightarrow 64 \rightarrow 64 \rightarrow 1$, taking concatenated state and action vectors.

#### Exact Training Reward Formulations:
During sequential 24-hour training episodes, agents receive continuous Mean Absolute Error (MAE) reward signals:

* **Agent 1 (P1 - Local Peak-Valley Flattener):**
  $$r_{1, i} = -\frac{1}{24} \sum_{h=1}^{24} \left| a_{P1, i}(h) - y_{\text{net}, i}(h) \right|$$
  where $y_{\text{net}, i} = \text{MinMax}(P_{\text{net}, i}[t+1:t+25]) \in [0, 1]^{24}$.
* **Agent 2 (P2 - Collaborative Inter-Network Balancer):**
  $$r_{2, i} = -\frac{1}{24} \sum_{h=1}^{24} \left| a_{P2, i}(h) - y_{\text{diff\_dist}, i}(h) \right|$$
  where $y_{\text{diff\_dist}, i} = \text{MinMax}(\Delta U_{i, \text{dist}}[t+1:t+25]) \in [0, 1]^{24}$.

#### Exact Dynamic EV Tariff Equation:
The final collaborative dynamic EV charging tariff for network $i$ at hour $t$ is computed as:
$$p_{\text{EV}, i}(t) = p_{\text{conv}}(t) + 0.5 \cdot a_{P1, i}(t) + 0.5 \cdot a_{P2, i}(t)$$
where $p_{\text{conv}}(t)$ is the baseline conventional time-of-use tariff and $a_{P1, i}(t), a_{P2, i}(t) \in [0, 1]$ are the continuous action components output by the trained Actor networks.

---

## 🔄 End-to-End Execution Pipeline

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
│   • Evaluates closed-form optimal demand response functions.                     │
│   • Notebook: Stage2_EV_Demand_Distance_Renewable.ipynb                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 3: Collaborative Multi-Agent DRL Training (Canonical v2)                   │
│   • Trains 8 DDPG Actor-Critic agents from scratch with distance-weighted P2.    │
│   • Evaluates continuous dynamic pricing actions on the test window.             │
│   • Notebook: Stage3_DDPG_v2.ipynb                                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 4: Multi-Seed Benchmark Evaluation & Ablation Suite (Canonical v2)        │
│   • Evaluates Peak-to-Average Ratio (PAR), Max-Min Imbalance, and Std Imbalance. │
│   • Executes complete 4-way ablation isolating Gaps 1, 2, and 3.                 │
│   • Notebook: Stage4_Evaluation_v2.ipynb                                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Results & Benchmark Comparison

### Headline Strategy Benchmarks

Benchmarking across four distinct operational strategies (evaluated across 5 inference stochasticity seeds, $\mu_i = 0.5 \times \max(\text{EV\_Unbalanced}_i)$):

| Strategy | Strategy Description | Max-Min Imbalance (Mean $\pm$ Std) | Std Imbalance (Mean $\pm$ Std) | Peak-to-Average Ratio (PAR) |
| :--- | :--- | :---: | :---: | :---: |
| **PV** | Standalone Peak-Valley Pricing (No Balancing) | $0.272920 \pm 0.000806$ | $0.115208 \pm 0.000305$ | $1.344 \pm 0.010$ |
| **PVB_OLD** | Base Paper Pairwise Balancing ($N=2$ extension) | $0.260820 \pm 0.000109$ | $0.111630 \pm 0.000043$ | $1.335 \pm 0.001$ |
| **PVB_NEW** | $N=4$ Network Balancing (Gap 1 Resolved) | $0.254381 \pm 0.000108$ | $0.104417 \pm 0.000043$ | $1.378 \pm 0.001$ |
| **PVB_FULL** | **Proposed: $N$-Network + Distance + Renewables** | **$0.220979 \pm 0.000110$** | **$0.090078 \pm 0.000043$** | **$1.416 \pm 0.002$** |

> **Key Findings:**
> * The proposed **`PVB_FULL`** architecture reduces maximum-minimum network utilization imbalance from **0.272920 down to 0.220979 (a 19.03% reduction)** over uncoordinated PV pricing, and outperforms the base paper's naive pairwise method by **15.28%**.
> * Cross-network utilization standard deviation decreases from **0.115208 to 0.090078 (a 21.81% reduction)**.

---

### 4-Way Ablation Study

To isolate the marginal contribution of each gap, a systematic 4-way ablation was performed:

| Configuration | Features Included | Max-Min Imbalance (Mean $\pm$ Std) | Std Imbalance (Mean $\pm$ Std) | Marginal Contribution |
| :--- | :--- | :---: | :---: | :---: |
| **(a) Gap 1 Only** | $N$-Network Formulation ($N=4$) | $0.254381 \pm 0.000108$ | $0.104417 \pm 0.000043$ | Multi-network coordination baseline |
| **(b) Gap 1 + Gap 2** | $N$-Network + Spatial Distance Friction | $0.253620 \pm 0.000108$ | $0.104139 \pm 0.000043$ | $\Delta = -0.000761$ ($-0.30\%$) |
| **(c) Gap 1 + Gap 3** | $N$-Network + Renewable Net Load | $0.221623 \pm 0.000110$ | $0.090232 \pm 0.000043$ | $\Delta = -0.032758$ ($-12.88\%$) |
| **(d) PVB_FULL** | **All Three Gaps Combined (Proposed)** | **$0.220979 \pm 0.000110$** | **$0.090078 \pm 0.000043$** | **$-13.13\%$ overall reduction** |

---

### Analysis of Peak-to-Average Ratio (PAR) Dynamics

While inter-network imbalance and grid utilization spread decrease significantly under `PVB_FULL`, the Peak-to-Average Ratio increases from $1.344$ (PV) to $1.416$ (PVB_FULL). 

An empirical investigation of the 24-hour total load curves reveals the physical mechanism driving this metric:
* **Absolute Peak Load Decreases Across All Networks:** In every network, the absolute maximum demand (the numerator of PAR) drops under `PVB_FULL` compared to `PV`:
  * Residential peak: $650.6 \text{ kW} \rightarrow 648.3 \text{ kW}$ ($-0.35\%$)
  * Commercial peak: $556.8 \text{ kW} \rightarrow 545.3 \text{ kW}$ ($-2.06\%$)
  * Industrial peak: $1322.5 \text{ kW} \rightarrow 1287.2 \text{ kW}$ ($-2.67\%$)
  * Institutional peak: $787.8 \text{ kW} \rightarrow 714.5 \text{ kW}$ ($-9.30\%$)
* **Daytime Renewable Generation Depresses 24-Hour Mean Daily Load:** The integration of rooftop solar PV and wind microgrids (Gap 3) subtracts substantial energy during midday hours, which lowers the **mean 24-hour daily net load** (the denominator of PAR) by $5.7\% - 10.8\%$.
* **Ratio Effect:** Because residential and commercial residual peak hours occur during non-solar periods (e.g. evening $18:00 - 20:00$ for Residential, early morning $08:00$ for Commercial/Industrial), solar generation cannot shave these non-sunlight hours as aggressively as it reduces midday load. Because the denominator ($\text{Mean Load}$) decreases by $\sim 6\% - 11\%$ while the numerator ($\text{Peak Load}$) decreases by $\sim 0.3\% - 9\%$, the mathematical ratio $\text{PAR} = \frac{\max(P)}{\text{mean}(P)}$ increases. This is an established property of distribution grids with high distributed photovoltaic penetration.

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

## 📁 Repository Structure

```
MINI_PROJECT/
├── README.md                                    # Project documentation
├── .gitignore                                   # Git ignore rules
│
├── Stage0_N_Network_Dataset.ipynb              # Stage 0: 4-network generation & N=2 equivalence proof
├── Stage0_N_Network_Dataset_executed.ipynb     # Pre-run notebook with outputs
├── Stage0_5_Distance_Matrix.ipynb              # Stage 0.5: Tetouan coordinates & Haversine matrix
├── Stage0_5_Distance_Matrix_executed.ipynb     # Pre-run notebook with outputs
├── Stage0_75_Renewable_Profiles.ipynb          # Stage 0.75: Physics-based Solar PV & Wind modeling
├── Stage0_75_Renewable_Profiles_executed.ipynb # Pre-run notebook with outputs
├── Stage1_Load_Forecasting.ipynb               # Stage 1: XGBoost load forecasting
├── Stage1_Load_Forecasting_executed.ipynb      # Pre-run notebook with outputs
├── Stage2_EV_Demand_Distance_Renewable.ipynb   # Stage 2: EV demand with distance friction & net load
├── Stage2_EV_Demand_Distance_Renewable_executed.ipynb # Pre-run notebook with outputs
├── Stage3_DDPG_v2.ipynb                        # Stage 3: Multi-Agent DDPG training (Canonical)
├── Stage3_DDPG_v2_executed.ipynb               # Pre-run notebook with outputs
├── Stage4_Evaluation_v2.ipynb                  # Stage 4: Strategy evaluation & ablation suite (Canonical)
├── Stage4_Evaluation_v2_executed.ipynb         # Pre-run notebook with outputs
│
├── DRLDynamicPricing/                          # Cloned base paper reference repository (Lepolesa et al., 2025)
│   ├── DDPG_test_file_revised.ipynb            # Original test script
│   ├── commercial_data.xlsx                    # Raw commercial feeder readings
│   └── hourly_data.xlsx                        # Raw weather and residential load readings
│
├── data/                                       # Generated datasets, matrices & evaluation metrics
│   ├── final_evaluation_summary_canonical.csv  # Official canonical benchmark comparison table
│   ├── ablation_study_summary_canonical.csv    # Official canonical 4-way ablation results
│   ├── evaluation_metrics_seeds_canonical.csv  # Detailed 5-seed inference breakdown
│   ├── ddpg_evaluation_pricing_canonical.csv   # 24-hour dynamic tariff schedules from trained agents
│   ├── robustness_benchmark_training_seeds.csv # 4-training-seed robustness check table
│   ├── robustness_ablation_training_seeds.csv  # 4-training-seed ablation robustness table
│   ├── robustness_ddpg_pricing_training_seeds.csv # 4-training-seed DDPG pricing variance table
│   ├── network_distance_matrix.csv             # 4x4 Haversine distance matrix (km)
│   ├── network_zone_coordinates.csv            # Latitude/longitude coordinates (Tetouan)
│   ├── renewable_profiles.csv                  # 8736h Solar & Wind generation time-series
│   ├── n_network_net_load.csv                  # Full 8736h net load and util_diff_dist dataset
│   └── n_network_24h_test.csv                  # 24-hour ground truth central test dataset
│
└── models/                                     # Saved PyTorch neural network checkpoints
    ├── ddpg_residential_p1.pth                 # Actor-Critic: Residential Peak-Valley
    ├── ddpg_residential_p2.pth                 # Actor-Critic: Residential Balancer (Distance-Weighted)
    ├── ddpg_commercial_p1.pth                  # Actor-Critic: Commercial Peak-Valley
    ├── ddpg_commercial_p2.pth                  # Actor-Critic: Commercial Balancer (Distance-Weighted)
    ├── ddpg_industrial_p1.pth                  # Actor-Critic: Industrial Peak-Valley
    ├── ddpg_industrial_p2.pth                  # Actor-Critic: Industrial Balancer (Distance-Weighted)
    ├── ddpg_institutional_p1.pth               # Actor-Critic: Institutional Peak-Valley
    └── ddpg_institutional_p2.pth               # Actor-Critic: Institutional Balancer (Distance-Weighted)
```

---

## ⚙️ Installation & Quickstart

### Prerequisites
* Python 3.10 or higher
* Recommended: CPU (fully deterministic) or Apple Silicon / NVIDIA GPU

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
pip install torch torchvision torchaudio xgboost scikit-learn pandas numpy openpyxl matplotlib seaborn jupyter nbconvert
```

### 3. Run the Canonical Pipeline
Execute the canonical notebooks in sequence:
```bash
jupyter nbconvert --to notebook --execute Stage0_N_Network_Dataset.ipynb
jupyter nbconvert --to notebook --execute Stage0_5_Distance_Matrix.ipynb
jupyter nbconvert --to notebook --execute Stage0_75_Renewable_Profiles.ipynb
jupyter nbconvert --to notebook --execute Stage1_Load_Forecasting.ipynb
jupyter nbconvert --to notebook --execute Stage2_EV_Demand_Distance_Renewable.ipynb
jupyter nbconvert --to notebook --execute Stage3_DDPG_v2.ipynb
jupyter nbconvert --to notebook --execute Stage4_Evaluation_v2.ipynb
```

---

## 🔍 Reproducibility & Verification Protocol

Our verification protocol explicitly separates two distinct testing dimensions:

### 1. Inference-Time Demand Stochasticity (5 Seeds)
* **Seed Set:** `EVAL_SEEDS = [42, 101, 2024, 777, 999]`
* **Purpose:** Evaluates the impact of stochastic Gaussian noise perturbations in EV arrival and charging demand generation at inference evaluation time.
* **Scope:** Quantifies inference-level error bounds on the closed-form demand formulation.

### 2. DDPG Training-Initialization Robustness (4 Seeds)
* **Seed Set:** `MASTER_SEEDS = [42, 7, 123, 2025]`
* **Purpose:** Tests neural network weight initialization, replay buffer transition sampling order, and gradient update trajectories by retraining all 8 agents from scratch for each seed.
* **Findings:** The headline closed-form load-balancing metrics are deterministic and mathematically invariant to RL initialization by construction. The collaborative DDPG pricing policy, evaluated separately, exhibits typical RL run-to-run variability ($6.7\%–13.0\%$ relative standard deviation across training seeds) while consistently learning to peak tariffs during high-load hours and discount during midday solar availability across all seeds.

### 3. Backward Compatibility Verification
* Stage 0 includes an automated assertion verifying that setting $N=2$ produces numerical identity with the base paper pairwise formulation:
  ```python
  assert np.allclose(util_diff_n2[:, 0], util_diff_base_paper[:, 0], atol=1e-7)
  ```

---

## 📖 References & Citations

1. **Lepolesa et al. (2025):** *"Dynamic Electric Vehicle Charging Pricing for Load Balancing in Power Distribution Networks based on Collaborative DDPG Agents,"* IEEE Transactions on Smart Grid.
2. **Lillicrap, T. P., et al. (2016):** *"Continuous control with deep reinforcement learning,"* International Conference on Learning Representations (ICLR).
3. **Chen, T., & Guestrin, C. (2016):** *"XGBoost: A Scalable Tree Boosting System,"* ACM SIGKDD.
4. **Tetouan City Power Consumption Dataset:** Moroccan Smart Grid load monitoring data (UCI Machine Learning Repository / Kaggle).

---

*Academic Research Implementation — Multi-Network EV Dynamic Pricing.*
