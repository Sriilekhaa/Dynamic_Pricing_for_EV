"""Regenerate the result figures for the manuscript.

All figures are drawn from the canonical CSV outputs in ../data so that every
value visible in a figure is identical to the numbers in the tables.
Run:  python3 make_figures.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "data")
OUT = os.path.join(ROOT, "figs")
os.makedirs(OUT, exist_ok=True)


def read(name):
    with open(os.path.join(DATA, name)) as fh:
        return [dict(r) for r in csv.DictReader(fh)]


NETS = ["Residential", "Commercial", "Industrial", "Institutional"]

NETWORKS = ["residential", "commercial", "industrial", "institutional"]
MAX_CAP = {"residential": 1_166_666, "commercial": 1_000_000,
           "industrial": 1_500_000, "institutional": 800_000}
ALPHA = 0.15
EVAL_SEEDS = [42, 101, 2024, 777, 999]
STRATEGIES = ["PV", "PVB_OLD", "PVB_NEW", "PVB_FULL"]


def _gauss_base(mu, sigma, n, seed):
    """Seeded non-negative Gaussian draw (notebook CELL 4, verbatim)."""
    np.random.seed(seed)
    inc = 0
    while True:
        r = np.random.normal(mu, sigma, n)
        if all(x >= 0 for x in r):
            return r
        inc += 1
        if inc == 100000:
            return np.zeros(n)


def _eval_arrays():
    """Recompute the Stage-4 canonical 24 h total loads per strategy/seed."""
    rows = read("n_network_24h_test.csv")
    conv = {k: np.array([float(r[f"{k}_load"]) for r in rows]) for k in NETWORKS}
    net = {k: np.array([float(r[f"{k}_net_load"]) for r in rows]) for k in NETWORKS}
    mu = {k: 0.5 * max(float(r[f"{k}_ev_unbalanced_W"]) for r in rows)
          for k in NETWORKS}
    cu = {k: conv[k] / MAX_CAP[k] for k in NETWORKS}
    nu = {k: net[k] / MAX_CAP[k] for k in NETWORKS}
    with open(os.path.join(DATA, "network_distance_matrix.csv")) as fh:
        mat = list(csv.reader(fh))
    dm = np.array([[float(mat[i][j + 1]) for j in range(4)]
                   for i in range(1, 5)])
    W = np.exp(-ALPHA * dm)
    np.fill_diagonal(W, 0.0)

    def pv_load(load, m, seed):
        s = MinMaxScaler().fit_transform(load.reshape(-1, 1)).flatten()
        return m * (1 - s) + _gauss_base(0.2 * m, 0.1 * m, 24, seed)

    def pvb_load(load, m, d, seed):
        s1 = MinMaxScaler().fit_transform(load.reshape(-1, 1)).flatten()
        s2 = MinMaxScaler().fit_transform(d.reshape(-1, 1)).flatten()
        return 0.5 * m * ((1 - s1) + (1 - s2)) + _gauss_base(0.1 * m, 0.01 * m,
                                                             24, seed)

    u_old = {k: np.clip(cu[k] - cu[("commercial" if k == "residential"
                                     else "residential")], 0.0, None)
             for k in NETWORKS}
    u_new = {k: np.clip(cu[k] - np.mean([cu[j] for j in NETWORKS if j != k],
                                        axis=0), 0.0, None) for k in NETWORKS}
    u_full = {k: np.clip(nu[k] - np.average([nu[j] for j in NETWORKS],
                                            weights=W[NETWORKS.index(k)],
                                            axis=0), 0.0, None)
              for k in NETWORKS}

    def totals(strat, seed):
        tot = {}
        for k in NETWORKS:
            c_l, m = conv[k], mu[k]
            if strat == "PV":
                tot[k] = c_l + pv_load(c_l, m, seed)
            elif strat in ("PVB_OLD", "PVB_NEW"):
                d = u_old[k] if strat == "PVB_OLD" else u_new[k]
                tot[k] = c_l + pvb_load(c_l, m, d, seed)
            else:
                tot[k] = net[k] + pvb_load(net[k], m, u_full[k], seed)
        return tot

    return totals


def fig4_imbalance_envelope():
    totals = _eval_arrays()
    h = np.arange(24)
    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    styles = [("-", "0.0"), ("--", "0.35"), ("-.", "0.55"), (":", "0.75")]
    for strat, (ls, c) in zip(STRATEGIES, styles):
        envs = []
        for s in EVAL_SEEDS:
            t = totals(strat, s)
            um = np.array([t[k] / MAX_CAP[k] for k in NETWORKS])
            envs.append(np.max(um, axis=0) - np.min(um, axis=0))
        envs = np.array(envs)
        ax.plot(h, envs.mean(axis=0), color=c, ls=ls, lw=1.2, label=strat)
        ax.fill_between(h, envs.min(axis=0), envs.max(axis=0), color=c,
                        alpha=0.12)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"Imbalance envelope $\max_i u_i-\min_i u_i$")
    ax.set_xlim(0, 23)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    fig.savefig(os.path.join(OUT, "fig4_imbalance_by_strategy.png"))
    plt.close(fig)


def fig8_par_decomposition():
    totals = _eval_arrays()
    fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
    par = {"PV": [], "FULL": []}
    peak = {"PV": [], "FULL": []}
    for k in NETWORKS:
        t_pv = totals("PV", EVAL_SEEDS[0])[k]
        t_fl = totals("PVB_FULL", EVAL_SEEDS[0])[k]
        par["PV"].append(t_pv.max() / t_pv.mean())
        par["FULL"].append(t_fl.max() / t_fl.mean())
        peak["PV"].append(t_pv.max() / 1000.0)
        peak["FULL"].append(t_fl.max() / 1000.0)

    ax2 = ax.twinx()
    x = np.arange(4)
    w = 0.32
    b1 = ax2.bar(x - w / 2, peak["PV"], w, color="0.85", edgecolor="black",
                 linewidth=0.5, label="Peak PV")
    b2 = ax2.bar(x + w / 2, peak["FULL"], w, color="0.55", edgecolor="black",
                 linewidth=0.5, hatch="//", label="Peak PVB_FULL")
    for b, p in zip(b1 + b2, peak["PV"] + peak["FULL"]):
        ax2.text(b.get_x() + b.get_width() / 2, min(p - 35, 1380), f"{p:.0f}",
                 ha="center", va="center", fontsize=5.5, color="black")
    ax2.set_ylabel("Peak total load (kW)")
    ax2.set_ylim(0, 1500)
    for i, k in enumerate(NETWORKS):
        ax.text(i, par["FULL"][i] + 0.09,
                f"{par['PV'][i]:.3f}$\\rightarrow${par['FULL'][i]:.3f}",
                ha="center", va="bottom", fontsize=6.5)
    ax.plot(x, par["PV"], "ko", ms=3)
    ax.plot(x, par["FULL"], "ks", ms=3)
    ax.set_xticks(x)
    ax.set_xticklabels(NETS)
    ax.set_ylabel(r"Peak$-$to$-$average ratio $D_{\max}/D_{\rm mean}$")
    ax.set_ylim(1.0, 1.85)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h2 + h1, l2 + l1, frameon=False, ncol=2, loc="upper left",
              fontsize=6)
    fig.savefig(os.path.join(OUT, "fig8_par_decomposition.png"))
    plt.close(fig)


def fig3_net_load_profiles():
    rows = read("n_network_net_load.csv")
    h = [int(r["Hour_index"]) for r in rows]
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.8))
    for ax, net in zip(axs.ravel(), NETS):
        k = net.lower()
        conv = [float(r[f"{k}_conv_load"]) / 1000.0 for r in rows]
        netl = [float(r[f"{k}_net_load"]) / 1000.0 for r in rows]
        ax.plot(h, conv, color="0.2", label="Conventional")
        ax.plot(h, netl, color="0.55", ls="--", label="Net load")
        ax.set_title(net)
        ax.set_xlim(0, 23)
        if ax in (axs[1, 0], axs[1, 1]):
            ax.set_xlabel("Hour")
        if ax in (axs[0, 0], axs[1, 0]):
            ax.set_ylabel("Load (kW)")
        ax.grid(alpha=0.3)
    axs.ravel()[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_net_load_profiles.png"))
    plt.close(fig)


def fig5_ablation_contributions():
    rows = read("ablation_study_summary_canonical.csv")
    values = [float(r["Max_Min_Imbalance_Mean"]) for r in rows]
    short = ["(a)", "(b)", "(c)", "(d)"]
    base = values[0]
    margins = [(v - base) / base * 100.0 for v in values]
    fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
    cols = ["0.85", "0.6", "0.35", "0.15"]
    bars = ax.bar(short, values, color=cols, edgecolor="black", linewidth=0.5, width=0.6)
    for b, v, m in zip(bars, values, margins):
        ax.annotate(f"{v:.4f}\n({m:+.2f}%)", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel(r"Maximum$-$minimum imbalance $\bar{I}$")
    ax.grid(axis="y", alpha=0.35)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["(a)\nGap 1", "(b)\n+ Distance", "(c)\n+ Renewables",
                        "(d)\nPVB_FULL"], fontsize=6.5)
    fig.savefig(os.path.join(OUT, "fig5_ablation_contributions.png"))
    plt.close(fig)


def fig6_seasonal_results():
    rows = read("seasonal_eval_canonical.csv")
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    strats = ["PV", "PVB_OLD", "PVB_NEW", "PVB_FULL"]
    cases = ["Case A (Fixed Jan mu)", "Case B (Per-Window Recalculated mu)"]
    titles = [r"Case A (fixed $\mu_i$)", r"Case B (per-season $\mu_i$)"]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=True, constrained_layout=True)
    hatches = ["", "//", "xx", "."]
    grays = ["0.15", "0.45", "0.7", "0.88"]
    for ax, case, title in zip(axs, cases, titles):
        data = {s: [0.0] * 4 for s in strats}
        for r in rows:
            if r["Mu_Mode"] != case:
                continue
            for s in strats:
                data[s][seasons.index(r["Season"])] = float(r[s])
        width = 0.2
        x = range(4)
        for i, s in enumerate(strats):
            bars = ax.bar([j + (i - 1.5) * width for j in x], data[s], width,
                          color=grays[i], edgecolor="black", linewidth=0.4,
                          hatch=hatches[i], label=s)
        ax.set_xticks(list(x))
        ax.set_xticklabels(seasons)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    axs[0].set_ylabel(r"Maximum$-$minimum imbalance $\bar{I}$")
    axs[0].legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(os.path.join(OUT, "fig6_seasonal_results.png"))
    plt.close(fig)


def fig7_ddpg_prices():
    rows = read("ddpg_evaluation_pricing_canonical.csv")
    h = [int(r["Hour"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    conv = [float(r["Conv_Price"]) for r in rows]
    ax.step(h, conv, where="mid", color="0.3", ls=":", lw=1.0, label="Conventional")
    styles = [("-", "0.0"), ("--", "0.35"), ("-.", "0.55"), (":", "0.75")]
    for (net, (ls, c)) in zip(NETS, styles):
        k = net.lower()
        p = [float(r[f"{k}_ev_dynamic_price"]) for r in rows]
        ax.step(h, p, where="mid", color=c, ls=ls, lw=1.0, label=net)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Dynamic price (a.u.)")
    ax.set_xlim(0, 23)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_ddpg_prices.png"))
    plt.close(fig)


def fig9_ev_peak_uniformity():
    rows = read("ev_demand_profiles_test_24h.csv")
    ratios = []
    reduced = []
    for net in NETS:
        k = net.lower()
        unbal = max(float(r[f"{k}_ev_unbalanced_W"]) for r in rows)
        full = max(float(r[f"{k}_ev_pvb_full_W"]) for r in rows)
        ratios.append(full / unbal)
        reduced.append((1 - full / unbal) * 100.0)
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    bars = ax.bar(range(4), ratios, color="0.55", edgecolor="black", linewidth=0.5, width=0.55)
    for b, r, red in zip(bars, ratios, reduced):
        ax.annotate(f"{r:.3f}\n({red:.1f}% less)", (b.get_x() + b.get_width() / 2, r),
                    ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0.5, color="0.2", ls=":", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(NETS)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel(r"EV peak ratio  $D_{\rm PVB\_FULL}/D_{\rm unbal}$")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_ev_peak_uniformity.png"))
    plt.close(fig)


fig3_net_load_profiles()
fig4_imbalance_envelope()
fig5_ablation_contributions()
fig6_seasonal_results()
fig7_ddpg_prices()
fig8_par_decomposition()
fig9_ev_peak_uniformity()
print("wrote figures to", OUT)
print("wrote figures to", OUT)