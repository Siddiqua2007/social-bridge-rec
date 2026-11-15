"""Publication-style figures (IEEE double-column) for the paper.

Usage (from src/):
    python paper_figs.py ..\\results\\ciao_v2 ..\\results\\lastfm_lakshmi_v2 ..\\results\\paper

Reads <prefix>_results.csv and <prefix>_exp2.csv of both datasets (same files plots.py uses)
and writes  paper_fig1_kept_bridges / paper_fig2_tradeoff / paper_fig3_bridge_channel  as .pdf (vector) and .png (300 dpi).
Each figure has Ciao in the top row and Last.fm in the bottom row, so one \\includegraphics per figure is enough.
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ciao, lfm, out = sys.argv[1], sys.argv[2], sys.argv[3]
DATA = [("Ciao", ciao), ("Last.fm", lfm)]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

NAME = {"full": "Full graph", "random": "Random", "pref": "Pref", "hdsr": "HDSR-style",
        "pref+bridge": "Pref+bridge", "hdsr+bridge": "HDSR+bridge"}
ORDER = ["full", "random", "pref", "hdsr", "pref+bridge", "hdsr+bridge"]
COL = {"full": "#000000", "random": "#7f7f7f", "pref": "#1f77b4", "hdsr": "#d95f02",
       "pref+bridge": "#2ca02c", "hdsr+bridge": "#9467bd"}
MRK = {"full": "o", "random": "s", "pref": "^", "hdsr": "v", "pref+bridge": "D", "hdsr+bridge": "P"}
LS = {"full": "-", "random": "--", "pref": "-", "hdsr": "-", "pref+bridge": "--", "hdsr+bridge": "--"}
XW = 7.1  # IEEE double-column width in inches


def load(prefix):
    return pd.read_csv(prefix + "_results.csv"), pd.read_csv(prefix + "_exp2.csv")


def save(fig, name):
    fig.savefig(f"{out}_{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}_{name}.png", bbox_inches="tight")
    plt.close(fig)


def tag(ax, text):
    ax.set_title(text, loc="left", fontsize=8, fontweight="bold", pad=3)


# ---------------- Fig 1: share of kept edges that cross communities ----------------
fig, axs = plt.subplots(1, 2, figsize=(XW, 2.1))
for ax, (ds, pre) in zip(axs, DATA):
    R, _ = load(pre)
    info = R[R.split == "info"].groupby("variant")["kept_cross_share"].mean()
    vs = [v for v in ORDER if v in info.index]
    ax.bar(range(len(vs)), [info[v] for v in vs], color=[COL[v] for v in vs], width=0.62, edgecolor="black", linewidth=0.4)
    ax.axhline(info["full"], color="black", ls="--", lw=0.8, label="Full graph")
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels([NAME[v] for v in vs], rotation=25, ha="right")
    ax.set_ylabel("Crossing share of kept edges")
    ax.set_ylim(0, max(info[vs].max(), info["full"]) * 1.3)
    ax.set_title(ds, fontsize=8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
axs[0].legend(loc="upper left", frameon=False)
fig.tight_layout()
save(fig, "paper_fig1_kept_bridges")

# ---------------- Fig 2: accuracy-diversity trade-off ----------------
M = [("entropy@20", "Entropy@20"), ("tail_share@20", "Tail share@20"), ("hit_entropy@100", "Hit-entropy@100")]
fig, axs = plt.subplots(2, 3, figsize=(XW, 4.3))
for r, (ds, pre) in enumerate(DATA):
    R, _ = load(pre)
    T = R[R.split == "test"].groupby(["variant", "beta"])[["recall@20"] + [m for m, _ in M]].mean().reset_index()
    for c, (m, lab) in enumerate(M):
        ax = axs[r, c]
        for v in ORDER:
            g = T[T.variant == v].sort_values("beta")
            if len(g):
                ax.plot(g["recall@20"], g[m], color=COL[v], marker=MRK[v], ls=LS[v], label=NAME[v])
        ax.set_xlabel("Recall@20")
        ax.set_ylabel(lab)
        ax.grid(True)
        tag(ax, f"({'abc'[c]}) {ds}" if r == 0 else f"({'def'[c]}) {ds}")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.97))
save(fig, "paper_fig2_tradeoff")

# ---------------- Fig 3: bridging channel vs strong-only control ----------------
M3 = [("recall@20", "Recall@20"), ("entropy@20", "Entropy@20"),
      ("tail_share@20", "Tail share@20"), ("hit_entropy@100", "Hit-entropy@100")]
KIND = {"bridge_sweep": ("Bridging channel", "#d95f02", "o", "-"),
        "strong_control": ("Strong-only control", "#1f77b4", "s", "--")}
fig, axs = plt.subplots(2, 4, figsize=(XW, 3.9))
for r, (ds, pre) in enumerate(DATA):
    _, R2 = load(pre)
    g = R2[R2.split == "test"].groupby(["kind", "beta_bridge"])[[m for m, _ in M3]]
    mu, sd = g.mean().reset_index(), g.std().fillna(0).reset_index()
    for c, (m, lab) in enumerate(M3):
        ax = axs[r, c]
        for k, (nm, col, mk, ls) in KIND.items():
            a, b = mu[mu.kind == k], sd[sd.kind == k]
            ax.errorbar(a["beta_bridge"], a[m], yerr=b[m], color=col, marker=mk, ls=ls, capsize=1.5, elinewidth=0.6, label=nm)
        ax.set_xlabel(r"Weight on bridging ties $\beta_b$")
        ax.set_ylabel(lab)
        ax.grid(True)
        tag(ax, f"({'abcd'[c]}) {ds}" if r == 0 else f"({'efgh'[c]}) {ds}")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "paper_fig3_bridge_channel")
print("saved paper_fig1/2/3 (.pdf + .png) with prefix", out)
