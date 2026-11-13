"""Make the 3 paper figures from the CSVs.   python plots.py ..\\results\\ciao_v2
Needs the v2 results (hit-only metrics); coverage@100 is NOT plotted because it saturates (~0.94-0.95 for every method)."""
import sys, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[1]
R = pd.read_csv(out + "_results.csv"); R2 = pd.read_csv(out + "_exp2.csv")

# ---- Fig 1: which edges survive pruning
info = R[R.split == "info"].groupby("variant")["kept_cross_share"].mean()
fig, ax = plt.subplots(figsize=(6.5, 3.8)); info.plot.bar(ax=ax, color="tab:blue")
ax.axhline(info["full"], color="gray", ls="--", lw=1, label="full graph")
ax.set_ylabel("share of kept edges\nthat bridge communities"); ax.set_xlabel("")
ax.set_title("Which edges survive pruning?"); ax.legend(fontsize=8)
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(out + "_fig1_kept_bridges.png", dpi=200); plt.close()

# ---- Fig 2: accuracy-diversity trade-off (each point = one social weight beta, mean over seeds)
M = ["entropy@20", "tail_share@20", "hit_entropy@100"]
T = R[R.split == "test"].groupby(["variant", "beta"])[["recall@20"] + M].mean().reset_index()
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for v, g in T.groupby("variant"):
    for ax, m in zip(axs, M):
        ax.plot(g["recall@20"], g[m], marker="o", ms=4, label=v)
for ax, m in zip(axs, M):
    ax.set_xlabel("Recall@20 (accuracy)"); ax.set_ylabel(m); ax.grid(alpha=.3)
axs[0].legend(fontsize=7)
fig.suptitle("Accuracy-diversity trade-off (each point = one social weight beta)")
plt.tight_layout(); plt.savefig(out + "_fig2_tradeoff.png", dpi=200); plt.close()

# ---- Fig 3: bridging channel vs strong-only control (mean +- std over seeds)
M3 = ["recall@20", "entropy@20", "tail_share@20", "hit_entropy@100"]
g3 = R2[R2.split == "test"].groupby(["kind", "beta_bridge"])[M3]
mu, sd = g3.mean().reset_index(), g3.std().fillna(0).reset_index()
fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
for ax, m in zip(axs, M3):
    for k in mu.kind.unique():
        a, b = mu[mu.kind == k], sd[sd.kind == k]
        ax.errorbar(a["beta_bridge"], a[m], yerr=b[m], marker="o", ms=4, capsize=2, label=k)
    ax.set_xlabel("weight moved to bridging ties"); ax.set_ylabel(m); ax.grid(alpha=.3)
axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(out + "_fig3_bridge_channel.png", dpi=200); plt.close()
print("saved 3 figures next to", out)
