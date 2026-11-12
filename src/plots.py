"""Make the 3 paper figures from the CSVs.   python plots.py results/ciao"""
import sys, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[1]
R = pd.read_csv(out + "_results.csv"); R2 = pd.read_csv(out + "_exp2.csv")

info = R[R.split == "info"].groupby("variant")["kept_cross_share"].mean()
fig, ax = plt.subplots(figsize=(6, 3.5)); info.plot.bar(ax=ax, color="tab:blue")
ax.set_ylabel("share of kept edges that cross communities"); ax.set_title("Which edges survive pruning?")
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(out + "_fig1_kept_bridges.png", dpi=200); plt.close()

T = R[R.split == "test"].groupby(["variant", "beta"])[["recall@20", "coverage@100", "entropy@100"]].mean().reset_index()
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for v, g in T.groupby("variant"):
    axs[0].plot(g["recall@20"], g["coverage@100"], marker="o", label=v)
    axs[1].plot(g["recall@20"], g["entropy@100"], marker="o", label=v)
for ax, y in zip(axs, ["coverage@100", "entropy@100"]):
    ax.set_xlabel("Recall@20 (accuracy)"); ax.set_ylabel(y); ax.grid(alpha=.3)
axs[0].legend(fontsize=7); fig.suptitle("Accuracy-diversity trade-off (each point = one social weight beta)")
plt.tight_layout(); plt.savefig(out + "_fig2_tradeoff.png", dpi=200); plt.close()

T2 = R2[R2.split == "test"].groupby(["kind", "beta_bridge"])[["recall@20", "coverage@100", "entropy@100", "tail_share@100"]].mean().reset_index()
fig, axs = plt.subplots(1, 4, figsize=(14, 3.3))
for ax, m in zip(axs, ["recall@20", "coverage@100", "entropy@100", "tail_share@100"]):
    for k, g in T2.groupby("kind"): ax.plot(g["beta_bridge"], g[m], marker="o", label=k)
    ax.set_xlabel("weight moved to bridging ties"); ax.set_ylabel(m); ax.grid(alpha=.3)
axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(out + "_fig3_bridge_channel.png", dpi=200); plt.close()
print("saved 3 figures next to", out)
