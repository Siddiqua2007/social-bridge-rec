r"""Build full paper tables from <out>_results.csv / <out>_exp2.csv (mean +- std over seeds).
Usage:  python make_tables.py ..\results\ciao_v2
Beta for Exp 1 is chosen per (variant, seed) on VALIDATION ndcg@20, numbers are TEST."""
import sys
import pandas as pd

out = sys.argv[1]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
R = pd.read_csv(out + "_results.csv"); R2 = pd.read_csv(out + "_exp2.csv")
R = R[R.split != "info"]
cols = ["recall@20", "ndcg@20", "coverage@20", "entropy@20", "tail_share@20", "hit_coverage@20",
        "coverage@100", "entropy@100", "tail_share@100", "hit_coverage@100", "hit_entropy@100"]
cols = [c for c in cols if c in R.columns]

val = R[R.split == "val"]
best = val.loc[val.groupby(["variant", "seed"])["ndcg@20"].idxmax(), ["variant", "seed", "beta"]]
print("=== CHOSEN BETA per variant (count over seeds) ===")
print(best.groupby("variant")["beta"].value_counts().to_string())
test = R[R.split == "test"].merge(best, on=["variant", "seed", "beta"])
g = test.groupby("variant")[cols]
t1 = g.mean().round(4).astype(str) + " +- " + g.std().fillna(0).round(4).astype(str)
print("\n=== EXP 1 (test) ===\n", t1.T.to_string())
t1.to_csv(out + "_table_full.csv")

ref = R[(R.split == "test") & (R.beta == 0.0)].groupby("seed")[cols].first()
print("\n=== NO-SOCIAL reference (mean +- std) ===")
print((ref.mean().round(4).astype(str) + " +- " + ref.std().fillna(0).round(4).astype(str)).to_string())

c2 = [c for c in cols if c in R2.columns]
g2 = R2[R2.split == "test"].groupby(["kind", "beta_bridge"])[c2]
t2 = g2.mean().round(4).astype(str) + " +- " + g2.std().fillna(0).round(4).astype(str)
print("\n=== EXP 2 (test) ===\n", t2.to_string())
t2.to_csv(out + "_exp2_full.csv")
