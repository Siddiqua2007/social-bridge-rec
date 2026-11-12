"""
Run the whole study (all numbers for the paper come from here).

Smoke test on fake data (do this first, ~10 seconds):
    python make_synthetic.py data/syn
    python run_experiments.py --inter data/syn_inter.txt --social data/syn_social.txt --seeds 2 --out results/syn

Real data (adapt the column numbers / --format to YOUR files; use peek.py to look at them first):
    python run_experiments.py --inter data/ciao/ratings.txt --inter-cols 0 1 --social data/ciao/trust.txt \
        --seeds 3 --keep 0.7 --out results/ciao

Outputs (in results/):  <out>_stats.csv  <out>_diagnostics.csv  <out>_results.csv  <out>_exp2.csv  <out>_main_table.csv
"""
import argparse, os, time
import numpy as np, pandas as pd
import socrec as sr


def load(a):
    if a.format == "adjlist":
        inter_df = sr.read_adjlist(a.inter)
    else:
        inter_df = sr.read_pairs(a.inter, a.inter_cols, a.sep, a.skiprows)
    soc_df = sr.read_pairs(a.social, a.social_cols, a.sep, a.skiprows)
    return sr.build_dataset(inter_df, soc_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", required=True)
    ap.add_argument("--social", required=True)
    ap.add_argument("--format", default="pairs", choices=["pairs", "adjlist"], help="interaction file layout")
    ap.add_argument("--inter-cols", nargs=2, type=int, default=[0, 1])
    ap.add_argument("--social-cols", nargs=2, type=int, default=[0, 1])
    ap.add_argument("--sep", default=r"\s+")
    ap.add_argument("--skiprows", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--keep", type=float, default=0.7, help="fraction of social edges each pruner keeps")
    ap.add_argument("--gamma", type=float, default=0.5, help="bridge-protection strength (ours)")
    ap.add_argument("--bridge", default="cross", choices=["cross", "lowcon", "either", "both"])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--beta-strong", type=float, default=0.2, help="Exp2: fixed weight of the strong-tie channel")
    ap.add_argument("--beta-bridge", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--out", default="results/run")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    inter, se, nu, ni = load(a)
    st = sr.dataset_stats(inter, se, nu, ni)
    print("DATASET STATS:", {k: round(float(v), 4) for k, v in st.items()})
    pd.Series(st).to_csv(a.out + "_stats.csv")

    G = sr.make_graph(se, nu)
    t = time.time(); con_arr = sr.constraint_array(G, nu); print(f"constraint computed in {time.time()-t:.0f}s")

    variants = {                       # name -> (what the baseline pruner ranks by, bridge-protection gamma)
        "full":        ("full", 0.0),
        "random":      ("random", 0.0),
        "pref":        ("pref", 0.0),
        "hdsr":        ("hdsr", 0.0),
        "pref+bridge": ("pref", a.gamma),
        "hdsr+bridge": ("hdsr", a.gamma),
    }
    diag, res, res2 = [], [], []
    for seed in range(a.seeds):
        t0 = time.time()
        Rtr, Rva, Rte = sr.split(inter, nu, ni, seed)
        E = sr.edge_table(se, Rtr, nu, seed=seed, con_arr=con_arr)       # features from TRAIN data only
        d = sr.diagnose(E, seed=seed); d["seed"] = seed; diag.append(d)
        rng = np.random.default_rng(seed)
        S = sr.item_knn(Rtr)

        # ---------------- Experiment 1: same edge budget, different edge selectors ----------------
        Ws, cfg1 = {}, {}
        for name, (base, g) in variants.items():
            keep = sr.keep_mask(E, base, a.keep, rng, gamma=g, bridge=a.bridge)
            Ws[name] = sr.social_matrix(E, keep, nu)
            res.append(dict(seed=seed, variant=name, split="info", beta=np.nan, kept_edges=int(keep.sum()),
                            kept_cross_share=float(E.cross.values[keep].mean()), kept_nov=float(E.nov.values[keep].mean())))
            for b in a.betas:
                cfg1[(name, b)] = (1 - b, [(name, b)])
        for split_name, mask, truth in [("val", Rtr, Rva), ("test", Rtr + Rva, Rte)]:
            out = sr.evaluate_many(Rtr, S, Ws, cfg1, mask, truth)
            for (name, b), m in out.items():
                res.append(dict(seed=seed, variant=name, split=split_name, beta=b, **m))

        # ---------------- Experiment 2: does the BRIDGING channel add diversity? ----------------
        flag = sr.bridge_flag(E, a.bridge)
        W2 = {"strong": sr.social_matrix(E, ~flag, nu), "bridge": sr.social_matrix(E, flag, nu)}
        cfg2 = {}
        bs = a.beta_strong
        for bb in a.beta_bridge:
            cfg2[("bridge_sweep", bb)] = (1 - bs - bb, [("strong", bs), ("bridge", bb)])
            cfg2[("strong_control", bb)] = (1 - bs - bb, [("strong", bs + bb)])   # same TOTAL social weight, no bridging
        for split_name, mask, truth in [("val", Rtr, Rva), ("test", Rtr + Rva, Rte)]:
            out = sr.evaluate_many(Rtr, S, W2, cfg2, mask, truth)
            for (kind, bb), m in out.items():
                res2.append(dict(seed=seed, kind=kind, beta_bridge=bb, beta_strong=bs, split=split_name, **m))
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(diag).to_csv(a.out + "_diagnostics.csv", index=False)
        pd.DataFrame(res).to_csv(a.out + "_results.csv", index=False)
        pd.DataFrame(res2).to_csv(a.out + "_exp2.csv", index=False)
    summarize(pd.DataFrame(diag), pd.DataFrame(res), pd.DataFrame(res2), a.out)


def summarize(D, R, R2, out):
    pd.set_option("display.width", 200)
    print("\n=== STEP 1 DIAGNOSTIC (mean over seeds) ===")
    print(D.drop(columns=["seed"]).mean().round(4).to_string())
    info = R[R.split == "info"].groupby("variant")[["kept_edges", "kept_cross_share", "kept_nov"]].mean()
    print("\n=== KEPT-EDGE PROFILE ===\n", info.round(3))
    R = R[R.split != "info"]
    val = R[R.split == "val"]
    best = val.loc[val.groupby(["variant", "seed"])["ndcg@20"].idxmax(), ["variant", "seed", "beta"]]
    print("\n=== CHOSEN BETA (on validation) ===\n", best.groupby("variant")["beta"].value_counts().to_string())
    test = R[R.split == "test"].merge(best, on=["variant", "seed", "beta"])
    cols = ["recall@20", "ndcg@20", "coverage@100", "entropy@100", "tail_share@100"]
    g = test.groupby("variant")[cols]
    tab = g.mean().round(4).astype(str) + " +- " + g.std().fillna(0).round(4).astype(str)
    print("\n=== EXP 1 MAIN TABLE (test; beta picked on validation; mean +- std over seeds) ===\n", tab.to_string())
    tab.to_csv(out + "_main_table.csv")
    ref = R[(R.split == "test") & (R.beta == 0.0)].groupby("seed")[cols].first().mean().round(4)
    print("\nNO-SOCIAL reference (beta=0, ItemKNN only):\n", ref.to_string())
    t2 = R2[R2.split == "test"].groupby(["kind", "beta_bridge"])[cols].mean().round(4)
    print("\n=== EXP 2: bridge channel vs same-total-weight strong-only control (test, mean over seeds) ===\n", t2.to_string())
    t2.to_csv(out + "_exp2_table.csv")


if __name__ == "__main__":
    main()
