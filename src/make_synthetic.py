"""Make a small synthetic dataset (planted communities) in the same plain-text format as Ciao.
Use ONLY to check that your setup runs. Never report results on it."""
import numpy as np, sys

def main(out_prefix="data/syn", nu=2000, ni=4000, ncom=8, seed=0):
    rng = np.random.default_rng(seed)
    ucom = rng.integers(0, ncom, nu)
    icom = rng.integers(0, ncom, ni)
    pop = 1.0 / (np.arange(1, ni + 1) ** 0.8)
    pop = pop[rng.permutation(ni)]
    inter = set()
    for u in range(nu):
        n = int(np.clip(rng.geometric(0.05), 6, 80))
        w = pop * np.where(icom == ucom[u], 6.0, 1.0)         # 70-80% own-community items
        w = w / w.sum()
        for i in rng.choice(ni, size=n, replace=False, p=w):
            inter.add((u, i))
    soc = set()
    for u in range(nu):
        for _ in range(rng.poisson(4)):
            if rng.random() < 0.9:
                cand = np.where(ucom == ucom[u])[0]
            else:
                cand = np.arange(nu)
            v = int(rng.choice(cand))
            if v != u:
                soc.add((min(u, v), max(u, v)))
    with open(out_prefix + "_inter.txt", "w") as f:
        for u, i in sorted(inter): f.write(f"u{u} i{i} 5\n")
    with open(out_prefix + "_social.txt", "w") as f:
        for u, v in sorted(soc): f.write(f"u{u} u{v}\n")
    print("wrote", len(inter), "interactions,", len(soc), "social edges")

if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["data/syn"]))
