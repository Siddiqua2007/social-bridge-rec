"""Real-data check: do users have enough positive-score candidates for top-100? (ties would inflate coverage)"""
import argparse, numpy as np
import socrec as sr
from run_experiments import load

ap = argparse.ArgumentParser()
ap.add_argument("--inter", required=True); ap.add_argument("--social", required=True)
ap.add_argument("--format", default="pairs", choices=["pairs", "adjlist"])
ap.add_argument("--inter-cols", nargs=2, type=int, default=[0, 1])
ap.add_argument("--social-cols", nargs=2, type=int, default=[0, 1])
ap.add_argument("--sep", default=r"\s+"); ap.add_argument("--skiprows", type=int, default=0)
a = ap.parse_args()

inter, se, nu, ni = load(a)
Rtr, Rva, Rte = sr.split(inter, nu, ni, 0)
S = sr.item_knn(Rtr)
users = np.where(np.asarray(Rte.sum(1)).ravel() > 0)[0]
users = np.random.default_rng(0).choice(users, min(500, len(users)), replace=False)
base = (Rtr[users] @ S).toarray()
base[Rtr[users].nonzero()] = -np.inf
npos = (base > 0).sum(1)
print("users checked:", len(users))
print("share of users with < 100 positive-score candidates: %.3f" % (npos < 100).mean())
print("median positive-score candidates per user:", int(np.median(npos)))