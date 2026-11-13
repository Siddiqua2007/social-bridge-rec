"""
socrec.py  --  small CPU-only toolkit for the study
"Do preference-based social-edge pruners remove bridging weak ties, and what does that cost in diversity?"

Everything here is plain numpy / scipy / pandas / networkx (no GPU, no torch).
Recommender = ItemKNN base score + social neighbour score over a (possibly pruned) social graph.
Only the SOCIAL EDGE SELECTION changes between variants, so differences are caused by the edges.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms.community import louvain_communities


# ----------------------------------------------------------------------------- data
def read_adjlist(path):
    """LightGCN-style file: each line 'user item1 item2 ...'  ->  (user, item) pairs."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            rows += [(p[0], x) for x in p[1:]]
    return pd.DataFrame(rows, columns=["a", "b"]).drop_duplicates()


def read_pairs(path, cols=(0, 1), sep=r"\s+", skiprows=0):
    """Read two columns (ids kept as strings) from a text/csv file."""
    df = pd.read_csv(path, sep=sep, header=None, skiprows=skiprows,
                     usecols=list(cols), engine="python", dtype=str)
    df.columns = ["a", "b"]
    return df.dropna().drop_duplicates()


def build_dataset(inter_df, social_df, min_user=5, min_item=2):
    """Filter, re-index, make social graph undirected. Returns (inter[n,2], social[m,2], n_users, n_items)."""
    d = inter_df.copy()
    for _ in range(3):
        d = d[d.groupby("a")["b"].transform("size") >= min_user]
        d = d[d.groupby("b")["a"].transform("size") >= min_item]
    users = pd.Index(sorted(d["a"].unique()))
    items = pd.Index(sorted(d["b"].unique()))
    uid = {u: i for i, u in enumerate(users)}
    iid = {x: i for i, x in enumerate(items)}
    inter = np.c_[d["a"].map(uid).values, d["b"].map(iid).values].astype(np.int64)
    s = social_df[social_df["a"].isin(uid) & social_df["b"].isin(uid)]
    se = np.c_[s["a"].map(uid).values, s["b"].map(uid).values].astype(np.int64)
    se = se[se[:, 0] != se[:, 1]]
    se = np.unique(np.sort(se, axis=1), axis=0)          # undirected, no duplicates
    return inter, se, len(users), len(items)


def dataset_stats(inter, se, nu, ni):
    return dict(users=nu, items=ni, interactions=len(inter), social_edges=len(se),
                interaction_density_pct=100 * len(inter) / (nu * ni),
                social_density_pct=100 * len(se) / (nu * (nu - 1) / 2),
                avg_items_per_user=len(inter) / nu, avg_friends_per_user=2 * len(se) / nu)


def split(inter, nu, ni, seed, val_frac=0.1, test_frac=0.2, min_n=5):
    """Per-user random split. Returns train, val, test as binary csr matrices (users x items)."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(inter)), inter[:, 0]))   # group by user, random order inside
    inter = inter[order]
    _, starts, counts = np.unique(inter[:, 0], return_index=True, return_counts=True)
    role = np.zeros(len(inter), dtype=np.int8)                  # 0 train, 1 val, 2 test
    for s, c in zip(starts, counts):
        if c < min_n:
            continue
        nt = max(1, int(round(c * test_frac)))
        nv = max(1, int(round(c * val_frac)))
        role[s:s + nt] = 2
        role[s + nt:s + nt + nv] = 1

    def mk(r):
        m = role == r
        return sp.csr_matrix((np.ones(m.sum(), dtype=np.float32), (inter[m, 0], inter[m, 1])), shape=(nu, ni))
    return mk(0), mk(1), mk(2)


# ----------------------------------------------------------------------------- graph features
def constraint_array(G, nu):
    """Burt's network constraint per node (LOW = structural-hole / brokerage position). Slow on big graphs: compute once."""
    con = nx.constraint(G)
    return np.array([con.get(n, np.nan) for n in range(nu)], dtype=float)


def make_graph(se, nu):
    G = nx.Graph()
    G.add_nodes_from(range(nu))
    G.add_edges_from(map(tuple, se))
    return G


def edge_table(se, R, nu, seed=0, con_arr=None):
    """One row per undirected social edge with structural + preference features.
    R must be the TRAIN matrix only (no test leakage)."""
    G = nx.Graph()
    G.add_nodes_from(range(nu))
    G.add_edges_from(map(tuple, se))
    comm = np.zeros(nu, dtype=int)
    for k, c in enumerate(louvain_communities(G, seed=seed)):
        comm[list(c)] = k
    if con_arr is None:
        con_arr = constraint_array(G, nu)
    A = sp.csr_matrix((np.ones(2 * len(se), dtype=np.float32),
                       (np.r_[se[:, 0], se[:, 1]], np.r_[se[:, 1], se[:, 0]])), shape=(nu, nu))
    deg = np.asarray(A.sum(1)).ravel()
    u, v = se[:, 0], se[:, 1]

    def inter_cnt(M):
        return np.asarray(M[u].multiply(M[v]).sum(1)).ravel()

    def jac(M):
        i = inter_cnt(M)
        s = np.asarray(M.sum(1)).ravel()
        union = s[u] + s[v] - i
        return np.divide(i, union, out=np.zeros_like(i, dtype=float), where=union > 0)

    sR = np.asarray(R.sum(1)).ravel()
    iR = inter_cnt(R)
    nov_uv = np.divide(sR[v] - iR, sR[v], out=np.zeros(len(u)), where=sR[v] > 0)   # new items u gets from v
    nov_vu = np.divide(sR[u] - iR, sR[u], out=np.zeros(len(u)), where=sR[u] > 0)
    E = pd.DataFrame(dict(
        u=u, v=v,
        tie=jac(A),                                              # neighbour-overlap tie strength
        pref=jac(R),                                             # item-overlap (preference) similarity
        nov=(nov_uv + nov_vu) / 2,                               # novelty gain of the edge
        cross=(comm[u] != comm[v]).astype(int),                  # Louvain community crossing
        con=np.fmin(con_arr[u], con_arr[v]),
        deg_u=deg[u], deg_v=deg[v]))
    E["con"] = E["con"].fillna(1.0)
    E["hdsr"] = hdsr_edge_score(E)
    return E


def hdsr_edge_score(E):
    """HDSR-style edge weight, re-implemented from the paper's Eq. (7)-(9):
    w = mean( softmax_over_src_neighbours(item-Jaccard), softmax_over_src_neighbours(degree of dst) ).
    Directed weights are averaged to get one score per undirected edge. NOT the official code."""
    n = len(E)
    D = pd.concat([
        pd.DataFrame(dict(src=E.u.values, dst=E.v.values, pref=E.pref.values, dd=E.deg_v.values)),
        pd.DataFrame(dict(src=E.v.values, dst=E.u.values, pref=E.pref.values, dd=E.deg_u.values))],
        ignore_index=True)

    def softmax(col):
        x = D[col].astype(float)
        x = x - x.groupby(D.src).transform("max")
        e = np.exp(x)
        return e / e.groupby(D.src).transform("sum")

    w = (softmax("pref") + softmax("dd")) / 2
    return (w.values[:n] + w.values[n:]) / 2


# ----------------------------------------------------------------------------- pruning
def _pct_rank(x, rng):
    x = np.asarray(x, dtype=float) + rng.random(len(x)) * 1e-9      # random tie-break
    return pd.Series(x).rank(method="average", pct=True).values


def bridge_flag(E, kind="cross", q=0.25):
    cross = E.cross.values.astype(bool)
    lowcon = E.con.values <= np.nanquantile(E.con.values, q)
    return {"cross": cross, "lowcon": lowcon, "either": cross | lowcon, "both": cross & lowcon}[kind]


def keep_mask(E, base, keep_ratio, rng, gamma=0.0, bridge="cross"):
    """Keep the top keep_ratio fraction of edges. Same budget for every variant.
    base   : 'full' | 'random' | 'pref' | 'tie' | 'hdsr'   (what the baseline pruner ranks edges by)
    gamma  : >0 adds bridging protection:  score = pct_rank(base) + gamma * bridging_flag   (ours)"""
    n = len(E)
    if base == "full":
        return np.ones(n, dtype=bool)
    b = {"random": rng.random(n), "pref": E.pref.values, "tie": E.tie.values, "hdsr": E.hdsr.values}[base]
    score = _pct_rank(b, rng)
    if gamma > 0:
        score = score + gamma * bridge_flag(E, bridge)
    k = int(np.ceil(keep_ratio * n))
    keep = np.zeros(n, dtype=bool)
    keep[np.argsort(-score, kind="stable")[:k]] = True
    return keep


def social_matrix(E, keep, nu):
    u, v = E.u.values[keep], E.v.values[keep]
    A = sp.csr_matrix((np.ones(2 * len(u), dtype=np.float32), (np.r_[u, v], np.r_[v, u])), shape=(nu, nu))
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    return (sp.diags(1.0 / d) @ A).tocsr().astype(np.float32)


# ----------------------------------------------------------------------------- recommender
def item_knn(R, k=50):
    """Item-item cosine similarity, top-k neighbours per item."""
    R = R.tocsr().astype(np.float32)
    S = (R.T @ R).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    cnt = np.asarray(R.sum(0)).ravel()
    inv = sp.diags(1.0 / np.sqrt(np.maximum(cnt, 1)))
    S = (inv @ S @ inv).tocsr()
    rows, cols, vals = [], [], []
    for i in range(S.shape[0]):
        a, b = S.indptr[i], S.indptr[i + 1]
        if b == a:
            continue
        d, c = S.data[a:b], S.indices[a:b]
        if b - a > k:
            idx = np.argpartition(-d, k - 1)[:k]
            d, c = d[idx], c[idx]
        rows.append(np.full(len(c), i)); cols.append(c); vals.append(d)
    if not rows:
        return sp.csr_matrix(S.shape, dtype=np.float32)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=S.shape, dtype=np.float32)


def _rownorm(M):
    m = M.max(1, keepdims=True)
    return M / np.maximum(m, 1e-12)


def evaluate_many(R, S, Ws, configs, mask, truth, Ks=(20, 100), batch=128, head_frac=0.2):
    """Evaluate many scoring configs in ONE pass over users (base scores computed once per batch).
    score = base_w * norm(ItemKNN)  +  sum_k w_k * norm(share of friends in Ws[name_k] who consumed the item)
    Ws      : dict name -> row-normalised social matrix
    configs : dict label -> (base_w, [(W_name, w), ...])
    Returns dict[label] -> metrics."""
    nu, ni = R.shape
    Kmax = max(Ks)
    users = np.where(np.asarray(truth.sum(1)).ravel() > 0)[0]
    pop = np.asarray(R.sum(0)).ravel()
    head = np.zeros(ni, dtype=bool)
    head[np.argsort(-pop)[:int(head_frac * ni)]] = True
    disc = 1.0 / np.log2(np.arange(2, Kmax + 2))
    need = sorted({n for _, parts in configs.values() for n, _ in parts})
    acc = {c: dict(rec={K: 0.0 for K in Ks}, ndcg={K: 0.0 for K in Ks},
                   cnt={K: np.zeros(ni) for K in Ks}, hcnt={K: np.zeros(ni) for K in Ks}) for c in configs}
    for s0 in range(0, len(users), batch):
        B = users[s0:s0 + batch]
        base = _rownorm((R[B] @ S).toarray())
        mb = mask[B].tocoo()
        T = truth[B].tocsr()
        ntruth = np.asarray(T.sum(1)).ravel()
        soc = {n: _rownorm((Ws[n][B] @ R).toarray()) for n in need}
        for c, (bw, parts) in configs.items():
            sc = bw * base
            for n, w in parts:
                if w:
                    sc = sc + w * soc[n]
            sc[mb.row, mb.col] = -np.inf
            top = np.argpartition(-sc, Kmax - 1, axis=1)[:, :Kmax]
            ts = np.take_along_axis(sc, top, 1)
            top = np.take_along_axis(top, np.argsort(-ts, axis=1), 1)
            hit = np.asarray(T[np.repeat(np.arange(len(B)), Kmax), top.ravel()]).reshape(len(B), Kmax)
            for K in Ks:
                h = hit[:, :K]
                acc[c]["rec"][K] += (h.sum(1) / ntruth).sum()
                ideal = np.cumsum(disc[:K])[np.minimum(ntruth, K).astype(int) - 1]
                acc[c]["ndcg"][K] += ((h * disc[:K]).sum(1) / ideal).sum()
                acc[c]["cnt"][K] += np.bincount(top[:, :K].ravel(), minlength=ni)
                acc[c]["hcnt"][K] += np.bincount(top[:, :K][h.astype(bool)], minlength=ni)   # only CORRECT recommendations
    out = {}
    for c, a in acc.items():
        m = {}
        for K in Ks:
            cnt = a["cnt"][K]
            p = cnt / max(cnt.sum(), 1)
            nz = p[p > 0]
            m[f"recall@{K}"] = a["rec"][K] / len(users)
            m[f"ndcg@{K}"] = a["ndcg"][K] / len(users)
            m[f"coverage@{K}"] = (cnt > 0).mean()
            m[f"entropy@{K}"] = float(-(nz * np.log(nz)).sum())
            m[f"tail_share@{K}"] = 1.0 - cnt[head].sum() / max(cnt.sum(), 1)
            hc = a["hcnt"][K]                                   # hit-only diversity (Li et al., CIKM 2024 concern)
            hp = hc / max(hc.sum(), 1); hz = hp[hp > 0]
            m[f"hit_coverage@{K}"] = (hc > 0).mean()
            m[f"hit_entropy@{K}"] = float(-(hz * np.log(hz)).sum())
            m[f"hit_tail_share@{K}"] = 1.0 - hc[head].sum() / max(hc.sum(), 1)
        out[c] = m
    return out


# ----------------------------------------------------------------------------- diagnostics (Step 1)
def diagnose(E, frac=0.30, n_null=500, seed=0):
    """Do preference-weak edges (what a preference-based pruner removes first) bridge communities more than chance?"""
    rng = np.random.default_rng(seed)
    cross = E.cross.values
    out = dict(n_edges=len(E), overall_cross=cross.mean(),
               novelty_cross=E.nov[E.cross == 1].mean(), novelty_within=E.nov[E.cross == 0].mean())
    z = E.pref.values == 0
    out["zero_pref_share"] = z.mean()
    out["cross_in_zero_pref"] = cross[z].mean() if z.any() else np.nan
    out["cross_in_pos_pref"] = cross[~z].mean() if (~z).any() else np.nan
    for col in ["pref", "hdsr"]:
        r = _pct_rank(E[col].values, rng)
        low = r <= frac
        null = [cross[rng.choice(len(E), low.sum(), replace=False)].mean() for _ in range(n_null)]
        out[f"cross_in_bottom{int(frac*100)}_{col}"] = cross[low].mean()
        out[f"null_lo_{col}"] = np.percentile(null, 2.5)
        out[f"null_hi_{col}"] = np.percentile(null, 97.5)
    return out
