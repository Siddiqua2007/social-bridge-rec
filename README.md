# social-bridge-rec
 
Structure-aware social recommendation: do bridging weak ties (structural holes) help diversity?
Course project for *Social Networking & Mining*.
 
We run a **controlled study**: one simple recommender (ItemKNN + friends' votes) is kept fixed, and only
the *social-edge selection* changes. This isolates the effect of which social ties are kept.
It is **not** a state-of-the-art model and makes no SOTA claim.
 
## Research questions
 
1. **RQ1.** Do preference-based pruners (item-Jaccard, HDSR-style) remove community-bridging ties more often than chance?
2. **RQ2.** At the same edge budget, does protecting bridging ties improve diversity (entropy, tail share, coverage) at acceptable accuracy cost?
3. **RQ3.** Does a dedicated bridging-tie channel add diversity beyond simply giving the same total social weight to strong ties?
A *bridging* tie is defined in two ways: crossing Louvain communities (`--bridge cross`, default) or touching a node with low Burt constraint (`--bridge lowcon`).
 
## What is implemented
 
| Edge selector | Idea |
|---|---|
| `full` | keep all social edges |
| `random` | keep a random subset |
| `pref` | keep edges with highest item-Jaccard similarity of train interactions |
| `hdsr` | HDSR-style edge weight, our re-implementation of Eq. 7-9 of Hierarchical Denoising (not the official code; we prune deterministically, the paper drops edges stochastically) |
| `pref+bridge`, `hdsr+bridge` | same, plus a bonus for bridging edges (`--gamma`) |
 
All selectors keep the same number of edges (`--keep`, default 0.7).
Social weight `beta` is chosen per variant and seed on the validation split (NDCG@20); numbers are reported on the test split.
 
Metrics: Recall@K, NDCG@K, Coverage@K, Entropy@K, tail share@K (share of recommendation slots outside the 20% most popular training items), and hit-only versions (`hit_coverage`, `hit_entropy`, `hit_tail_share`) computed only over correct recommendations. Coverage@100 saturates (about 0.94 for every method), so we report @20 and entropy/tail share.
 
## Repository layout
 
```
src/
  socrec.py           data loading, edge features, selectors, recommender, metrics
  run_experiments.py  Step 1 diagnostic, Exp 1, Exp 2
  make_tables.py      mean +- std tables from the result CSVs
  plots.py            three figures
  check_ties.py       sanity check: enough positive-score candidates per user?
  peek.py             look at a raw data file and guess its format
  make_synthetic.py   fake data, only for smoke tests (never for results)
results/              CSVs, tables, figures, terminal logs (prefix = run name)
docs/                 related-work table and reference list
```
 
## Setup
 
Python 3.13 was used. CPU only (numpy, scipy, pandas, networkx, matplotlib).
 
```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```
 
## Data
 
`data/` and `external/` are git-ignored. We use the **Ciao** files from the DivSR repository
(<https://github.com/ll0ruc/DivSR>, `data/ciao/ciao.train.rating` and `ciao.links`).
Only the DivSR *train* file is used as our interaction pool; we make our own per-user random split
(70/10/20 train/val/test). Ratings are treated as binary interactions. Users with fewer than 5
interactions and items with fewer than 2 are filtered, and the social graph is made undirected.
Statistics are written to `results/<run>_stats.csv`.
 
Original dataset reference: Tang, Gao, Liu, "mTrust: Discerning multi-faceted trust in a connected world", WSDM 2012.
 
## Reproduce
 
Smoke test on synthetic data (code check only):
 
```
cd src
python make_synthetic.py ..\data\syn
python run_experiments.py --inter ..\data\syn_inter.txt --social ..\data\syn_social.txt --seeds 1 --out ..\results\syn
```
 
Real run (Louvain-crossing bridges):
 
```
cd src
python run_experiments.py --inter ..\external\DivSR\data\ciao\ciao.train.rating --social ..\external\DivSR\data\ciao\ciao.links --seeds 3 --betas 0 0.1 0.2 0.3 0.5 0.7 0.9 --out ..\results\ciao_v2 > ..\results\ciao_v2_terminal_output.txt
python plots.py ..\results\ciao_v2
python make_tables.py ..\results\ciao_v2 > ..\results\ciao_v2_tables.txt
```
 
Burt-constraint bridges: add `--bridge lowcon` and use `--out ..\results\ciao_lowcon`.
For other datasets use `peek.py` first, then `--format adjlist`, `--sep ","`, `--skiprows`, `--inter-cols`, `--social-cols` as needed.
 
## Results on Ciao (3 seeds, mean +- std over seeds)
 
Full tables: `results/ciao_v2_tables.txt`, `results/ciao_lowcon_tables.txt`. Figures: `results/ciao_*_fig*.png`.
 
- **RQ1.** HDSR-style pruning removes bridging ties more often than chance (about 0.37 of its lowest-weight 30% edges cross communities, random 0.27-0.28). Preference-Jaccard pruning does not (76% of edges have zero overlap, so its ranking is mostly random tie-breaking).
- **Social signal.** Recall@20 rises from 0.0156 (no social) to 0.0481 (full graph), but Entropy@20 falls from 8.87 to 7.64 and tail share@20 from 0.95 to 0.27. Coverage@20 does not fall (0.665 to 0.677).
- **RQ2.** Pruning of any kind gives back some entropy and tail share at some recall cost. Adding bridge protection does **not** improve diversity, with either bridge definition, and sometimes lowers it (for example hdsr 7.84 vs hdsr+bridge 7.73 Entropy@20 with Louvain bridges).
- **RQ3.** At equal total social weight the bridging channel gives a higher tail share than the strong-only control (for example 0.49 vs 0.40 at weight 0.3) but lower recall and lower hit-only diversity, so the extra tail exposure comes mostly from incorrect recommendations.
These are small differences from 3 seeds on one dataset; differences within one std should be read as no difference.
Second dataset: TBD (`results/yelp_lakshmi_*`).
 
## Limitations
 
ItemKNN recommender (no GNN), undirected social graph, random (not temporal) split, only the DivSR train file as interaction pool,
aggregate item-level diversity (no per-user ILD, since no item features), Louvain is randomised (hence seeds), HDSR is a re-implementation,
DBA-SR and DivSR baselines were not run, 3 seeds and no significance tests.
 
## Main related work
 
DivSR (Li & Zhou, DASFAA 2025), Hierarchical Denoising for Robust Social Recommendation (Hu et al., IEEE TKDE 2024), DBA-SR (Zu et al., IEEE TCSS 2026),
Enhancing Structural Diversity in Social Networks by Recommending Weak Ties (Sanz-Cruzado & Castells, RecSys 2018),
On Evaluation Metrics for Diversity-enhanced Recommendations (Li et al., CIKM 2024), DeepHole (IEEE TCSS 2025).
See `docs/` for the full table and references.
 
## Team
 
Siddiqua, Lakshmi, and Jyoti.