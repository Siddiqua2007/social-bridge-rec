# Datasets

---

## 1. LastFM dataset (HetRec 2011)

### Source
- File: `hetrec2011-lastfm-2k.zip`
- Downloaded from: https://grouplens.org
- Files used: `user_artists.dat` (user, artist, weight) and `user_friends.dat` (userID, friendID)

### Original dataset paper
When using this dataset, cite the Last.fm website (http://lastfm.com) and the HetRec '11 workshop:

> Iván Cantador, Peter Brusilovsky, and Tsvi Kuflik. 2011. 2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011). In *Proceedings of the 5th ACM Conference on Recommender Systems (RecSys 2011)*. ACM, New York, NY, USA.

```bibtex
@inproceedings{Cantador:RecSys2011,
  author = {Cantador, Iv\'{a}n and Brusilovsky, Peter and Kuflik, Tsvi},
  title = {2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011)},
  booktitle = {Proceedings of the 5th ACM conference on Recommender Systems},
  series = {RecSys 2011},
  year = {2011},
  location = {Chicago, IL, USA},
  publisher = {ACM},
  address = {New York, NY, USA}
}
```

### Raw size
- `user_artists.dat`: 92,834 interaction rows (92,835 lines with header)
- `user_friends.dat`: 25,434 friend rows (25,435 lines with header); each friendship appears in both directions, corresponding to **12,717 undirected pairs**

### Preprocessing in our code
- **Filtering:** Iterative filtering (3 passes) keeping users with $\ge 5$ interactions and items with $\ge 2$ interactions.
- **Weight column:** Interactions are treated as **binary**; column 3 (`weight`, number of plays) is ignored (`--inter-cols 0 1`).
- **Social graph:** Treated as an **undirected** friendship graph.
- **Data split:** Per-user random split of **70% train / 10% validation / 20% test**.

### Final dataset statistics (verified with `results/lastfm_*_stats.csv`)
- **Users:** 1,873
- **Items:** 6,942
- **Interactions:** 82,113 (density: `0.6315%`, avg `43.84` items/user)
- **Social edges:** 12,588 undirected edges (density: `0.7180%`, avg `13.44` friends/user)

---

## 2. Ciao dataset

### Source & Preprocessing
- Built from the DivSR repository files (`ciao.train.rating` and `ciao.links`).
- **Filtering & Split:** Iterative filtering (users $\ge 5$ interactions, items $\ge 2$ interactions, 3 passes), undirected friendship graph, and per-user random split **70/10/20** (train/val/test).

### Final dataset statistics (verified with `results/ciao_v2_stats.csv`)
- **Users:** 5,716
- **Items:** 24,583
- **Interactions:** 122,475 (density: `0.0872%`, avg `21.43` items/user)
- **Social edges:** 73,822 undirected edges (density: `0.4520%`, avg `25.83` friends/user)

> Our Ciao version is built from the DivSR train file and differs from versions used in other papers (for example DGNN reports 1,925 users and 15,053 items), so absolute numbers are not directly comparable across papers.
