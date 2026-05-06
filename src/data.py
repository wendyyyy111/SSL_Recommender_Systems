import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config


def load_ml1m(data_dir: str):
    ratings_path = os.path.join(data_dir, "ratings.dat")
    movies_path = os.path.join(data_dir, "movies.dat")
    users_path = os.path.join(data_dir, "users.dat")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1",
    )

    user_meta = {}
    if os.path.exists(users_path):
        users_df = pd.read_csv(
            users_path,
            sep="::",
            engine="python",
            names=["UserID", "Gender", "Age", "Occupation", "ZipCode"],
            encoding="latin-1",
        )
        for row in users_df.itertuples(index=False):
            user_meta[int(row.UserID)] = {"Gender": str(row.Gender).strip()}

    ratings = ratings[ratings["Rating"] >= 4].copy()
    ratings = ratings.sort_values(["UserID", "Timestamp", "MovieID"]).reset_index(drop=True)

    user_ids = sorted(ratings["UserID"].unique().tolist())
    item_ids = sorted(movies["MovieID"].unique().tolist())

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: j for j, i in enumerate(item_ids)}
    idx2item = {j: i for i, j in item2idx.items()}

    movies = movies[movies["MovieID"].isin(item2idx)].copy()
    movies["item_idx"] = movies["MovieID"].map(item2idx)

    genre_set = set()
    for g in movies["Genres"].fillna("").tolist():
        for t in str(g).split("|"):
            genre_set.add(t)
    genre_list = sorted(list(genre_set))
    genre2idx = {g: i for i, g in enumerate(genre_list)}

    n_items = len(item_ids)
    n_users = len(user_ids)
    n_genres = len(genre_list)

    titles = [""] * n_items
    item_genre_mat = np.zeros((n_items, n_genres), dtype=np.float32)
    item_genre_strings = [""] * n_items

    for row in movies.itertuples(index=False):
        i = int(row.item_idx)
        titles[i] = row.Title
        item_genre_strings[i] = str(row.Genres)
        for g in str(row.Genres).split("|"):
            if g in genre2idx:
                item_genre_mat[i, genre2idx[g]] = 1.0

    user_hist = {user2idx[u]: [] for u in user_ids}
    for row in ratings.itertuples(index=False):
        u = user2idx[int(row.UserID)]
        i = item2idx[int(row.MovieID)]
        user_hist[u].append(i)

    eligible_users = []
    train_pos = {}
    heldout = {}

    for u, hist in user_hist.items():
        if len(hist) >= 2:
            eligible_users.append(u)
            train_pos[u] = list(hist[:-1])
            heldout[u] = hist[-1]

    user_gender = {}
    for raw_uid in user_ids:
        uidx = user2idx[raw_uid]
        g = user_meta.get(int(raw_uid), {}).get("Gender")
        if g == "M":
            user_gender[uidx] = 1
        elif g == "F":
            user_gender[uidx] = 0

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_genres": n_genres,
        "genre_list": genre_list,
        "genre2idx": genre2idx,
        "titles": titles,
        "item_genre_strings": item_genre_strings,
        "item_genre_mat": item_genre_mat,
        "eligible_users": sorted(eligible_users),
        "train_pos": train_pos,
        "heldout": heldout,
        "user_hist": user_hist,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "user_gender": user_gender,
    }


def build_state_conditioned_audit_data(
    cfg: Config,
    data_dict,
    bucket0_idx: List[int],
    bucket1_idx: List[int],
):
    user_hist = data_dict["user_hist"]
    item_genre_mat = data_dict["item_genre_mat"]

    b0_mask = (item_genre_mat[:, bucket0_idx].sum(axis=1) > 0)
    b1_mask = (item_genre_mat[:, bucket1_idx].sum(axis=1) > 0)

    audit_users = []
    train_pos_sc = {}
    heldout_z0 = {}
    heldout_z1 = {}

    W = max(int(cfg.holdout_window), 2)

    for u in data_dict["eligible_users"]:
        hist = list(user_hist[u])
        if len(hist) < cfg.min_train_prefix + W:
            continue

        prefix = hist[:-1]
        heldout_item = int(hist[-1])
        suffix = hist[-W:]
        has_b0 = any(bool(b0_mask[i]) for i in suffix)
        has_b1 = any(bool(b1_mask[i]) for i in suffix)

        if len(prefix) < cfg.min_train_prefix:
            continue
        if not (has_b0 and has_b1):
            continue

        audit_users.append(u)
        train_pos_sc[u] = list(prefix)
        heldout_z0[u] = heldout_item
        heldout_z1[u] = heldout_item

    return sorted(audit_users), train_pos_sc, heldout_z0, heldout_z1


def split_dev_val_test_users(
    users: List[int],
    seed: int,
    dev_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    users = list(users)
    rng = np.random.RandomState(seed)
    rng.shuffle(users)

    total = dev_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"dev_ratio + val_ratio + test_ratio must equal 1. Got {total}")

    n = len(users)
    n_dev = int(n * dev_ratio)
    n_val = int(n * val_ratio)

    dev_users = sorted(users[:n_dev])
    val_users = sorted(users[n_dev:n_dev + n_val])
    test_users = sorted(users[n_dev + n_val:])

    if len(dev_users) == 0 or len(val_users) == 0 or len(test_users) == 0:
        raise ValueError(
            f"Bad split sizes: dev={len(dev_users)}, val={len(val_users)}, test={len(test_users)}"
        )

    return dev_users, val_users, test_users


def compute_user_history_genre_dist(users: List[int], train_pos: Dict[int, List[int]], item_genre_mat: np.ndarray):
    n_genres = item_genre_mat.shape[1]
    h_user = {}
    for u in users:
        items = train_pos[u]
        mat = item_genre_mat[items]
        s = mat.sum(axis=0)
        denom = float(s.sum())
        if denom <= 0:
            h = np.zeros(n_genres, dtype=np.float32)
        else:
            h = (s / denom).astype(np.float32)
        h_user[u] = h
    return h_user