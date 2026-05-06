from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
 -----------------------------
# BPR-MF
# -----------------------------
class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

def sample_bpr_triplets(
    eligible_users: List[int],
    train_pos: Dict[int, List[int]],
    heldout: Dict[int, int],
    n_items: int,
    samples_per_user: int,
    seed: int,
    forbid_items: Optional[Dict[int, set]] = None,
):
    rng = np.random.RandomState(seed)
    users_out, pos_out, neg_out = [], [], []

    if forbid_items is None:
        train_pos_sets = {u: set(train_pos[u]) | {heldout[u]} for u in eligible_users}
    else:
        train_pos_sets = {u: set(train_pos[u]) | set(forbid_items[u]) for u in eligible_users}

    for u in eligible_users:
        pos_items = train_pos[u]
        if len(pos_items) == 0:
            continue
        forbid = train_pos_sets[u]
        for _ in range(samples_per_user):
            i = pos_items[rng.randint(len(pos_items))]
            j = rng.randint(n_items)
            while j in forbid:
                j = rng.randint(n_items)
            users_out.append(u)
            pos_out.append(i)
            neg_out.append(j)

    return (
        np.array(users_out, dtype=np.int64),
        np.array(pos_out, dtype=np.int64),
        np.array(neg_out, dtype=np.int64),
    )


def train_bpr_mf(cfg: Config, data_dict, seed: int):
    device = torch.device(cfg.device)
    model = BPRMF(data_dict["n_users"], data_dict["n_items"], cfg.embed_dim).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    eligible_users = data_dict["eligible_users"]
    train_pos = data_dict["train_pos"]
    heldout = data_dict["heldout"]
    n_items = data_dict["n_items"]
    forbid_items = data_dict.get("forbid_items")

    for epoch in range(cfg.epochs):
        users_arr, pos_arr, neg_arr = sample_bpr_triplets(
            eligible_users=eligible_users,
            train_pos=train_pos,
            heldout=heldout,
            n_items=n_items,
            samples_per_user=cfg.samples_per_user_per_epoch,
            seed=seed * 1000 + epoch,
            forbid_items=forbid_items,
        )

        perm = np.random.RandomState(seed * 1000 + epoch + 17).permutation(len(users_arr))
        users_arr, pos_arr, neg_arr = users_arr[perm], pos_arr[perm], neg_arr[perm]

        model.train()
        for start in range(0, len(users_arr), cfg.batch_size):
            end = start + cfg.batch_size
            u = torch.tensor(users_arr[start:end], dtype=torch.long, device=device)
            i = torch.tensor(pos_arr[start:end], dtype=torch.long, device=device)
            j = torch.tensor(neg_arr[start:end], dtype=torch.long, device=device)

            x_ui = model(u, i)
            x_uj = model(u, j)
            loss_bpr = -F.logsigmoid(x_ui - x_uj).mean()

            reg = (
                model.user_emb(u).pow(2).sum(dim=1).mean()
                + model.item_emb(i).pow(2).sum(dim=1).mean()
                + model.item_emb(j).pow(2).sum(dim=1).mean()
                + model.item_bias(i).pow(2).mean()
                + model.item_bias(j).pow(2).mean()
            )
            loss = loss_bpr + cfg.l2 * reg

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def precompute_base_scores(model: BPRMF, users: List[int], device: str):
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        user_ids = torch.tensor(users, dtype=torch.long, device=device)
        scores = model.score_all_items(user_ids).cpu().numpy().astype(np.float32)
    return {u: scores[idx] for idx, u in enumerate(users)}

def build_lightgcn_norm_adj(data_dict, device: str):
    n_users = data_dict["n_users"]
    n_items = data_dict["n_items"]
    train_pos = data_dict["train_pos"]

    rows = []
    cols = []

    for u, items in train_pos.items():
        for i in items:
            rows.append(u)
            cols.append(n_users + i)
            rows.append(n_users + i)
            cols.append(u)

    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    vals = np.ones(len(rows), dtype=np.float32)

    n_nodes = n_users + n_items
    deg = np.zeros(n_nodes, dtype=np.float32)
    np.add.at(deg, rows, 1.0)
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    norm_vals = deg_inv_sqrt[rows] * vals * deg_inv_sqrt[cols]

    idx = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long, device=device)
    val = torch.tensor(norm_vals, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes), device=device).coalesce()
    return adj

class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int, n_layers: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def computer(self, norm_adj):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)

        out = torch.stack(embs, dim=0).mean(dim=0)
        user_out, item_out = torch.split(out, [self.n_users, self.n_items], dim=0)
        return user_out, item_out

    def bpr_scores(self, u, i, j, norm_adj):
        user_out, item_out = self.computer(norm_adj)
        pu = user_out[u]
        qi = item_out[i]
        qj = item_out[j]
        x_ui = (pu * qi).sum(dim=-1)
        x_uj = (pu * qj).sum(dim=-1)
        return x_ui, x_uj

    def score_all_items(self, user_ids: torch.Tensor, norm_adj):
        user_out, item_out = self.computer(norm_adj)
        pu = user_out[user_ids]
        return pu @ item_out.T

def train_lightgcn(cfg: Config, data_dict, seed: int):
    device = torch.device(cfg.device)
    model = LightGCN(
        n_users=data_dict["n_users"],
        n_items=data_dict["n_items"],
        embed_dim=cfg.embed_dim,
        n_layers=cfg.lightgcn_layers,
    ).to(device)

    norm_adj = build_lightgcn_norm_adj(data_dict, cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lightgcn_lr)

    eligible_users = data_dict["eligible_users"]
    train_pos = data_dict["train_pos"]
    heldout = data_dict["heldout"]
    n_items = data_dict["n_items"]

    for epoch in range(cfg.lightgcn_epochs):
        users_arr, pos_arr, neg_arr = sample_bpr_triplets(
            eligible_users=eligible_users,
            train_pos=train_pos,
            heldout=heldout,
            n_items=n_items,
            samples_per_user=cfg.samples_per_user_per_epoch,
            seed=seed * 3000 + epoch,
        )

        perm = np.random.RandomState(seed * 3000 + epoch + 123).permutation(len(users_arr))
        users_arr, pos_arr, neg_arr = users_arr[perm], pos_arr[perm], neg_arr[perm]

        model.train()
        for start in range(0, len(users_arr), cfg.batch_size):
            end = start + cfg.batch_size
            u = torch.tensor(users_arr[start:end], dtype=torch.long, device=device)
            i = torch.tensor(pos_arr[start:end], dtype=torch.long, device=device)
            j = torch.tensor(neg_arr[start:end], dtype=torch.long, device=device)

            x_ui, x_uj = model.bpr_scores(u, i, j, norm_adj)
            loss_bpr = -F.logsigmoid(x_ui - x_uj).mean()

            reg = (
                model.user_emb(u).pow(2).sum(dim=1).mean()
                + model.item_emb(i).pow(2).sum(dim=1).mean()
                + model.item_emb(j).pow(2).sum(dim=1).mean()
            )
            loss = loss_bpr + cfg.lightgcn_l2 * reg

            opt.zero_grad()
            loss.backward()
            opt.step()

    return {"model": model, "norm_adj": norm_adj}

def precompute_base_scores_lightgcn(bundle, users: List[int], device: str):
    model = bundle["model"]
    norm_adj = bundle["norm_adj"]
    device = torch.device(device)

    model.eval()
    with torch.no_grad():
        user_ids = torch.tensor(users, dtype=torch.long, device=device)
        scores = model.score_all_items(user_ids, norm_adj).cpu().numpy().astype(np.float32)

    return {u: scores[idx] for idx, u in enumerate(users)}

class SASRecLite(nn.Module):
    def __init__(
        self,
        n_items: int,
        max_seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def encode(self, seqs: torch.Tensor):
        B, L = seqs.shape
        pos = torch.arange(L, device=seqs.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seqs) + self.pos_emb(pos)

        pad_mask = (seqs == 0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        lengths = (seqs != 0).sum(dim=1).clamp(min=1)
        gather_idx = (lengths - 1).view(B, 1, 1).expand(B, 1, h.size(-1))
        h_last = h.gather(1, gather_idx).squeeze(1)
        return self.ln(h_last)

    def forward(self, seqs: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        h = self.encode(seqs)
        pos_emb = self.item_emb(pos_items)
        neg_emb = self.item_emb(neg_items)
        pos_score = (h * pos_emb).sum(dim=-1)
        neg_score = (h * neg_emb).sum(dim=-1)
        return pos_score, neg_score

    def score_all_items_from_seqs(self, seqs: torch.Tensor):
        h = self.encode(seqs)
        item_mat = self.item_emb.weight[1:]  # exclude padding
        return h @ item_mat.T

def sample_sasrec_training_instances(
    eligible_users: List[int],
    train_pos: Dict[int, List[int]],
    heldout: Dict[int, int],
    n_items: int,
    max_seq_len: int,
    samples_per_user: int,
    seed: int,
):
    rng = np.random.RandomState(seed)

    seqs = []
    pos_items = []
    neg_items = []

    forbid_map = {}
    for u in eligible_users:
        hist_u = train_pos[u]
        if not isinstance(hist_u, (list, tuple, np.ndarray)):
            raise TypeError(f"[BAD train_pos[{u}]] type={type(hist_u)}, value={hist_u}")
        forbid_map[u] = set(hist_u) | {heldout[u]}

    for u in eligible_users:
        hist = train_pos[u]

        if not isinstance(hist, (list, tuple, np.ndarray)):
            raise TypeError(f"[BAD HIST] user={u}, type={type(hist)}, value={hist}")

        if len(hist) < 2:
            continue

        for _ in range(samples_per_user):
            t = int(rng.randint(1, len(hist)))
            prefix = hist[max(0, t - max_seq_len):t]
            target = hist[t]

            if not isinstance(prefix, (list, tuple, np.ndarray)):
                raise TypeError(
                    f"[BAD PREFIX] user={u}, type={type(prefix)}, value={prefix}, hist_type={type(hist)}, t={t}"
                )

            if isinstance(target, (list, tuple, np.ndarray)):
                raise TypeError(
                    f"[BAD TARGET] user={u}, type={type(target)}, value={target}, hist={hist}, t={t}"
                )

            neg = int(rng.randint(n_items))
            while neg in forbid_map[u]:
                neg = int(rng.randint(n_items))

            seq = np.zeros(max_seq_len, dtype=np.int64)

            try:
                tok = np.array([int(x) + 1 for x in prefix[-max_seq_len:]], dtype=np.int64)
            except Exception as e:
                raise TypeError(
                    f"[TOK BUILD FAILED] user={u}, prefix_type={type(prefix)}, "
                    f"prefix={prefix}, hist_type={type(hist)}, hist={hist}, err={repr(e)}"
                )

            if len(tok) > 0:
                seq[-len(tok):] = tok

            seqs.append(seq)
            pos_items.append(int(target) + 1)
            neg_items.append(int(neg) + 1)

    return (
        np.array(seqs, dtype=np.int64),
        np.array(pos_items, dtype=np.int64),
        np.array(neg_items, dtype=np.int64),
    )
def train_sasrec(cfg: Config, data_dict, seed: int):
    device = torch.device(cfg.device)

    model = SASRecLite(
        n_items=data_dict["n_items"],
        max_seq_len=cfg.sasrec_max_seq_len,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.sasrec_num_heads,
        num_layers=cfg.sasrec_num_layers,
        dropout=cfg.sasrec_dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.sasrec_lr, weight_decay=cfg.sasrec_l2)

    eligible_users = data_dict["eligible_users"]
    train_pos = data_dict["train_pos"]
    heldout = data_dict["heldout"]
    n_items = data_dict["n_items"]

    for epoch in range(cfg.sasrec_epochs):
        seqs, pos_items, neg_items = sample_sasrec_training_instances(
            eligible_users=eligible_users,
            train_pos=train_pos,
            heldout=heldout,
            n_items=n_items,
            max_seq_len=cfg.sasrec_max_seq_len,
            samples_per_user=cfg.samples_per_user_per_epoch,
            seed=seed * 7000 + epoch,
        )

        if len(seqs) == 0:
            break

        perm = np.random.RandomState(seed * 7000 + epoch + 19).permutation(len(seqs))
        seqs, pos_items, neg_items = seqs[perm], pos_items[perm], neg_items[perm]

        model.train()
        for start in range(0, len(seqs), cfg.batch_size):
            end = start + cfg.batch_size
            s = torch.tensor(seqs[start:end], dtype=torch.long, device=device)
            p = torch.tensor(pos_items[start:end], dtype=torch.long, device=device)
            n = torch.tensor(neg_items[start:end], dtype=torch.long, device=device)

            pos_score, neg_score = model(s, p, n)
            loss = -F.logsigmoid(pos_score - neg_score).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model
def precompute_base_scores_sasrec(cfg: Config, model: SASRecLite, users: List[int], train_pos: Dict[int, List[int]], device: str):
    device = torch.device(device)
    model.eval()

    seqs = []
    for u in users:
        hist = train_pos[u][-cfg.sasrec_max_seq_len:]
        seq = np.zeros(cfg.sasrec_max_seq_len, dtype=np.int64)
        tok = np.array([x + 1 for x in hist], dtype=np.int64)
        if len(tok) > 0:
            seq[-len(tok):] = tok
        seqs.append(seq)

    with torch.no_grad():
        seqs_t = torch.tensor(np.array(seqs, dtype=np.int64), dtype=torch.long, device=device)
        scores = model.score_all_items_from_seqs(seqs_t).cpu().numpy().astype(np.float32)

    return {u: scores[idx] for idx, u in enumerate(users)}

def train_and_precompute_ranker_scores(
    cfg: Config,
    data_dict,
    seed: int,
    users: List[int],
    ranker_name: Optional[str] = None,
):
    rk = (ranker_name or getattr(cfg, "ranker_name", "bprmf")).lower()

    if rk in {"bprmf", "bpr", "mf"}:
        model = train_bpr_mf(cfg, data_dict, seed)
        base_scores = precompute_base_scores(model, users, cfg.device)
        return model, base_scores

    if rk == "lightgcn":
        bundle = train_lightgcn(cfg, data_dict, seed)
        base_scores = precompute_base_scores_lightgcn(bundle, users, cfg.device)
        return bundle, base_scores

    if rk == "sasrec":
        model = train_sasrec(cfg, data_dict, seed)
        base_scores = precompute_base_scores_sasrec(cfg, model, users, data_dict["train_pos"], cfg.device)
        return model, base_scores

    raise ValueError(f"Unknown ranker_name={ranker_name}")



