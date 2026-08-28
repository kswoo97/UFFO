
import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from tqdm import tqdm
from collections import defaultdict

# ================================================================
# Data loading
# ================================================================

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('./datasets/%s_final.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# ================================================================
# LightGCN utilities and sampler
# ================================================================

def build_norm_adj_from_tensor(num_users: int, num_items: int, interactions_2xd: torch.LongTensor):
    """
    interactions_2xd: LongTensor [2, d], row0 users in [0, U-1], row1 items in [0, I-1]
    Returns torch.sparse_coo_tensor of shape (U+I, U+I)
    """
    assert interactions_2xd.dtype == torch.long
    assert interactions_2xd.dim() == 2 and interactions_2xd.size(0) == 2

    users = interactions_2xd[0].cpu().numpy().astype(np.int64)
    items = interactions_2xd[1].cpu().numpy().astype(np.int64)

    N = num_users + num_items
    u = users
    i = items + num_users  # item nodes shifted

    # edges both directions
    row = np.concatenate([u, i])
    col = np.concatenate([i, u])
    data = np.ones(len(row), dtype=np.float32)

    A = coo_matrix((data, (row, col)), shape=(N, N)).tocsr()

    deg = np.array(A.sum(axis=1)).squeeze().astype(np.float32)  # (N,)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0

    A = A.tocoo()
    vals = deg_inv_sqrt[A.row] * A.data * deg_inv_sqrt[A.col]

    indices = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    values = torch.from_numpy(vals.astype(np.float32))
    sp = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    return sp

def build_user_pos_dict(num_users: int, interactions_2xd: torch.LongTensor):
    users = interactions_2xd[0].tolist()
    items = interactions_2xd[1].tolist()

    user_pos = [[] for _ in range(num_users)]
    for u, i in zip(users, items):
        user_pos[u].append(i)

    user_pos_sets = [set(lst) for lst in user_pos]
    users_with_pos = [u for u in range(num_users) if len(user_pos[u]) > 0]
    return user_pos, user_pos_sets, users_with_pos

class BPRSamplerNoReplace:
    def __init__(self, num_users, num_items, train_interactions_2xd: torch.LongTensor, seed: int = 42):
        self.num_users = num_users
        self.num_items = num_items
        self.base_seed = seed
        self.rng = random.Random(seed)

        self.user_pos, self.user_pos_sets, self.users_with_pos = build_user_pos_dict(
            num_users, train_interactions_2xd
        )

    def set_epoch(self, epoch: int):
        # Different randomness per epoch, but reproducible across runs
        self.rng.seed(self.base_seed + epoch)

    def epoch_batches(self, batch_size: int, device: str, drop_last: bool = False):
        users = self.users_with_pos[:]   # copy
        self.rng.shuffle(users)

        n = len(users)
        end = n - (n % batch_size) if drop_last else n

        for s in range(0, end, batch_size):
            batch_users = users[s:s + batch_size]

            pos_items = []
            neg_items = []

            for u in batch_users:
                pos = self.rng.choice(self.user_pos[u])

                while True:
                    neg = self.rng.randrange(self.num_items)
                    if neg not in self.user_pos_sets[u]:
                        break

                pos_items.append(pos)
                neg_items.append(neg)

            yield (
                torch.tensor(batch_users, dtype=torch.long, device=device),
                torch.tensor(pos_items, dtype=torch.long, device=device),
                torch.tensor(neg_items, dtype=torch.long, device=device),
            )

# ================================================================
# Static model
# ================================================================

class LightGCN_Att_Learn(nn.Module) :
    
    def __init__(self, num_users, num_items, train_adj, valid_adj, embedding_dim=64, num_layers=3, reg=1e-4, 
                 itemX1 = None, itemX2 = None, itemX3 = None, itemX4 = None, device = "cuda:0", tau = 0.5, 
                 balance_loss_term = 1.0):
        super().__init__()
        
        self.train_adj = train_adj
        self.valid_adj = valid_adj
        
        self.cur_ep = 1.0
        self.bal_term = balance_loss_term
        
        self.num_users = num_users
        self.num_items = num_items
        self.N = num_users + num_items

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.reg = reg

        self.user_emb = nn.Embedding(num_users, embedding_dim*2)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        self.item_emb1 = torch.nn.Embedding.from_pretrained(itemX1, freeze=True)
        self.item_emb2 = torch.nn.Embedding.from_pretrained(itemX2, freeze=True)
        self.item_emb3 = torch.nn.Embedding.from_pretrained(itemX3, freeze=True)
        self.item_emb4 = torch.nn.Embedding.from_pretrained(itemX4, freeze=True)

        reduce_dim = embedding_dim
        self.reduce_dim = reduce_dim
        self.proj_dim = reduce_dim
        
        self.att_user = torch.nn.Parameter(
            torch.zeros(self.reduce_dim, device=device)
        )
        
        self.projector1 = torch.nn.Sequential(
            torch.nn.Linear(itemX1.shape[1], reduce_dim, bias = True),
            torch.nn.LayerNorm(reduce_dim, eps = 1e-8),
        )
        
        self.projector2 = torch.nn.Sequential(
            torch.nn.Linear(itemX2.shape[1], reduce_dim, bias = True),
            torch.nn.LayerNorm(reduce_dim, eps = 1e-8),
        )
        
        self.projector3 = torch.nn.Sequential(
            torch.nn.Linear(itemX3.shape[1], reduce_dim, bias = True),
            torch.nn.LayerNorm(reduce_dim, eps = 1e-8),
        )
        
        self.projector4 = torch.nn.Sequential(
            torch.nn.Linear(itemX4.shape[1], reduce_dim, bias = True),
            torch.nn.LayerNorm(reduce_dim, eps = 1e-8),
        )

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        
        self.one_mats = torch.ones((self.num_users, 4), device = device) / 4
        
        self.identities = torch.ones((self.num_users, self.embedding_dim), device = device)
        self.one_mappers = torch.ones(self.embedding_dim, device = device)
        self.tau = tau
        self.global_seed = 0
        self.device = device
    
        
    def item_encoding(self, indexing = None) : 
        
        if indexing is not None : 
            
            Z1 = self.projector1(self.item_emb1(indexing)) # * self.att_item[:, 0]
            Z2 = self.projector2(self.item_emb2(indexing)) # * self.att_item[:, 1]
            Z3 = self.projector3(self.item_emb3(indexing)) # * self.att_item[:, 2]            
            Z4 = self.projector4(self.item_emb4(indexing)) # * self.att_item[:, 2]     
        else : 
        
            Z1 = self.projector1(self.item_emb1.weight) # * self.att_item[:, 0]
            Z2 = self.projector2(self.item_emb2.weight) # * self.att_item[:, 1]
            Z3 = self.projector3(self.item_emb3.weight) # * self.att_item[:, 2]
            Z4 = self.projector4(self.item_emb4.weight) # * self.att_item[:, 2]            
            
        Z = torch.hstack([Z1, Z2, Z3, Z4])
        
        return Z

    def propagate(self, edges) :
        
        ego = torch.cat([torch.hstack([self.user_emb.weight[: , :self.embedding_dim], (self.user_emb.weight[: , self.embedding_dim:].repeat_interleave(4, dim = 1))]), 
                         torch.hstack([self.item_emb.weight, self.item_encoding()])], dim=0)  # (N, d)
        
        embs = [ego]
        x = ego
        
        for _ in range(self.num_layers):
            x = torch.sparse.mm(edges, x)
            embs.append(x)
            
        out = torch.stack(embs, dim=0).mean(dim=0)
        
        inter_score = out[:self.num_users, self.reduce_dim : int(self.reduce_dim*5)].reshape(self.num_users, 4, self.reduce_dim)
        score_matrix = inter_score
        self.score = torch.matmul(score_matrix, (self.att_user))   # [N, 4]

        S = torch.softmax(self.score / self.tau, dim=1)            # [N, 4]
        
        self.one_vectors = (torch.hstack([self.identities, S.repeat_interleave(self.reduce_dim, dim = 1)])) 
        
        return out[: self.num_users] * self.one_vectors, out[self.num_users :], S, out[: self.num_users]

    def bpr_loss(self, users, pos_items, neg_items):
        
        U, I, S, unU = self.propagate(self.train_adj)
        
        u = U[users]
        p = I[pos_items]
        n = I[neg_items]
        
        new_u = unU[users]
        
        S_p = u * p
        S_n = u * n
        
        S_total = torch.vstack([new_u * p, new_u * n])

        pos_scores = S_p.sum(dim=-1)
        neg_scores = S_n.sum(dim=-1)

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        reduce_dim = self.reduce_dim
 
        # regularize ego embeddings
        u0 = self.user_emb(users)
        p0 = torch.hstack([self.item_emb(pos_items), self.item_encoding(pos_items)])
        n0 = torch.hstack([self.item_emb(neg_items), self.item_encoding(neg_items)])
        
        reg = (u0.pow(2).sum() + p0.pow(2).sum() + n0.pow(2).sum()) / users.size(0)
        totalP = S.mean(dim = 0)
        
        reg2 = torch.sum(totalP * torch.log(totalP))
        
        return loss + self.reg * reg + (self.bal_term) * reg2
    

    @torch.no_grad()
    def full_scores(self, valid, user_interests = None):
        
        if valid : 
            U, I, _, _ = self.propagate(self.train_adj)
        
        else : 
            U, I, _, _ = self.propagate(self.valid_adj)
            
        return U @ I.t()

# ================================================================
# Static evaluator
# ================================================================

def evaluate(
    model,
    train_user_pos_sets,
    test_user_pos_sets,
    device="cpu",
    is_valid=True,
):
    model.eval()

    scores = model.full_scores(is_valid).to(device)  # (U, I)
    num_users = scores.shape[0]

    # --------------------------------------------------
    # 1. Filter train positives using vectorized indexing
    # --------------------------------------------------
    train_users = []
    train_items = []

    for u in range(1, num_users):
        items = train_user_pos_sets[u]
        if len(items) == 0:
            continue

        train_users.extend([u] * len(items))
        train_items.extend(items)

    if len(train_users) > 0:
        train_users = torch.tensor(train_users, device=device, dtype=torch.long)
        train_items = torch.tensor(train_items, device=device, dtype=torch.long)
        scores[train_users, train_items] = -1e9

    # --------------------------------------------------
    # 2. Collect valid users and ground-truth items
    # --------------------------------------------------
    valid_users = []
    gt_items = []

    for u in range(1, num_users):
        true_test = test_user_pos_sets[u]

        if not true_test:
            continue

        gt_item = true_test[0] if isinstance(true_test, list) else next(iter(true_test))

        valid_users.append(u)
        gt_items.append(gt_item)

    n_valid = len(valid_users)

    if n_valid == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    valid_users = torch.tensor(valid_users, device=device, dtype=torch.long)
    gt_items = torch.tensor(gt_items, device=device, dtype=torch.long)

    # --------------------------------------------------
    # 3. Compute top-30 only for valid users
    # --------------------------------------------------
    valid_scores = scores[valid_users]  # (n_valid, I)

    top_items = torch.topk(valid_scores, k=30, dim=1).indices  # (n_valid, 30)

    # --------------------------------------------------
    # 4. Find ground-truth rank inside top-30
    # --------------------------------------------------
    hits = top_items == gt_items.unsqueeze(1)  # (n_valid, 30)

    hit_any = hits.any(dim=1)  # (n_valid,)

    # rank is only meaningful for hit users
    hit_positions = hits.float().argmax(dim=1)  # 0-based rank

    # --------------------------------------------------
    # 5. Compute HR / NDCG vectorized
    # --------------------------------------------------
    ranks = hit_positions[hit_any].float()

    HR10 = ((hit_any) & (hit_positions < 10)).float().mean().item()
    HR20 = ((hit_any) & (hit_positions < 20)).float().mean().item()
    HR30 = ((hit_any) & (hit_positions < 30)).float().mean().item()

    ndcg_values = torch.zeros(n_valid, device=device)

    ndcg_values[hit_any] = 1.0 / torch.log2(ranks + 2.0)

    NDCG10 = ndcg_values[(hit_any) & (hit_positions < 10)].sum().item() / n_valid
    NDCG20 = ndcg_values[(hit_any) & (hit_positions < 20)].sum().item() / n_valid
    NDCG30 = ndcg_values[(hit_any) & (hit_positions < 30)].sum().item() / n_valid

    return (
        NDCG10,
        NDCG20,
        NDCG30,
        HR10,
        HR20,
        HR30,
    )


# ================================================================
# Experiment utilities
# ================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_embeddings(dataset, device):
    base = "./datasets"
    names = ["gpt", "gemini", "siglip", "pe"]
    return tuple(
        torch.tensor(
            np.load(f"{base}/{dataset}_embs_{name}.npy"),
            dtype=torch.float32,
            device=device,
        )
        for name in names
    )


def prepare_static_interactions(dataset, device):
    user_train, user_valid, user_test, usernum, itemnum = dataset

    train_sets = [[], []]
    valid_sets = [[], []]
    train_valid_sets = [[], []]
    test_sets = [[], []]

    for u in user_train:
        train_sets[0].extend([u] * len(user_train[u]))
        train_sets[1].extend(user_train[u])

        train_valid_sets[0].extend([u] * (len(user_train[u]) + len(user_valid[u])))
        train_valid_sets[1].extend(user_train[u] + user_valid[u])

    for u in range(1, usernum + 1):
        if len(user_valid.get(u, [])) == 0 or len(user_test.get(u, [])) == 0:
            continue
        valid_sets[0].append(u)
        valid_sets[1].append(user_valid[u][0])
        test_sets[0].append(u)
        test_sets[1].append(user_test[u][0])

    return (
        torch.LongTensor(train_sets).to(device),
        torch.LongTensor(valid_sets).to(device),
        torch.LongTensor(train_valid_sets).to(device),
        torch.LongTensor(test_sets).to(device),
        usernum + 1,
        itemnum + 1,
    )


# ================================================================
# One static training run
# ================================================================

def train_one_run(
    num_users,
    num_items,
    train_interactions,
    train_valid_interactions,
    valid_interactions,
    test_interactions,
    X1,
    X2,
    X3,
    X4,
    args,
    lr,
    tau,
    balance_loss_term,
    seed,
):
    set_seed(seed)

    train_adj = build_norm_adj_from_tensor(
        num_users, num_items, train_interactions
    ).to(args.device)
    valid_adj = build_norm_adj_from_tensor(
        num_users, num_items, train_valid_interactions
    ).to(args.device)

    # This is the model used by the original static training code.
    model = LightGCN_Att_Learn(
        num_users=num_users,
        num_items=num_items,
        train_adj=train_adj,
        valid_adj=valid_adj,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        reg=args.wdecay,
        itemX1=X1,
        itemX2=X2,
        itemX3=X3,
        itemX4=X4,
        device=args.device,
        tau=tau,
        balance_loss_term=balance_loss_term,
    ).to(args.device)

    sampler = BPRSamplerNoReplace(
        num_users,
        num_items,
        train_interactions,
        seed,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.wdecay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.9,
    )

    _, train_pos_sets, _ = build_user_pos_dict(num_users, train_interactions)
    _, train_valid_pos_sets, _ = build_user_pos_dict(num_users, train_valid_interactions)
    _, valid_pos_sets, _ = build_user_pos_dict(num_users, valid_interactions)
    _, test_pos_sets, _ = build_user_pos_dict(num_users, test_interactions)

    best_valid_hr30 = -1.0
    best_result = None

    for epoch in tqdm(range(1, args.num_epochs + 1), leave=False):
        model.train()
        sampler.set_epoch(epoch)

        for users, pos, neg in sampler.epoch_batches(
            batch_size=args.batch_size,
            device=args.device,
            drop_last=True,
        ):
            loss = model.bpr_loss(users, pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        model.cur_ep = epoch + 1

        if epoch > args.eval_start_epoch:
            valid_result = evaluate(
                model,
                train_pos_sets,
                valid_pos_sets,
                device=args.device,
                is_valid=True,
            )
            test_result = evaluate(
                model,
                train_valid_pos_sets,
                test_pos_sets,
                device=args.device,
                is_valid=False,
            )

            if valid_result[5] > best_valid_hr30:
                best_valid_hr30 = valid_result[5]
                best_result = [tuple(valid_result), tuple(test_result)]

    if best_result is None:
        raise RuntimeError(
            "No evaluation was performed. Set eval_start_epoch < num_epochs."
        )

    return model, best_result


# ================================================================
# Running code
# ================================================================

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='toys_games', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--eval_start_epoch', default=30, type=int)

    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--wdecay', default=1e-4, type=float)

    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--result_path', default=None, type=str)
    
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--tau', default=1.0, type=float)
    return parser


def main(args):
    dataset = data_partition(args.dataset)
    trainE, validE, trainvalidE, testE, num_users, num_items = \
        prepare_static_interactions(dataset, args.device)

    X1, X2, X3, X4 = load_feature_embeddings(args.dataset, args.device)

    lr_grid = [args.lr]
    tau_grid = [args.tau]
    balance_loss_term = 1.0

    final_performance_dict = {}

    for lr in lr_grid:
        for tau in tau_grid:
            print()
            print(
                f"Dataset: {args.dataset} | LR: {lr} | "
                f"Tau: {tau}")

            all_performance = []

            for seed_idx in range(args.num_seeds):
                seed = (seed_idx + 1) * 2026

                _, result_set = train_one_run(
                    num_users=num_users,
                    num_items=num_items,
                    train_interactions=trainE,
                    train_valid_interactions=trainvalidE,
                    valid_interactions=validE,
                    test_interactions=testE,
                    X1=X1,
                    X2=X2,
                    X3=X3,
                    X4=X4,
                    args=args,
                    lr=lr,
                    tau=tau,
                    balance_loss_term=balance_loss_term,
                    seed=seed,
                )
                all_performance.append(result_set)

            test_ndcg20 = [r[1][1] for r in all_performance]
            test_hr20 = [r[1][4] for r in all_performance]
            test_ndcg30 = [r[1][2] for r in all_performance]
            test_hr30 = [r[1][5] for r in all_performance]

            print(
                f"Test NDCG@20: "
                f"{np.mean(test_ndcg20):.4f} ({np.std(test_ndcg20):.4f}) |"
                f"Test NDCG@30: "
                f"{np.mean(test_ndcg30):.4f} ({np.std(test_ndcg30):.4f}) |"
                f"Test HR@20: "
                f"{np.mean(test_hr20):.4f} ({np.std(test_hr20):.4f}) |"
                f"Test HR@30: "
                f"{np.mean(test_hr30):.4f} ({np.std(test_hr30):.4f}) |"
            )
            
    print('Done')


if __name__ == '__main__':
    main(build_parser().parse_args())
