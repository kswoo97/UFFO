
import argparse
import copy
import os
import pickle
import random
import sys
from collections import defaultdict
from multiprocessing import Process, Queue

import numpy as np
import torch
from tqdm import tqdm

# ================================================================
# Data loading + WarpSampler
# ================================================================

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return uid, seq, pos, neg

    np.random.seed(SEED)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0

    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)

        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, seed=42):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []

        for i in range(n_workers):
            worker_seed = seed + i
            p = Process(
                target=sample_function,
                args=(User, usernum, itemnum, batch_size, maxlen, self.result_queue, worker_seed),
            )
            p.daemon = True
            p.start()
            self.processors.append(p)

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    path = './datasets/%s_final.txt' % fname
    with open(path, 'r') as f:
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
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]

    return [user_train, user_valid, user_test, usernum, itemnum]


# ================================================================
# SASRec components
# ================================================================

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


def get_entropy_reg(fusion_weights, eps=1e-12):
    # fusion_weights: (B, F, T)
    avg_probs = fusion_weights.mean(dim=(0, 2))
    neg_entropy_loss = (avg_probs * torch.log(avg_probs + eps)).sum()
    return neg_entropy_loss, avg_probs

# ================================================================
# Multi-view SASRec model
# Views: ['id', 'gpt', 'gemini', 'siglip', 'pe']
# Personalized fusion: True
# Batched shared SASRec: True
# Entropy regularization: True
# ================================================================

class ID4FM_UserAdaptive_SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args, X1, X2, X3, X4, tau):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.hidden_units = args.hidden_units
        self.num_heads = args.num_heads
        self.num_features = 5
        self.num_fm_features = 4
        self.tau = tau

        # View 0: pure trainable ID representation.
        self.id_item_emb = torch.nn.Embedding(
            item_num + 1,
            args.hidden_units,
            padding_idx=0,
        )

        self.X1 = torch.nn.Embedding.from_pretrained(X1, freeze=True, padding_idx=0)
        self.X2 = torch.nn.Embedding.from_pretrained(X2, freeze=True, padding_idx=0)
        self.X3 = torch.nn.Embedding.from_pretrained(X3, freeze=True, padding_idx=0)
        self.X4 = torch.nn.Embedding.from_pretrained(X4, freeze=True, padding_idx=0)

        self.projector1 = torch.nn.Sequential(
            torch.nn.Linear(X1.shape[1], args.hidden_units),
            torch.nn.LayerNorm(args.hidden_units)
        )
        self.projector2 = torch.nn.Sequential(
            torch.nn.Linear(X2.shape[1], args.hidden_units),
            torch.nn.LayerNorm(args.hidden_units)
        )
        self.projector3 = torch.nn.Sequential(
            torch.nn.Linear(X3.shape[1], args.hidden_units),
            torch.nn.LayerNorm(args.hidden_units)
        )
        self.projector4 = torch.nn.Sequential(
            torch.nn.Linear(X4.shape[1], args.hidden_units),
            torch.nn.LayerNorm(args.hidden_units)
        )

        # Shared gating vector. The resulting fusion weights are
        # user/history-specific because they are computed from each
        # user's branch-specific SASRec hidden states.
        self.att_user = torch.nn.Parameter(
            torch.zeros(
                args.hidden_units,
                dtype=torch.float32,
                device=self.dev,
            )
        )

        self.pos_emb = torch.nn.Embedding(
            args.maxlen + 1, args.hidden_units, padding_idx=0
        )
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(
                torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            )
            self.attention_layers.append(
                torch.nn.MultiheadAttention(
                    args.hidden_units, args.num_heads, args.dropout_rate
                )
            )
            self.forward_layernorms.append(
                torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            )

        self.to(self.dev)

    def _to_long_tensor(self, x):
        if torch.is_tensor(x):
            return x.long().to(self.dev)
        return torch.LongTensor(x).to(self.dev)

    def encode_all_features(self, seq):
        """
        Return five SEPARATE item-view representations.

        Output shape:
            (B, 5, T, H)

        View order:
            0: ID
            1: GPT
            2: Gemini
            3: SigLIP
            4: PE

        No ID embedding is added to any FM view.
        """
        seq = self._to_long_tensor(seq)

        if seq.dim() == 1:
            seq = seq.unsqueeze(0)

        z_id = self.id_item_emb(seq)

        z1 = self.projector1(
            self.X1(seq)
        )
        z2 = self.projector2(
            self.X2(seq)
        )
        z3 = self.projector3(
            self.X3(seq)
        )
        z4 = self.projector4(
            self.X4(seq)
        )

        # Projector/LayerNorm bias can make padding nonzero.
        padding_mask = (
            seq == 0
        ).unsqueeze(-1)

        z_id = z_id.masked_fill(
            padding_mask,
            0.0,
        )
        z1 = z1.masked_fill(
            padding_mask,
            0.0,
        )
        z2 = z2.masked_fill(
            padding_mask,
            0.0,
        )
        z3 = z3.masked_fill(
            padding_mask,
            0.0,
        )
        z4 = z4.masked_fill(
            padding_mask,
            0.0,
        )

        return torch.stack(
            [
                z_id,
                z1,
                z2,
                z3,
                z4,
            ],
            dim=1,
        )

    def sasrec_encode_batched(
        self,
        input_embs,
        log_seqs,
    ):
        """
        Shared SASRec over all views in one batched pass.

        input_embs:
            (B, F, T, H)

        Internally:
            (B, F, T, H)
                -> (B*F, T, H)
                -> shared SASRec
                -> (B, F, T, H)

        The backbone parameters are shared, but each view receives its own
        view-specific input representation.
        """
        log_seqs = self._to_long_tensor(
            log_seqs
        )

        B, F, T, H = input_embs.shape

        assert F == self.num_features

        seqs = (
            input_embs
            * (
                self.hidden_units
                ** 0.5
            )
        )

        poss = torch.arange(
            1,
            T + 1,
            device=self.dev,
        ).unsqueeze(
            0
        ).expand(
            B,
            T,
        )

        poss = (
            poss
            * (
                log_seqs != 0
            ).long()
        )

        seqs = (
            seqs
            + self.pos_emb(
                poss
            ).unsqueeze(1)
        )

        seqs = self.emb_dropout(
            seqs
        )

        # Explicitly zero padding positions before shared SASRec.
        padding_mask = (
            log_seqs == 0
        ).unsqueeze(1).unsqueeze(-1)

        seqs = seqs.masked_fill(
            padding_mask,
            0.0,
        )

        # Core batch-processing step across views.
        seqs = seqs.reshape(
            B * F,
            T,
            H,
        )

        expanded_padding_mask = padding_mask.expand(B, F, T, 1).reshape(B * F, T, 1)
        padding_2d = expanded_padding_mask.squeeze(-1)

        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=self.dev), diagonal=1
        )
        attention_mask = causal.unsqueeze(0).expand(B * F, -1, -1).clone()
        attention_mask |= padding_2d.unsqueeze(1)

        padded = padding_2d.nonzero(as_tuple=False)
        if padded.numel() > 0:
            b, q = padded[:, 0], padded[:, 1]
            attention_mask[b, q, :] = True
            attention_mask[b, q, q] = False

        attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)

        for i in range(
            len(
                self.attention_layers
            )
        ):
            seqs = torch.transpose(
                seqs,
                0,
                1,
            )

            Q = self.attention_layernorms[
                i
            ](
                seqs
            )

            mha_outputs, _ = (
                self.attention_layers[
                    i
                ](
                    Q,
                    seqs,
                    seqs,
                    attn_mask=attention_mask,
                    need_weights=False,
                )
            )

            seqs = Q + mha_outputs

            seqs = torch.transpose(
                seqs,
                0,
                1,
            )

            seqs = self.forward_layernorms[
                i
            ](
                seqs
            )

            seqs = self.forward_layers[
                i
            ](
                seqs
            )

            seqs = seqs.masked_fill(
                expanded_padding_mask,
                0.0,
            )

        seqs = self.last_layernorm(
            seqs
        )

        seqs = seqs.masked_fill(
            expanded_padding_mask,
            0.0,
        )

        return seqs.reshape(
            B,
            F,
            T,
            H,
        )

    def get_fusion_weights(self, log_feats):
        """
        Personalized weights over ONLY the four FM views.

        log_feats:
            (B, 5, T, H)
            view 0 = ID
            views 1:5 = GPT, Gemini, SigLIP, PE

        returns:
            (B, 4, T)
        """
        fm_feats = log_feats[:, 1:, :, :]  # (B, 4, T, H)

        scores = (
            fm_feats
            * self.att_user.view(1, 1, 1, -1)
        ).sum(dim=-1)

        return torch.softmax(
            scores / self.tau,
            dim=1,
        )

    @torch.no_grad()
    def get_final_view_scores(self, log_seqs):
        """Return final-step gating logits and weights for ONLY the 4 FM views."""
        log_input_embs = self.encode_all_features(log_seqs)
        log_feats = self.sasrec_encode_batched(log_input_embs, log_seqs)

        fm_final_feats = log_feats[:, 1:, -1, :]  # (B, 4, H)
        raw_scores = (
            fm_final_feats * self.att_user.view(1, 1, -1)
        ).sum(dim=-1)  # (B, 4)

        fusion_weights = torch.softmax(raw_scores / self.tau, dim=1)
        return raw_scores, fusion_weights

    def get_train_logits_all_features(
        self,
        log_feats,
        target_seqs,
    ):
        target_embs = self.encode_all_features(
            target_seqs
        )

        return (
            log_feats
            * target_embs
        ).sum(
            dim=-1
        )

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """
        Final training score:
            ID score + personalized fusion of four FM scores.
        """
        log_input_embs = self.encode_all_features(log_seqs)
        log_feats = self.sasrec_encode_batched(log_input_embs, log_seqs)
        # (B, 5, T, H)

        pos_embs = self.encode_all_features(pos_seqs)
        neg_embs = self.encode_all_features(neg_seqs)

        pos_stream_scores = (log_feats * pos_embs).sum(dim=-1)  # (B,5,T)
        neg_stream_scores = (log_feats * neg_embs).sum(dim=-1)

        id_pos = pos_stream_scores[:, 0, :]                     # (B,T)
        id_neg = neg_stream_scores[:, 0, :]

        fm_pos = pos_stream_scores[:, 1:, :]                    # (B,4,T)
        fm_neg = neg_stream_scores[:, 1:, :]

        fm_weights = self.get_fusion_weights(log_feats)         # (B,4,T)

        personalized_fm_pos = (fm_weights * fm_pos).sum(dim=1)
        personalized_fm_neg = (fm_weights * fm_neg).sum(dim=1)

        pos_logits = id_pos + personalized_fm_pos
        neg_logits = id_neg + personalized_fm_neg

        # Entropy regularization is applied only to FM personalization.
        avg_weights = fm_weights.mean(dim=(0, 2))
        entropy_loss = torch.sum(
            avg_weights * torch.log(avg_weights + 1e-12)
        )

        return pos_logits, neg_logits, entropy_loss

    @torch.no_grad()
    def predict(self, user_ids, log_seqs, item_indices, fusion_weight_override=None):
        """
        Final prediction:
            ID score + personalized four-FM score.

        fusion_weight_override, when provided, must have shape (B, 4)
        and overrides only the FM weights.
        """
        log_input_embs = self.encode_all_features(log_seqs)
        log_feats = self.sasrec_encode_batched(log_input_embs, log_seqs)
        final_feat = log_feats[:, :, -1, :]                     # (B,5,H)
        B = final_feat.shape[0]

        if fusion_weight_override is None:
            fm_weights = self.get_fusion_weights(log_feats)[:, :, -1]  # (B,4)
        else:
            fm_weights = torch.as_tensor(
                fusion_weight_override,
                dtype=final_feat.dtype,
                device=self.dev,
            )

        item_indices = self._to_long_tensor(item_indices)
        if item_indices.dim() == 1:
            item_indices = item_indices.unsqueeze(0)

        item_embs = self.encode_all_features(item_indices)       # (1/B,5,I,H)

        if item_embs.shape[0] == 1 and B > 1:
            item_embs = item_embs.expand(B, -1, -1, -1)

        if item_embs.shape[0] != B:
            raise ValueError(
                "Candidate batch dimension must be 1 or equal to user batch size."
            )

        stream_scores = (
            item_embs * final_feat.unsqueeze(2)
        ).sum(dim=-1)                                            # (B,5,I)

        id_scores = stream_scores[:, 0, :]                       # (B,I)
        fm_scores = stream_scores[:, 1:, :]                      # (B,4,I)

        personalized_fm_scores = (
            fm_weights.unsqueeze(-1) * fm_scores
        ).sum(dim=1)

        return id_scores + personalized_fm_scores


# ================================================================
# Full-sort evaluator
# ================================================================

def _prepare_eval_data_user_sasrec(dataset, args, mode="test", max_users=None):
    train, valid, test, usernum, itemnum = dataset

    if max_users is None:
        users = range(1, usernum + 1)
    else:
        users = range(1, min(usernum + 1, max_users + 1))

    all_users = []
    all_seqs = []
    all_targets = []
    all_rated = []

    for u in users:
        if mode == "test":
            if len(train[u]) < 1 or len(valid[u]) < 1 or len(test[u]) < 1:
                continue

            seq = np.zeros(args.maxlen, dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1

            for item in reversed(train[u]):
                seq[idx] = item
                idx -= 1
                if idx == -1:
                    break

            rated = set(train[u])
            rated.add(valid[u][0])
            rated.add(0)
            target = test[u][0]

        elif mode == "valid":
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue

            seq = np.zeros(args.maxlen, dtype=np.int32)
            idx = args.maxlen - 1

            for item in reversed(train[u]):
                seq[idx] = item
                idx -= 1
                if idx == -1:
                    break

            rated = set(train[u])
            rated.add(0)
            target = valid[u][0]

        else:
            raise ValueError("Unknown evaluation mode: %s" % mode)

        all_users.append(u)
        all_seqs.append(seq)
        all_targets.append(target)
        all_rated.append(rated)

    return all_users, all_seqs, all_targets, all_rated


@torch.no_grad()
def _evaluate_fullsort_user_sasrec(
    model,
    dataset,
    args,
    mode="test",
    batch_size=256,
    max_users=None,
    fusion_weights_by_user=None,
):
    model.eval()

    all_users, all_seqs, all_targets, all_rated = _prepare_eval_data_user_sasrec(
        dataset, args, mode=mode, max_users=max_users
    )

    n_valid = len(all_users)
    if n_valid == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    NDCG10 = NDCG20 = NDCG30 = 0.0
    HT10 = HT20 = HT30 = 0.0

    all_items = np.arange(model.item_num + 1, dtype=np.int32)

    for start in range(0, n_valid, batch_size):
        end = min(start + batch_size, n_valid)

        batch_users = np.asarray(all_users[start:end], dtype=np.int32)
        batch_seqs = np.asarray(all_seqs[start:end], dtype=np.int32)
        batch_targets = torch.as_tensor(
            all_targets[start:end], dtype=torch.long, device=model.dev
        )

        if fusion_weights_by_user is None:
            batch_weight_override = None
        else:
            batch_user_tensor = torch.as_tensor(
                batch_users,
                dtype=torch.long,
            )

            batch_weight_override = (
                fusion_weights_by_user[
                    batch_user_tensor
                ]
            )

            if torch.isnan(
                batch_weight_override
            ).any():
                raise ValueError(
                    "Missing shuffled fusion weight for at least "
                    "one evaluated user."
                )

        scores = model.predict(
            batch_users,
            batch_seqs,
            all_items,
            fusion_weight_override=batch_weight_override,
        )

        if scores.dim() != 2:
            raise ValueError(
                "model.predict() must return (B, num_items). Got %s"
                % str(tuple(scores.shape))
            )

        for row in range(end - start):
            target = batch_targets[row].item()
            target_score = scores[row, target].clone()
            rated_items = list(all_rated[start + row])
            scores[row, rated_items] = -float("inf")
            scores[row, target] = target_score

        sorted_items = torch.argsort(scores, dim=1, descending=True)
        target_matches = sorted_items == batch_targets.unsqueeze(1)
        ranks = target_matches.long().argmax(dim=1)

        hit10 = ranks < 10
        hit20 = ranks < 20
        hit30 = ranks < 30

        HT10 += hit10.sum().item()
        HT20 += hit20.sum().item()
        HT30 += hit30.sum().item()

        if hit10.any():
            NDCG10 += (1.0 / torch.log2(ranks[hit10].float() + 2.0)).sum().item()
        if hit20.any():
            NDCG20 += (1.0 / torch.log2(ranks[hit20].float() + 2.0)).sum().item()
        if hit30.any():
            NDCG30 += (1.0 / torch.log2(ranks[hit30].float() + 2.0)).sum().item()
        # sys.stdout.flush()

    return (
        NDCG10 / n_valid,
        NDCG20 / n_valid,
        NDCG30 / n_valid,
        HT10 / n_valid,
        HT20 / n_valid,
        HT30 / n_valid,
    )


def evaluate_batched(
    model,
    dataset,
    args,
    batch_size=256,
    max_users=None,
    fusion_weights_by_user=None,
):
    return _evaluate_fullsort_user_sasrec(
        model,
        dataset,
        args,
        mode="test",
        batch_size=batch_size,
        max_users=max_users,
        fusion_weights_by_user=fusion_weights_by_user,
    )


def evaluate_valid_batched(
    model,
    dataset,
    args,
    batch_size=256,
    max_users=None,
    fusion_weights_by_user=None,
):
    return _evaluate_fullsort_user_sasrec(
        model,
        dataset,
        args,
        mode="valid",
        batch_size=batch_size,
        max_users=max_users,
        fusion_weights_by_user=fusion_weights_by_user,
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


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


@torch.no_grad()
def extract_user_view_scores(model, dataset, args, mode="valid", batch_size=256):
    model.eval()
    all_users, all_seqs, _, _ = _prepare_eval_data_user_sasrec(
        dataset, args, mode=mode, max_users=None
    )

    raw_scores_by_user = torch.full(
        (model.user_num + 1, model.num_fm_features), float("nan"), dtype=torch.float32
    )
    weights_by_user = torch.full(
        (model.user_num + 1, model.num_fm_features), float("nan"), dtype=torch.float32
    )

    for start in range(0, len(all_users), batch_size):
        end = min(start + batch_size, len(all_users))
        batch_users = torch.as_tensor(all_users[start:end], dtype=torch.long)
        batch_seqs = np.asarray(all_seqs[start:end], dtype=np.int32)

        raw_scores, fusion_weights = model.get_final_view_scores(batch_seqs)
        raw_scores_by_user[batch_users] = raw_scores.detach().cpu()
        weights_by_user[batch_users] = fusion_weights.detach().cpu()

    return (
        raw_scores_by_user,
        weights_by_user,
        torch.as_tensor(all_users, dtype=torch.long),
    )


def load_feature_embeddings(dataset, device):
    base = "./datasets"
    return (torch.tensor(np.load(f"{base}/{dataset}_embs_gpt.npy"), dtype=torch.float32, device=device), torch.tensor(np.load(f"{base}/{dataset}_embs_gemini.npy"), dtype=torch.float32, device=device), torch.tensor(np.load(f"{base}/{dataset}_embs_siglip.npy"), dtype=torch.float32, device=device), torch.tensor(np.load(f"{base}/{dataset}_embs_pe.npy"), dtype=torch.float32, device=device))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='toys_games', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--eval_start_epoch', default=30, type=int)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=3, type=int)

    parser.add_argument('--maxlen', default=20, type=int)
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)

    parser.add_argument('--wdecay', default=1e-6, type=float)
    parser.add_argument('--l2_emb', default=1e-6, type=float)
    parser.add_argument('--balance_loss_term', default=1.0, type=float)

    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--result_path', default=None, type=str)
    
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--tau', default=1.0, type=float)
    return parser

# ================================================================
# One training run
# ================================================================

def train_one_run(dataset, X1, X2, X3, X4, args, lr, tau, seed):
    user_train, user_valid, user_test, usernum, itemnum = dataset
    set_seed(seed)

    sampler = WarpSampler(
        user_train, usernum, itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=args.num_workers,
        seed=seed,
    )

    model = ID4FM_UserAdaptive_SASRec(
        usernum,
        itemnum,
        args,
        X1=X1,
        X2=X2,
        X3=X3,
        X4=X4,
        tau=tau,
    ).to(args.device)

    model.pos_emb.weight.data[0, :] = 0
    model.id_item_emb.weight.data[0, :] = 0

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=args.wdecay
    )

    num_batch = (len(user_train) - 1) // args.batch_size + 1
    best_valid_hr30 = -1.0
    best_result = None
    best_epoch = None
    best_state_dict = None
    best_user_view_scores = None
    best_user_view_weights = None
    best_evaluated_user_ids = None

    try:
        for epoch in tqdm(range(1, args.num_epochs + 1)):
            model.train()

            for _ in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                u = np.array(u)
                seq = np.array(seq)
                pos = np.array(pos)
                neg = np.array(neg)

                pos_logits, neg_logits, entropy_loss = model(u, seq, pos, neg)
                pos_labels = torch.ones(pos_logits.shape, device=args.device)
                neg_labels = torch.zeros(neg_logits.shape, device=args.device)

                indices = np.where(pos != 0)
                loss = criterion(
                    pos_logits[indices],
                    pos_labels[indices],
                )

                loss += criterion(
                    neg_logits[indices],
                    neg_labels[indices],
                )

                loss += (
                    args.balance_loss_term
                    * entropy_loss
                )

                for param in model.id_item_emb.parameters():
                    loss += (
                        args.l2_emb
                        * torch.norm(param)
                    )

                # IMPORTANT:
                # Do not modify any trainable parameter in-place between
                # the forward pass and loss.backward(). The ID embedding
                # table participates both in the recommendation logits
                # and the L2 term above, so zeroing row 0 here changes
                # its autograd version and causes:
                #
                #   RuntimeError: variable needed for gradient computation
                #   has been modified by an inplace operation
                #
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Safe: autograd has finished using the forward graph.
                # padding_idx=0 already prevents a gradient on row 0;
                # this post-step cleanup also counteracts AdamW decay.
                with torch.no_grad():
                    model.id_item_emb.weight[
                        0
                    ].zero_()

            if epoch > args.eval_start_epoch:
                model.eval()

                valid_result = evaluate_valid_batched(
                    model, dataset, args, batch_size=args.eval_batch_size
                )
                test_result = evaluate_batched(
                    model, dataset, args, batch_size=args.eval_batch_size
                )

                if valid_result[5] > best_valid_hr30:
                    best_valid_hr30 = valid_result[5]
                    best_result = [tuple(valid_result), tuple(test_result)]
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(model.state_dict())
                    (
                        best_user_view_scores,
                        best_user_view_weights,
                        best_evaluated_user_ids,
                    ) = extract_user_view_scores(
                        model, dataset, args,
                        mode='valid',
                        batch_size=args.eval_batch_size,
                    )
    finally:
        sampler.close()

    if best_result is None:
        raise RuntimeError('No evaluation was performed. Set eval_start_epoch < num_epochs.')

    model.load_state_dict(
        best_state_dict
    )
    model.eval()

    return (
        model,
        best_result,
        best_user_view_scores,
        best_user_view_weights,
        best_evaluated_user_ids,
        best_epoch,
    )


# ================================================================
# Main
# ================================================================

def main(args):
    dataset = data_partition(
        args.dataset
    )

    (
        user_train,
        user_valid,
        user_test,
        usernum,
        itemnum,
    ) = dataset

    avg_seq_len = (
        sum(
            len(user_train[u])
            for u in user_train
        )
        / len(user_train)
    )

    print(
        "average sequence length: "
        "%.2f"
        % avg_seq_len
    )

    X1, X2, X3, X4 = (
        load_feature_embeddings(
            args.dataset,
            args.device,
        )
    )

    stream_names = ["id", "gpt", "gemini", "siglip", "pe"]
    personalized_view_names = ["gpt", "gemini", "siglip", "pe"]

    lr_grid = [
        args.lr
    ]

    tau_grid = [
        args.tau
    ]

    for lr in lr_grid : 
        
        for tau in tau_grid :
            
            print()

            print(
                f"Dataset: {args.dataset} | "
                f"LR: {lr} | "
                f"Tau: {tau}"
            )

            all_performance = []
            all_view_information = []

            for seed_idx in range(
                args.num_seeds
            ):
                seed = (
                    seed_idx + 1
                ) * 2026

                (
                    _,
                    result_set,
                    user_view_scores,
                    user_view_weights,
                    evaluated_user_ids,
                    best_epoch,
                ) = train_one_run(
                    dataset=dataset,
                    X1=X1,
                    X2=X2,
                    X3=X3,
                    X4=X4,
                    args=args,
                    lr=lr,
                    tau=tau,
                    seed=seed,
                )

                all_performance.append(
                    result_set
                )


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

    print("Done")


if __name__ == "__main__":
    main(
        build_parser().parse_args()
    )
