import pdb
import numpy as np
import torch
import torch.nn as nn
import manifolds
import torch.nn.functional as F


class HGSSModel(nn.Module):
    def __init__(self, num_users, args):
        super(HGSSModel, self).__init__()
        self.c = torch.tensor([args.c]).to('cuda')
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.num_users = num_users
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.num_layers = args.num_layers
        self.args = args
        self.embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=args.embedding_dim).to('cuda')
        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        ###  投影到双曲空间的参数
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))
        self.embedding.weight = manifolds.ManifoldParameter(self.embedding.weight, True, self.manifold, self.c) ###  加入黎曼优化器的参数？


    def encode(self, adj):
        adj = adj.to('cuda')
        x = self.manifold.proj(self.embedding.weight, self.c)
        x_tangent = self.manifold.logmap0(x, self.c)
        all_emb = [x_tangent]
        for _ in range(self.num_layers):
            cur_emb = torch.spmm(adj, all_emb[-1])
            all_emb.append(cur_emb)
        y = sum(all_emb[1:])
        y = self.manifold.expmap0(y, self.c)
        y = self.manifold.proj(y, self.c)
        return y


    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        return sqdist


    def compute_loss(self, embeddings, triples):
        train_edges = triples[:, [0, 1]]
        sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]
        pos_scores = self.decode(embeddings, train_edges)
        neg_scores_list = [self.decode(embeddings, sampled_false_edges) for sampled_false_edges in sampled_false_edges_list]
        neg_scores = torch.cat(neg_scores_list, dim=1)
        loss = pos_scores - neg_scores + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss


    def compute_loss_adaptive_margin(self, embeddings, triples):
        # pdb.set_trace()
        train_edges = triples[:, [0, 1]]
        false_edges = triples[:, [0, 2]]
        pos_scores = self.decode(embeddings, train_edges)
        neg_scores = self.decode(embeddings, false_edges)
        e_u = embeddings[triples[:, 0]]
        e_i = embeddings[triples[:, 1]]
        e_o = torch.zeros(e_u.shape, dtype=torch.float32).to('cuda')
        e_o[:, 0] = 1
        theta = self.manifold.sqdist(e_u, e_o, self.c) + self.manifold.sqdist(e_i, e_o, self.c) - self.manifold.sqdist(e_u, e_i, self.c)
        scale = (e_u[:, 0] * e_i[:, 0]).view(-1, 1)
        theta = theta / scale
        margin = torch.sigmoid(theta) #/ 5
        loss = pos_scores - neg_scores + margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss


    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_users
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = h[i, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix