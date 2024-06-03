import pdb
import numpy as np
import torch
import torch.nn as nn
import manifolds
import torch.nn.functional as F
import models.encoders as encoders
from utils.helper import default_device
from collections import defaultdict
import pdb

'''
Hyperbolic Graph Learning for Social Recommendation
(1) hyperbolic heterogeneous graph learning
(2) pre-trained hyperbolic social feature enhancement
'''

######################################   Hyperbolic Model  #########################################
def normal_distribution(x):
    mean = torch.mean(x, dim=1)
    std = torch.std(x, dim=1)
    return (x - mean.unsqueeze(1)) * 0.1 / std.unsqueeze(1)

def normal_dis(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) * 0.1 / std

class HGSRModel(nn.Module):
    def __init__(self, users_items, args):
        super(HGSRModel, self).__init__()
        self.args = args
        self.c = torch.tensor([args.c]).to('cuda')
        # self.c = nn.Parameter(torch.tensor([args.c], dtype=torch.float), requires_grad=True).to('cuda') # learnable curve
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.nnodes = args.n_nodes
        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.num_layers = args.num_layers
        self.interest_weight = args.interest_weight
        self.pretrain_type = args.pretrain_type

        ###  parameters in hyperbolic space  ###
        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items, embedding_dim=args.embedding_dim).to('cuda')
        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))
        self.embedding.weight = manifolds.ManifoldParameter(self.embedding.weight, True, self.manifold, self.c)

        ### load pre-trained social embeddings ###
        self._load_pretrained_social_features()


    def _load_pretrained_social_features(self):
        if self.pretrain_type == 'hyperbolic':
            feature_path = './pretrained/' + self.args.dataset + '/hypergnn/' + str(self.args.embedding_dim) + '_dim/'
            user_social_feature = np.load(feature_path + 'H_user_embeddings.npy', allow_pickle=True)
        elif self.pretrain_type == 'euclidean':
            feature_path = './pretrained/' + self.args.dataset + '/euclignn/' + self.args.embedding_dim + '_dim/'
            user_social_feature = np.load(feature_path + 'E_user_embeddings.npy', allow_pickle=True)
        user_social_feature = normal_dis(user_social_feature)
        self.user_social_feature = torch.from_numpy(user_social_feature).to('cuda')


    def manifold_parameters(self, x):
        x.weight = nn.Parameter(self.manifold.expmap0(x.weight, self.c))
        x.weight = manifolds.ManifoldParameter(x.weight, True, self.manifold, self.c)

    def tangent_parameters(self, x):
        y = self.manifold.proj(x.weight, self.c)
        y_tangent = self.manifold.logmap0(y, self.c)
        return y_tangent


    def encode(self, adj_uv, adj_uu):
        adj_uu = adj_uu.to('cuda')
        adj_uv = adj_uv.to('cuda')
        x = self.manifold.proj(self.embedding.weight, self.c)
        x_tangent = self.manifold.logmap0(x, self.c)
        user_feature_tan = self.manifold.proj(self.user_social_feature, self.c)
        user_feature_tan = self.manifold.logmap0(user_feature_tan, self.c)
        # ####################   all embedding   ############################
        z_tangent = torch.cat([user_feature_tan, x_tangent[self.num_users:]], 0)
        all_emb = [torch.cat([x_tangent, z_tangent], 1)]
        #all_emb = [x_tangent]
        for _ in range(self.num_layers):
            cur_emb = self.interest_weight * torch.spmm(adj_uv, all_emb[-1]) + (1-self.interest_weight) * torch.spmm(adj_uu, all_emb[-1])
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
        train_edges = triples[:, [0, 1]]
        false_edges = triples[:, [0, 2]]
        pos_scores = self.decode(embeddings, train_edges)
        neg_scores = self.decode(embeddings, false_edges)
        e_u = embeddings[triples[:, 0]]
        e_i = embeddings[triples[:, 1]]
        e_o = torch.zeros(e_u.shape, dtype=torch.float32).to('cuda')
        e_o[:, 0] = 1
        theta = self.manifold.sqdist(e_u, e_o, self.c) + self.manifold.sqdist(e_i, e_o, self.c) \
                - self.manifold.sqdist(e_u, e_i, self.c)
        # theta = torch.clamp(theta, min=1e-9, max=1e9)
        # scale = (e_u[:, 0] * e_i[:, 0]).view(-1, 1)
        # scale = torch.exp(-d_u/(d_u+d_i)).view(-1, 1)
        # theta = theta * scale
        margin = torch.sigmoid(theta)
        loss = pos_scores - neg_scores + margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss.mean()


    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = h[i, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h[np.arange(num_users, num_users + num_items), :]
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix
