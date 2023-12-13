import os, pdb
from time import time
import traceback
import torch
from shutil import copyfile
from models.hgss import HGSSModel
from rgd.rsgd import RiemannianSGD
from utils.data_generator import Data
from utils.helper import default_device, set_seed
from utils.log import Logger
import sys
sys.path.append('../')
from evaluator.evaluate import *
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.populairty_sampler import Popularity_Sampler
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



def parse_args():
    ###  dataset parameters   ###
    parser = argparse.ArgumentParser(description='Hyperbolic Social Pretraining')
    parser.add_argument('--dataset', type=str, default='ciao', help='which data to use')
    parser.add_argument('--num_neg', type=int, default=1, help='number of negative samples')
    parser.add_argument('--norm_adj', type=str, default='True', help=' ')

    ###  training parameters  ###
    parser.add_argument('--log', type=str, default='True', help='write log or not?')
    parser.add_argument('--runid', type=str, default='0', help='current log id')
    parser.add_argument('--epochs', type=int, default=600, help='maximum number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='l2 regularization strength')
    parser.add_argument('--momentum', type=float, default=0.95, help='')
    parser.add_argument('--seed', type=int, default=1234, help='seed for data split and training')
    parser.add_argument('--log_freq', type=int, default=1, help='how often to compute print train/val metrics (in epochs)')
    parser.add_argument('--eval_freq', type=int, default=20, help='how often to compute val metrics (in epochs)')

    ###  model parameters  ###
    parser.add_argument('--c', type=float, default=1, help='hyperbolic radius, set to None for trainable curvature')
    parser.add_argument('--network', type=str, default='resSumGCN', help='choice of StackGCNs, plainGCN, resSumGCN')
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden layers in gcn encoder')
    parser.add_argument('--embedding_dim', type=int, default=64, help='latent embedding dimension')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for init')
    parser.add_argument('--margin', type=float, default=0.1, help='margin value in the metric learning loss')
    return parser.parse_args()


def split_data(user_users):
    traindata, testdata = defaultdict(set), defaultdict(set)
    for u, users in user_users.items():
        user_users[u] = list(user_users[u])
        random.shuffle(user_users[u])
        tmp_length = len(user_users[u])
        if tmp_length > 0 and tmp_length < 10:
            first_length = 1
        elif tmp_length > 0:
            first_length = int(tmp_length*0.2)
        testdata[u] = user_users[u][: first_length]
        traindata[u] = user_users[u][first_length: ]
    return traindata, testdata



def generate_social_adj(traindata, num_users):
    adj_indices, adj_values = [], []
    for u, users in traindata.items():
        len_u = len(users) #+ 1
        # adj_indices.append([u, u])
        # adj_values.append(1.0 / len_u)
        for v in users:
            adj_indices.append([u, v])
            adj_values.append(1.0 / len_u)
    adj_indices = np.asarray(adj_indices).T
    adj_values = np.asarray(adj_values)
    graph = torch.sparse.FloatTensor(torch.LongTensor(adj_indices), torch.FloatTensor(adj_values), [num_users, num_users])
    return graph



def train(model):
    social_graph = generate_social_adj(traindata, data.num_users)
    optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr, \
                              weight_decay=args.weight_decay, momentum=args.momentum)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Total number of parameters: {tot_params}")

    # === Train model
    max_recall, max_ndcg = 0, 0
    for epoch in range(args.epochs):
        avg_loss = 0.
        # === batch training
        t1 = time()
        data_iter = training_data.sample_batch_data('social', 'random')
        k = 0
        for triples in data_iter:
            k += 1
            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(social_graph)
            # train_loss = model.compute_loss(embeddings, triples)
            train_loss = model.compute_loss_adaptive_margin(embeddings, triples)
            train_loss.backward()
            optimizer.step()
            avg_loss += train_loss
        # === evaluate at the end of each batch
        t2 = time()
        avg_loss = avg_loss.detach().cpu().numpy() / k
        writer.add_scalar('loss', avg_loss, epoch)
        log.write('Train:{:3d}, Loss:{:.4f}, Time:{:.4f}\n'.format(epoch, avg_loss, t2-t1))

        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            start = time()
            embeddings = model.encode(social_graph)
            print(time() - start)
            pred_matrix = model.predict(embeddings, data)
            print(time() - start)
            recall, ndcg = evaluate(testdata, traindata, [20], pred_matrix, testdata.keys())
            log.write('Time:{:.4f}, Recall@20:{:.4f}, NDCG@20:{:.4f}\n\n'.format(time()-start, recall[20], ndcg[20]))
            max_ndcg = max(max_ndcg, ndcg[20])
            writer.add_scalar('Recall', recall[20], epoch)
            writer.add_scalar('NDCG', ndcg[20], epoch)
            if max_ndcg == ndcg[20]:
                best_model = model_save_path + 'model.pt'
                torch.save(model.state_dict(), best_model)

    model.load_state_dict(torch.load(best_model))
    model.eval()
    embeddings = model.encode(social_graph)
    np.save(record_path + 'H_user_embeddings.npy', embeddings.detach().cpu().numpy())


if __name__ == '__main__':
    args = parse_args()
    record_path = './pretrained/' + args.dataset + '/hypergnn/' + str(args.embedding_dim) + '_dim/'
    model_save_path = record_path + 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    print('model saved path is', model_save_path)
    copyfile('run_pretrain_social.py', record_path + 'run_pretrain_social.py')
    copyfile('./models/hgss.py', record_path + 'hgss.py')
    if args.log:
        log = Logger(record_path)
        for arg in vars(args):
            log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    writer = SummaryWriter(model_save_path + 'log')
    set_seed(args.seed)
    data = Data(args.dataset, args.norm_adj)
    args.feat_dim = args.embedding_dim
    traindata, testdata = split_data(data.user_users)
    training_data = Popularity_Sampler(traindata, data.num_users, data.num_users, neg_sample=1, batch_size=args.batch_size)
    model = HGSSModel(data.num_users, args)
    model = model.to(default_device())

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print('model is running on', next(model.parameters()).device)
    train(model)
