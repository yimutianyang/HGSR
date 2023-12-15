import os, pdb
from time import time
import torch
from shutil import copyfile
from models.hgsr import HGSRModel
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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    ###  dataset parameters   ###
    parser = argparse.ArgumentParser(description='HGSR+Social Pretraining')
    parser.add_argument('--dataset', type=str, default='flickr', help='which data to use')
    parser.add_argument('--num_neg', type=int, default=1, help='number of negative samples')
    parser.add_argument('--norm_adj', type=str, default='True', help=' ')

    ###  training parameters  ###
    parser.add_argument('--log', type=str, default='True', help='write log or not?')
    parser.add_argument('--runid', type=str, default='0', help='current log id')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='l2 regularization strength')
    parser.add_argument('--momentum', type=float, default=0.95, help='')
    parser.add_argument('--seed', type=int, default=1234, help='seed for data split and training')
    parser.add_argument('--log_freq', type=int, default=1,
                        help='how often to compute print train/val metrics (in epochs)')
    parser.add_argument('--eval_freq', type=int, default=10, help='how often to compute val metrics (in epochs)')

    ###  model parameters  ###
    parser.add_argument('--pretrain_type', type=str, default='hyperbolic', help='social pretraining in hyperbolic or euclidean space')
    parser.add_argument('--negative_sampling', type=str, default='random', help='negative sampling strategies, popular or random')
    parser.add_argument('--c', type=float, default=1, help='hyperbolic radius, set to None for trainable curvature')
    parser.add_argument('--network', type=str, default='resSumGCN', help='choice of StackGCNs, plainGCN, resSumGCN')
    parser.add_argument('--num_layers', type=int, default=3, help='number of hidden layers in gcn encoder')
    parser.add_argument('--embedding_dim', type=int, default=64, help='latent embedding dimension')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for init')
    parser.add_argument('--margin', type=float, default=0.1, help='margin value in the metric learning loss')
    parser.add_argument('--interest_weight', type=float, default=0.8,
                        help='balance weight for social aggregation on node representation')
    return parser.parse_args()


def train(model):
    adj_social, social_degree, inter_degree, item_degree = data.hetero_graph()
    adj_input = data.agcn_adj_matrix()
    optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Total number of parameters: {tot_params}")
    num_pairs = data.adj_train.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1
    print(num_batches)
    ###  ========================== Train model =================================  ###
    max_recall, max_ndcg = 0, 0
    for epoch in range(args.epochs):
        avg_loss = 0.
        # === batch training === #
        t1 = time()
        data_iter = training_data.sample_batch_data('recommendation', args.negative_sampling)  ### popularity sampling or random sampling
        for triples in data_iter:
            model.train()
            optimizer.zero_grad()
            if args.dataset == 'flickr':
                embeddings = model.encode(adj_input, adj_social)
            else:
                embeddings = model.encode(data.adj_train_norm, adj_social)
            # train_loss = model.compute_loss(embeddings, triples)  ### margin loss
            train_loss = model.compute_loss_adaptive_margin(embeddings, triples)  ### Adaptive margin loss
            train_loss.backward()
            optimizer.step()
            avg_loss += train_loss / num_batches
        # === evaluate at the end of each batch === #
        t2 = time()
        avg_loss = avg_loss.detach().cpu().numpy()
        writer.add_scalar('loss', avg_loss, epoch)
        log.write('Train:{:3d}, Loss:{:.4f}, Time:{:.4f}\n'.format(epoch, avg_loss, t2 - t1))

        if (epoch + 1) % args.eval_freq == 0 and epoch > 0:
            model.eval()
            start = time()
            if args.dataset == 'flickr':
                embeddings = model.encode(adj_input, adj_social)
            else:
                embeddings = model.encode(data.adj_train_norm, adj_social)
            print(time() - start)
            pred_matrix = model.predict(embeddings, data)
            print(time() - start)
            recall, ndcg = evaluate(data.test_dict, data.train_dict, [5, 10, 20], pred_matrix, data.test_dict.keys())
            log.write('Time:{:.4f}, Recall@10:{:.4f}, NDCG@10:{:.4f}, Recall@20:{:.4f}, NDCG@20:{:.4f}\n'.format(
                time() - start, recall[10], ndcg[10], recall[20], ndcg[20]))
            max_ndcg = max(max_ndcg, ndcg[20])
            writer.add_scalar('Recall', recall[20], epoch)
            writer.add_scalar('NDCG', ndcg[20], epoch)
            if max_ndcg == ndcg[20]:
                best_model = model_save_path + 'model.pt'
                torch.save(model.state_dict(), best_model)
    # sampler.close()
    model.load_state_dict(torch.load(best_model))
    model.eval()
    if args.dataset == 'flickr':
        embeddings = model.encode(adj_input, adj_social)
    else:
        embeddings = model.encode(data.adj_train_norm, adj_social)
    pred_matrix = model.predict(embeddings, data)
    for key in [5, 10, 20, 30, 40, 50]:
        recall, ndcg = evaluate(data.test_dict, data.train_dict, [key], pred_matrix, data.test_dict.keys())
        log.write('Topk:{:3d}, Recall:{:.4f}, NDCG:{:.4f}\n'.format(key, recall[key], ndcg[key]))


def test(model):
    adj_social, social_degree, inter_degree, item_degree = data.hetero_graph()
    adj_input = data.agcn_adj_matrix()
    best_model = model_save_path + 'model.pt'
    model.load_state_dict(torch.load(best_model))
    model.eval()
    if args.dataset in ['epinions', 'ciao', 'dianping']:
        embeddings = model.encode(data.adj_train_norm, adj_social)
    elif args.dataset in ['flickr']:
        embeddings = model.encode(adj_input, adj_social)  # flickr
    pred_matrix = model.predict(embeddings, data)
    for topk in [10, 20, 30, 40, 50]:
        recall, ndcg = evaluate(data.test_dict, data.train_dict, [topk], pred_matrix, data.test_dict.keys())  ### all testdata
        log.write('Topk:{:3d}, Recall:{:.4f}, NDCG:{:.4f}\n'.format(topk, recall[topk], ndcg[topk]))

    u1, u2, u3, u4 = data.split_user_group([32, 64, 128])
    recall_1, ndcg_1 = evaluate(data.test_dict, data.train_dict, [20], pred_matrix, u1)
    recall_2, ndcg_2 = evaluate(data.test_dict, data.train_dict, [20], pred_matrix, u2)
    recall_3, ndcg_3 = evaluate(data.test_dict, data.train_dict, [20], pred_matrix, u3)
    recall_4, ndcg_4 = evaluate(data.test_dict, data.train_dict, [20], pred_matrix, u4)
    log.write('Recall_U1:{:.4f}, Recall_U2:{:.4f}, Recall_U3:{:.4f}, Recall_U4:{:.4f}\n'.format(
        recall_1[20], recall_2[20], recall_3[20], recall_4[20]))
    log.write('NDCG_U1:{:.4f}, NDCG_U2:{:.4f}, NDCG_U3:{:.4f}, NDCG_U4:{:.4f}\n'.format(
        ndcg_1[20], ndcg_2[20], ndcg_3[20], ndcg_4[20]))


if __name__ == '__main__':
    args = parse_args()
    record_path = './saved/' + args.dataset + '/' + args.runid + '/'
    model_save_path = record_path + 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    print('model saved path is', model_save_path)
    copyfile('run_hgsr.py', record_path + 'run_hgsr.py')
    copyfile('./models/hgsr.py', record_path + 'hgsr.py')
    copyfile('./utils/data_generator.py', record_path + 'data_generator.py')
    if args.log:
        log = Logger(record_path)
        for arg in vars(args):
            log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    writer = SummaryWriter(model_save_path + 'log')
    set_seed(args.seed)
    data = Data(args.dataset, args.norm_adj)
    args.n_nodes = data.num_users + data.num_items
    args.feat_dim = args.embedding_dim
    training_data = Popularity_Sampler(data.train_dict, data.num_users, data.num_items, neg_sample=1,
                                       batch_size=args.batch_size)
    model = HGSRModel((data.num_users, data.num_items), args)
    model = model.to(default_device())

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print('model is running on', next(model.parameters()).device)
    train(model)
    test(model)
