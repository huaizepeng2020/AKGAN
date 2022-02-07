__author__ = "anonymity"

import random

import torch
import dgl
import numpy as np

from time import time
from prettytable import PrettyTable
from tqdm import tqdm
import pickle
import gc

from utils.parser import parse_args
from utils.data_loader import load_data, load_data_nor, load_data_both
from modules.KGIN_subexp1_one import RS_KGA0_att2_subexp1
from utils.evaluate import test2
from utils.helper import early_stopping

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0
n_new_node = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict


def load_obj(name, directory):
    with open(directory + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, train_kg_dict, n_params, graph, graph_UIS = load_data_both(args)
    if device != 'cpu':
        graph = graph.to(device)
        graph_UIS = graph_UIS.to(device)

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    n_params['epoch_num'] = len(train_cf) // args.batch_size + 1
    model = RS_KGA0_att2_subexp1(n_params, args).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        """training"""
        loss, s, cos_loss = 0, 0, 0
        train_s_t = time()

        # for s in tqdm(range(0, len(train_cf), args.batch_size),
        #               desc='epoch:{},batching cf data set'.format(epoch)):
        for s in range(0, len(train_cf), args.batch_size):
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'])

            model.train()

            batch_loss, _, batch_cos_loss = model(batch, graph, graph_UIS)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            cos_loss += batch_cos_loss.item()

        train_e_t = time()

        if epoch % 10 == 9 or epoch == 1:
            """testing"""
            test_s_t = time()
            ret = test2(model, graph, graph_UIS, user_dict, n_params, args)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", 'cos loss', "recall", "ndcg",
                                     "precision",
                                     "hit_ratio"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, cos_loss, ret['recall'], ret['ndcg'],
                 ret['precision'], ret['hit_ratio']]
            )
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop, best_flag = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                                   stopping_step, expected_order='acc',
                                                                                   flag_step=10)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

        else:
            print('using time %.4f, training loss at epoch %d: %.4f' % (
                train_e_t - train_s_t, epoch, loss))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
