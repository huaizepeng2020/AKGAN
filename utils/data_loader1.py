import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import dgl
import torch
import multiprocessing
import pickle

import random
from time import time
from collections import defaultdict
import warnings
import scipy, gc
import threading
from collections import Counter

warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    triplets, train_kg_dict = read_triplets_gui(directory + 'kg_final.txt')

    # edge = np.load('IS_graph_{}.npy'.format(args.dataset))

    print('building the graph ...')
    graph, graph_UIS, graph_r = build_graph(train_cf, triplets)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations),
        'num_pre': int(args.num_pre),
        'n_kg_train': len(triplets)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, train_kg_dict, n_params, graph, graph_UIS


def load_data_both(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    triplets, train_kg_dict = read_triplets_gui(directory + 'kg_final.txt')

    # edge = np.load('IS_graph_{}.npy'.format(args.dataset))

    print('building the graph ...')
    graph, graph_AU, graph_UIS = build_graph_user2atr(train_cf, triplets)
    # graph, graph_AU, graph_UIS = build_graph_user2atr_mul(train_cf, triplets)

    # graph, graph_UIS, graph_r = build_graph(train_cf, triplets)

    # print('building the graph ...')
    # graph_nor, _ = build_graph_nor(train_cf, triplets)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations),
        'num_pre': int(args.num_pre),
        'n_kg_train': len(triplets)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, train_kg_dict, n_params, graph, graph_AU, graph_UIS


def load_data_nor(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    triplets, train_kg_dict = read_triplets_gui(directory + 'kg_final.txt')

    # edge = np.load('IS_graph_{}.npy'.format(args.dataset))

    print('building the graph ...')
    graph, graph_UIS = build_graph_nor(train_cf, triplets)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations),
        'num_pre': int(args.num_pre),
        'n_kg_train': len(triplets)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, train_kg_dict, n_params, graph, graph_UIS


def build_graph(train_data, triplets):
    idx = np.where(triplets[:, 2] < n_items)[0]
    all_r = triplets[:, 1][idx]
    new_r_all = np.unique(all_r)

    """build konwledge graph"""
    relation_dict = {}
    for i in range(n_relations):
        idx = np.where(triplets[:, 1] == i)[0]
        node_pair = triplets[:, [0, 2]][idx]
        name = ('item', i, 'item')
        relation_dict[name] = (node_pair[:, 0].tolist(), node_pair[:, 1].tolist())
    graph = dgl.heterograph(relation_dict)

    # relation_deg
    r_deg = []
    for i in range(n_relations):
        idx = np.where(triplets[:, 1] == i)[0]
        data = [1] * len(idx)
        r_M = scipy.sparse.coo_matrix((data, (triplets[idx, 0], triplets[idx, 2])),
                                      shape=(n_entities, n_entities))
        r_deg.append(torch.tensor(np.sum(r_M, axis=0).tolist()))

    r_deg = torch.cat(r_deg, dim=0).permute(1, 0)
    deg_all = torch.sum(r_deg, dim=1)
    graph.ndata['r_deg'] = r_deg
    graph.ndata['deg_all'] = deg_all

    """build user-item and item-item graph"""
    relation_dict_ui = {}

    name = ('item', 1, 'user')
    relation_dict_ui[name] = (train_data[:, 1], train_data[:, 0])

    name_graph_UIS = {'user': n_users, 'item': n_items}
    graph_UIS = dgl.heterograph(relation_dict_ui, name_graph_UIS)

    # graph_UIS.nodes['user'].data['id'] = torch.arange(n_users, dtype=torch.long)
    # graph_UIS.nodes['item'].data['id'] = torch.arange(n_items, dtype=torch.long)

    a = triplets[:, 1:].tolist()
    aa = []
    for i in a:
        aa.append(str(i))
    aa = list(set(aa))
    a = []
    for i in tqdm(aa):
        a.append(eval(i))
    a = np.array(a)

    """build konwledge graph"""
    graph_r = dgl.DGLGraph()
    graph_r.add_edges(torch.LongTensor([n_entities] * a.shape[0]),
                      torch.LongTensor(a[:, 1]),
                      {'type': torch.LongTensor(a[:, 0])})

    return graph, graph_UIS, graph_r


def build_graph_nor(train_data, triplets):
    """build konwledge graph"""
    graph = dgl.DGLGraph()
    graph.add_edges(torch.LongTensor(triplets[:, 0]),
                    torch.LongTensor(triplets[:, 2]),
                    {'type': torch.LongTensor(triplets[:, 1])})
    graph.ndata['id'] = torch.tensor(list(range(n_entities)))

    """build user-item and item-item graph"""
    relation_dict_ui = {}

    name = ('item', 1, 'user')
    relation_dict_ui[name] = (train_data[:, 1], train_data[:, 0])

    name_graph_UIS = {'user': n_users, 'item': n_items}
    graph_UIS = dgl.heterograph(relation_dict_ui, name_graph_UIS)

    data = [1] * len(train_data)
    r_M = scipy.sparse.coo_matrix((data, (train_data[:, 0], train_data[:, 1])),
                                  shape=(n_users, n_items))
    a = torch.tensor(np.sum(r_M, axis=1).tolist())
    graph_UIS.nodes['user'].data['deg'] = a

    return graph, graph_UIS


def build_graph_user2atr(train_data, triplets):
    # item_nei = []
    # for i in tqdm(range(n_items)):
    #     idx = np.where(triplets[:, 2] == i)[0]
    #     item_nei.append(triplets[idx].tolist())
    #
    # user_atr = []
    # for i in tqdm(range(n_users)):
    #     idx = np.where(train_data[:, 0] == i)[0]
    #     a = []
    #     for j in train_data[idx][:, 1]:
    #         a += item_nei[j]
    #
    #     a = np.array(a)
    #     a[:, 2] = i
    #     a = a.tolist()
    #     aa = [str(k) for k in a]
    #     result = Counter(aa)
    #     re_f = []
    #     for k in result.keys():
    #         re_f.append(eval(k) + [result[k]])
    #     user_atr += re_f
    #
    # user_atr = np.array(user_atr)
    # user_atr[:, 2] += n_entities

    """build konwledge graph"""
    graph = dgl.DGLGraph()
    graph.add_edges(torch.LongTensor(triplets[:, 0]),
                    torch.LongTensor(triplets[:, 2]),
                    {'type': torch.LongTensor(triplets[:, 1])})

    # graph1 = dgl.DGLGraph()
    # graph1.add_nodes(n_entities + n_users)
    # graph1.add_edges(torch.LongTensor(user_atr[:, 0]),
    #                  torch.LongTensor(user_atr[:, 2]),
    #                  {'type': torch.LongTensor(user_atr[:, 1]), 'weight': torch.LongTensor(user_atr[:, 3])})
    graph1 = dgl.DGLGraph()
    graph1.add_edges(torch.LongTensor(triplets[:, 0]),
                     torch.LongTensor(triplets[:, 2]),
                     {'type': torch.LongTensor(triplets[:, 1])})

    """build user-item and item-item graph"""
    relation_dict_ui = {}

    name = ('item', 1, 'user')
    relation_dict_ui[name] = (train_data[:, 1], train_data[:, 0])

    name_graph_UIS = {'user': n_users, 'item': n_items}
    graph_UIS = dgl.heterograph(relation_dict_ui, name_graph_UIS)

    data = [1] * len(train_data)
    r_M = scipy.sparse.coo_matrix((data, (train_data[:, 0], train_data[:, 1])),
                                  shape=(n_users, n_items))
    a = torch.tensor(np.sum(r_M, axis=1).tolist())
    graph_UIS.nodes['user'].data['deg'] = a

    return graph, graph1, graph_UIS


def build_graph_user2atr_mul(train_data, triplets):
    num_G = 3
    item_nei = []
    for i in tqdm(range(n_items)):
        idx = np.where(triplets[:, 2] == i)[0]
        item_nei.append(triplets[idx].tolist())

    user_atr = []
    for t in range(num_G):
        user_atr.append([])

    idx_UI = []
    for t in range(num_G):
        idx_UI.append([])

    for i in tqdm(range(n_users)):
        idx = np.where(train_data[:, 0] == i)[0]

        a = []
        for t in range(num_G):
            a.append([])

        num_idx = len(idx) // num_G
        idx_list = []
        for t in range(num_G):
            idx_list.append(num_idx * t)
        idx_list.append(len(idx))

        for t in range(num_G):
            idx_UI[t] += idx[idx_list[t]:idx_list[t + 1]].tolist()

            for j in train_data[idx[idx_list[t]:idx_list[t + 1]]][:, 1]:
                a[t] += item_nei[j]
            a[t] = np.array(a[t])
            a[t][:, 2] = i
            a[t] = a[t].tolist()
            aa = [str(k) for k in a[t]]
            result = Counter(aa)
            re_f = []
            for k in result.keys():
                re_f.append(eval(k) + [result[k]])
            user_atr[t] += re_f

    for t in range(num_G):
        user_atr[t] = np.array(user_atr[t])
        user_atr[t][:, 2] += n_entities

    """build konwledge graph"""
    graph = dgl.DGLGraph()
    graph.add_edges(torch.LongTensor(triplets[:, 0]),
                    torch.LongTensor(triplets[:, 2]),
                    {'type': torch.LongTensor(triplets[:, 1])})

    graph1_all = []
    for t in range(num_G):
        graph1 = dgl.DGLGraph()
        graph1.add_nodes(n_entities + n_users)
        graph1.add_edges(torch.LongTensor(user_atr[t][:, 0]),
                         torch.LongTensor(user_atr[t][:, 2]),
                         {'type': torch.LongTensor(user_atr[t][:, 1]), 'weight': torch.LongTensor(user_atr[t][:, 3])})
        graph1_all.append(graph1)

    """build user-item and item-item graph"""
    graph_UIS_all = []
    for t in range(num_G):
        relation_dict_ui = {}

        name = ('item', 1, 'user')
        relation_dict_ui[name] = (train_data[idx_UI[t]][:, 1], train_data[idx_UI[t]][:, 0])

        name_graph_UIS = {'user': n_users, 'item': n_items}
        graph_UIS = dgl.heterograph(relation_dict_ui, name_graph_UIS)

        graph_UIS_all.append(graph_UIS)

    return graph, graph1_all, graph_UIS_all


def cal_deg(edges):
    a = torch.tensor([1] * len(edges)).unsqueeze(1)
    return {'deg1': a}


def build_graph0(train_data, triplets):
    idx = np.where(triplets[:, 2] < n_items)[0]
    all_r = triplets[:, 1][idx]
    new_r_all = np.unique(all_r)

    """build konwledge graph"""
    graph = dgl.DGLGraph()
    graph.add_edges(torch.LongTensor(triplets[:, 0]),
                    torch.LongTensor(triplets[:, 2]),
                    {'type': torch.LongTensor(triplets[:, 1])})

    # relation_deg
    r_deg = []
    for i in range(n_relations):
        idx = np.where(triplets[:, 1] == i)[0]
        data = [1] * len(idx)
        r_M = scipy.sparse.coo_matrix((data, (triplets[idx, 0], triplets[idx, 2])),
                                      shape=(n_entities, n_entities))
        r_deg.append(torch.tensor(np.sum(r_M, axis=0).tolist()))

    r_deg = torch.cat(r_deg, dim=0).permute(1, 0)
    deg_all = torch.sum(r_deg, dim=1)
    graph.ndata['r_deg'] = r_deg
    graph.ndata['deg_all'] = deg_all
    # torch.sum(deg_all)
    # len(triplets)

    """build user-item and item-item graph"""
    relation_dict_ui = {}

    # name = ('item', 0, 'item')
    # edge1 = edge.copy()
    # edge1[:, 0] = edge[:, 1]
    # edge1[:, 1] = edge[:, 0]
    # edges = np.concatenate((edge, edge1), axis=0)
    # relation_dict_ui[name] = (edges[:, 0], edges[:, 1])
    #
    # name = ('user', 1, 'item')
    # relation_dict_ui[name] = (train_data[:, 0], train_data[:, 1])

    name = ('item', 2, 'user')
    relation_dict_ui[name] = (train_data[:, 1], train_data[:, 0])

    name_graph_UIS = {'user': n_users, 'item': n_items}

    graph_UIS = dgl.heterograph(relation_dict_ui, name_graph_UIS)

    graph_UIS.nodes['user'].data['id'] = torch.arange(n_users, dtype=torch.long)
    graph_UIS.nodes['item'].data['id'] = torch.arange(n_items, dtype=torch.long)

    return graph, graph_UIS, new_r_all


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def read_triplets_gui(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # inv_triplets_np[:, 1] = can_triplets_np[:, 1]
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        # can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    train_kg_dict = defaultdict(list)
    for row in triplets:
        h, r, t = row
        train_kg_dict[h].append((t, r))

    return triplets, train_kg_dict
