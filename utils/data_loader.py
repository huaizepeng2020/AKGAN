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
    triplets, train_kg_dict,r_num = read_triplets_gui(directory + 'kg_final.txt')

    # edge = np.load('IS_graph_{}.npy'.format(args.dataset))

    print('building the graph ...')
    graph, graph_UIS,relation_num_dict= build_graph(train_cf, triplets)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations),
        'n_kg_train': len(triplets),
        'num_r': r_num,
        'num_r0':relation_num_dict
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, train_kg_dict, n_params, graph, graph_UIS


def load_data_KGE(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    triplets, train_kg_dict = read_triplets_gui(directory + 'kg_final.txt')

    print('building the graph ...')
    graph, graph_UIS = build_graph_KGE(train_cf, triplets)

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
    """build konwledge graph"""
    relation_dict = {}
    relation_num_dict = {}
    for i in range(n_relations):
        idx = np.where(triplets[:, 1] == i)[0]
        node_pair = triplets[:, [0, 2]][idx]
        name = ('item', i, 'item')
        relation_dict[name] = (node_pair[:, 0].tolist(), node_pair[:, 1].tolist())
        relation_num_dict[name] = len(idx)
    graph = dgl.heterograph(relation_dict)

    """build user-item and item-item graph"""
    relation_dict_ui = {}

    name = ('item', 1, 'user')
    relation_dict_ui[name] = (train_data[:, 1], train_data[:, 0])

    name_graph_UIS = {'user': n_users, 'item': n_items}
    graph_UIS = dgl.heterograph(relation_dict_ui, name_graph_UIS)

    return graph, graph_UIS,relation_num_dict


def build_graph_KGE(train_data, triplets):
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

    """build user-item and item-item graph"""
    relation_dict_ui = {}

    name = ('item', 1, 'user')
    relation_dict_ui[name] = (train_data[:, 1], train_data[:, 0])

    name = ('user', 2, 'item')
    relation_dict_ui[name] = (train_data[:, 0], train_data[:, 1])

    name_graph_UIS = {'user': n_users, 'item': n_items}
    graph_UIS = dgl.heterograph(relation_dict_ui, name_graph_UIS)

    return graph, graph_UIS


def build_graph_nor(train_data, triplets):
    """build konwledge graph"""
    graph = dgl.DGLGraph()
    graph.add_edges(torch.LongTensor(triplets[:, 0]),
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

    return graph, graph_UIS


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

    cs = []
    for ii in range(n_relations):
        idx = np.where(triplets[:, 1] == ii)[0]
        cs.append(len(list(set(triplets[idx, 0].tolist()))))
    cs = np.array(cs)
    # cs=cs/np.sum(cs)
    # dim=256
    # cs=cs*dim
    # cs0=np.around(cs)

    return triplets, train_kg_dict, cs
