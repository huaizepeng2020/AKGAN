from .metrics import *
from .parser import parse_args, parse_args1, parse_args10

import torch
import numpy as np
import multiprocessing
import heapq
import dgl
from tqdm import tqdm
from time import time

cores = multiprocessing.cpu_count() // 6

args = parse_args10()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(model, graph, graph_atr2pre, graph_pre2user, user_dict, n_params):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    with torch.no_grad():
        entity_gcn_emb, user_gcn_emb = model.generate(graph, graph_atr2pre, graph_pre2user)

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
            u_g_embeddings = user_gcn_emb[user_batch]

            if batch_test_flag:
                # batch-item test
                n_item_batchs = n_items // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), n_items))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                    item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(device)
                    i_g_embddings = entity_gcn_emb[item_batch]

                    i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == n_items
            else:
                # all-item test
                item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
                i_g_embddings = entity_gcn_emb[item_batch]
                rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

            user_batch_rating_uid = zip(rate_batch, user_list_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result


def test1(model, graph, graph_user, user_dict, n_params):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    with torch.no_grad():
        entity_gcn_emb, user_gcn_emb = model.generate(graph, graph_user)

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
            u_g_embeddings = user_gcn_emb[user_batch]

            if batch_test_flag:
                # batch-item test
                n_item_batchs = n_items // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), n_items))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                    item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(device)
                    i_g_embddings = entity_gcn_emb[item_batch]

                    i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == n_items
            else:
                # all-item test
                item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
                i_g_embddings = entity_gcn_emb[item_batch]
                rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

            user_batch_rating_uid = zip(rate_batch, user_list_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result


def test2(model, graph, user_pre_r, user_pre_node, user_dict, n_params):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    with torch.no_grad():
        entity_gcn_emb, user_gcn_emb = model.generate(graph, user_pre_r, user_pre_node)

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
            u_g_embeddings = user_gcn_emb[user_batch]

            if batch_test_flag:
                # batch-item test
                n_item_batchs = n_items // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), n_items))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                    item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(device)
                    i_g_embddings = entity_gcn_emb[item_batch]

                    i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == n_items
            else:
                # all-item test
                item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
                i_g_embddings = entity_gcn_emb[item_batch]
                rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

            user_batch_rating_uid = zip(rate_batch, user_list_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result


def test3(model, graph, graph_i2u_1, user_dict, n_params):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    with torch.no_grad():

        user_gcn_emb = []
        if n_users // args.test_batch_size > 0:
            for i in tqdm(range(n_users // args.test_batch_size)):
                if i < n_users // args.test_batch_size - 1:
                    sub_idx = list(range(i * args.test_batch_size, (i + 1) * args.test_batch_size))
                    sub_idx1 = list(range(0, args.test_batch_size))
                    sub_idx = (torch.tensor(sub_idx, device=device), torch.tensor(sub_idx1, device=device))
                else:
                    sub_idx = list(range(i * args.test_batch_size, n_users))
                    sub_idx1 = list(range(0, args.test_batch_size + n_users % args.test_batch_size))
                    sub_idx = (torch.tensor(sub_idx, device=device), torch.tensor(sub_idx1, device=device))

                # graph = dgl.node_subgraph(graph, {'item': list(range(n_entities)), 'user': sub_idx[0]})
                graph_i2u_1_sub = dgl.node_subgraph(graph_i2u_1, {'user': sub_idx[0], 'item': list(range(n_entities))})

                entity_gcn_emb, user_batch = model.generate(graph, graph_i2u_1_sub,
                                                            sub_idx,
                                                            user_dict['train_user_set'])
                user_gcn_emb.append(user_batch)
                torch.cuda.empty_cache()
            user_gcn_emb = torch.cat(user_gcn_emb, dim=0)
        else:
            sub_idx = list(range(0, n_users))
            sub_idx1 = list(range(0, n_users))
            sub_idx = (torch.tensor(sub_idx, device=device), torch.tensor(sub_idx1, device=device))

            # graph = dgl.node_subgraph(graph, {'item': list(range(n_entities)), 'user': sub_idx[0]})
            graph_i2u_1_sub = dgl.node_subgraph(graph_i2u_1, {'user': sub_idx[0], 'item': list(range(n_entities))})

            entity_gcn_emb, user_batch = model.generate(graph, graph_i2u_1_sub,
                                                        sub_idx,
                                                        user_dict['train_user_set'])

        # sub_idx = list(range(n_users))
        # sub_idx1 = list(range(n_users))
        # sub_idx = (torch.tensor(sub_idx, device=device), torch.tensor(sub_idx1, device=device))
        # entity_gcn_emb, user_gcn_emb = model.generate(graph, graph_i2u, graph_u2u, graph_i2u_0, sub_idx,
        #                                               user_dict['train_user_set'])

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
            u_g_embeddings = user_gcn_emb[user_batch]

            if batch_test_flag:
                # batch-item test
                n_item_batchs = n_items // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), n_items))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                    item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(device)
                    i_g_embddings = entity_gcn_emb[item_batch]

                    i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == n_items
            else:
                # all-item test
                item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
                i_g_embddings = entity_gcn_emb[item_batch]
                rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

            user_batch_rating_uid = zip(rate_batch, user_list_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result


def test3_1(model, graph, graph_i2u, graph_u2u, graph_i2u_0, graph_i2u_1, user_dict, n_params):
    args = parse_args1()
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    with torch.no_grad():

        user_gcn_emb = []
        for i in tqdm(range(n_users // args.test_batch_size)):
            if i < n_users // args.test_batch_size - 1:
                sub_idx = list(range(i * args.test_batch_size, (i + 1) * args.test_batch_size))
                sub_idx1 = list(range(0, args.test_batch_size))
                sub_idx = (torch.tensor(sub_idx, device=device), torch.tensor(sub_idx1, device=device))
            else:
                sub_idx = list(range(i * args.test_batch_size, n_users))
                sub_idx1 = list(range(0, args.test_batch_size + n_users % args.test_batch_size))
                sub_idx = (torch.tensor(sub_idx, device=device), torch.tensor(sub_idx1, device=device))
            # graph_i2u_0_sub = dgl.node_subgraph(graph_i2u_0, {'user': sub_idx[0], 'item': list(range(n_items))})

            graph_i2u_1_sub = dgl.node_subgraph(graph_i2u_1, {'user': sub_idx[0], 'item': list(range(n_entities))})

            # sub_idx_i2u = {}
            # for ii in range(n_relations):
            #     sub_idx_i2u['user{}'.format(ii)] = sub_idx[0]
            #     sub_idx_i2u['item{}'.format(ii)] = list(range(n_items))
            # graph_i2u_sub = dgl.node_subgraph(graph_i2u, sub_idx_i2u)

            entity_gcn_emb, user_batch = model.generate(graph, graph_i2u, graph_u2u, graph_i2u_0,
                                                        graph_i2u_1_sub,
                                                        sub_idx,
                                                        user_dict['train_user_set'])
            user_gcn_emb.append(user_batch)
            torch.cuda.empty_cache()
        user_gcn_emb = torch.cat(user_gcn_emb, dim=0)

        # sub_idx = list(range(n_users))
        # sub_idx1 = list(range(n_users))
        # sub_idx = (torch.tensor(sub_idx, device=device), torch.tensor(sub_idx1, device=device))
        # entity_gcn_emb, user_gcn_emb = model.generate(graph, graph_i2u, graph_u2u, graph_i2u_0, sub_idx,
        #                                               user_dict['train_user_set'])

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
            u_g_embeddings = user_gcn_emb[user_batch]

            if batch_test_flag:
                # batch-item test
                n_item_batchs = n_items // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), n_items))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                    item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(device)
                    i_g_embddings = entity_gcn_emb[item_batch]

                    i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == n_items
            else:
                # all-item test
                item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
                i_g_embddings = entity_gcn_emb[item_batch]
                rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

            user_batch_rating_uid = zip(rate_batch, user_list_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result


def test10(model, graph, graph_i2u, graph_u2u, graph_i2u_0, graph_i2u_1, user_dict, n_params):
    args = parse_args10()
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    with torch.no_grad():

        user_gcn_emb = []
        for i in tqdm(range(n_users // args.test_batch_size)):
            if i < n_users // args.test_batch_size - 1:
                sub_idx = list(range(i * args.test_batch_size, (i + 1) * args.test_batch_size))
                sub_idx1 = list(range(0, args.test_batch_size))
                sub_idx = (torch.tensor(sub_idx, device=device), torch.tensor(sub_idx1, device=device))
            else:
                sub_idx = list(range(i * args.test_batch_size, n_users))
                sub_idx1 = list(range(0, args.test_batch_size + n_users % args.test_batch_size))
                sub_idx = (torch.tensor(sub_idx, device=device), torch.tensor(sub_idx1, device=device))
            graph_i2u_0_sub = dgl.node_subgraph(graph_i2u_0, {'user': sub_idx[0], 'item': list(range(n_items))})

            # graph_i2u_1_sub = dgl.node_subgraph(graph_i2u_1, {'user': sub_idx[0], 'item': list(range(n_entities))})

            # sub_idx_i2u = {}
            # for ii in range(n_relations):
            #     sub_idx_i2u['user{}'.format(ii)] = sub_idx[0]
            #     sub_idx_i2u['item{}'.format(ii)] = list(range(n_items))
            # graph_i2u_sub = dgl.node_subgraph(graph_i2u, sub_idx_i2u)

            entity_gcn_emb, user_batch = model.generate(graph, graph_i2u, graph_u2u, graph_i2u_0_sub,
                                                        graph_i2u_1,
                                                        sub_idx,
                                                        user_dict['train_user_set'])
            user_gcn_emb.append(user_batch)
            torch.cuda.empty_cache()
        user_gcn_emb = torch.cat(user_gcn_emb, dim=0)

        # sub_idx = list(range(n_users))
        # sub_idx1 = list(range(n_users))
        # sub_idx = (torch.tensor(sub_idx, device=device), torch.tensor(sub_idx1, device=device))
        # entity_gcn_emb, user_gcn_emb = model.generate(graph, graph_i2u, graph_u2u, graph_i2u_0, sub_idx,
        #                                               user_dict['train_user_set'])

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
            u_g_embeddings = user_gcn_emb[user_batch]

            if batch_test_flag:
                # batch-item test
                n_item_batchs = n_items // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), n_items))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                    item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(device)
                    i_g_embddings = entity_gcn_emb[item_batch]

                    i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == n_items
            else:
                # all-item test
                item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
                i_g_embddings = entity_gcn_emb[item_batch]
                rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

            user_batch_rating_uid = zip(rate_batch, user_list_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result


def train3(model, graph, graph_i2u, graph_u2u, graph_i2u_0, user_dict, train_cf_pairs, s, n_items):
    model.train()
    batch = get_feed_dict(train_cf_pairs,
                          s, s + args.batch_size,
                          user_dict['train_user_set'], n_items)
    sub_idx1 = torch.unique(batch['users'], return_inverse=True)
    graph_i2u_0_sub = dgl.node_subgraph(graph_i2u_0, {'user': sub_idx1[0], 'item': list(range(n_items))})
    # graph_i2u_0_sub = dgl.in_subgraph(graph_i2u_0, {'user': sub_idx1[0]})

    batch_loss, _, _ = model(batch, graph, graph_i2u, graph_u2u, graph_i2u_0_sub, sub_idx1,
                             user_dict['train_user_set'])
    return batch_loss


def get_feed_dict(train_entity_pairs, start, end, train_user_set, n_items):
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
