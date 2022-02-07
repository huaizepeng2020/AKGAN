__author__ = "anonymity"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter_mean
import dgl
import math
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation


class RS_KGA0_att2_subexp1(nn.Module):
    def __init__(self, data_config, args_config):
        super(RS_KGA0_att2_subexp1, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.n_epoch = data_config['epoch_num']
        self.r_num = data_config['num_r']
        self.r_num0 = data_config['num_r0']

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.emb_size = args_config.dim
        self.emb_size1 = args_config.dim1
        self.k_att = args_config.k_att

        k = (self.emb_size - self.emb_size1) / 5000
        b = self.emb_size1
        cs1 = np.around(k * self.r_num + b)
        cs1 = np.array(cs1, dtype=np.int)
        idx1 = np.where(cs1 > self.emb_size)
        cs1[idx1] = self.emb_size
        self.dim_f = int(np.sum(cs1))
        self.dim_flag = np.array(cs1, dtype=np.int).tolist()


        self.dim_flag1 = [0]
        for i in range(self.n_relations):
            self.dim_flag1.append(self.dim_flag1[-1] + self.dim_flag[i])

        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.all_embed_cf = nn.Parameter(self.all_embed_cf)
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        gain = 1.414

        self.all_embed_cf = initializer(torch.empty(self.n_users + self.n_items, self.dim_f), gain=gain)
        self.all_embed = initializer(torch.empty(self.n_entities, self.dim_f), gain=gain)

        self.W_R = []
        for i in range(self.n_relations):
            a = torch.zeros(1, self.dim_f)
            a[0, self.dim_flag1[i]:self.dim_flag1[i + 1]] += 1
            self.W_R.append(a)
        self.W_R = torch.cat(self.W_R, dim=0).to(self.device)

    def _init_model(self):
        return GraphConv_KGA0_atr2(dim=self.emb_size,
                                   n_hops=self.context_hops,
                                   n_users=self.n_users,
                                   n_items=self.n_items,
                                   n_entities=self.n_entities,
                                   n_relations=self.n_relations,
                                   ratedrop=self.mess_dropout_rate,
                                   dim_flag=self.dim_flag,
                                   r_num0=self.r_num0,
                                   k_att=self.k_att)

    def forward(self, batch, graph, graph_UIS):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS, self.all_embed, self.all_embed_cf, True)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_UIS):
        return self.gcn(graph, graph_UIS, self.all_embed, self.all_embed_cf, False)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2

        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss


class GraphConv_KGA0_atr2(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, dim_flag, r_num0, k_att, ratedrop):
        super(GraphConv_KGA0_atr2, self).__init__()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.ratedrop = ratedrop
        self.dim_flag = dim_flag
        self.r_num0 = r_num0
        self.k_att = k_att

        self.dim_flag1 = [0]
        for i in range(self.n_relations):
            self.dim_flag1.append(self.dim_flag1[-1] + self.dim_flag[i])

        self.convs_ui0 = nn.ModuleList()

        self.tri_embed_u = nn.ModuleList()

        self.tri_embed = KGA00(n_users=n_users, n_entities=n_entities, n_items=n_items,
                               n_relations=n_relations, dim=dim, dim_flag1=self.dim_flag1)

        for j in range(n_hops):
            self.convs_ui0.append(
                KGA1(n_users=n_users, n_entities=n_entities, n_items=n_items,
                     n_relations=n_relations,
                     dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

        self.item2user = KGA2_atr2(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                   n_relations=n_relations,
                                   dim=dim, dim_flag=dim_flag, dim_flag1=self.dim_flag1, k_att=self.k_att)

    def forward(self, graph, graph_UIS, all_embed, all_embed_cf, dropout):
        entity_embed = all_embed
        user_embed_cf = all_embed_cf[:self.n_users, :]
        entity_embed_cf = all_embed_cf[self.n_users:self.n_users + self.n_items, :]

        if dropout:

            random_indices = np.random.choice(graph_UIS.edges()[0].shape[0],
                                              size=int(graph_UIS.edges()[0].shape[0] * self.ratedrop),
                                              replace=False)
            graph_UIS = dgl.edge_subgraph(graph_UIS, random_indices, preserve_nodes=True)

        """cal edge embedding"""
        entity_embed0 = self.tri_embed(graph, entity_embed)

        if dropout:
            entity_embed0 = self.dropout(entity_embed0)

        entity_res_emb = entity_embed0

        for j in range(self.n_hops):
            entity_embed0 = self.convs_ui0[j](graph, entity_embed0)
            if dropout:
                entity_embed0 = self.dropout(entity_embed0)
            entity_res_emb = torch.add(entity_res_emb, entity_embed0)


        """agg user from diffirent atr"""
        user_embed_agg = self.item2user(graph_UIS, entity_embed_cf, entity_res_emb, user_embed_cf)
        if dropout:
            user_embed_agg = self.dropout(user_embed_agg)

        user_emb_f = torch.add(user_embed_agg, user_embed_cf)
        entity_res_emb = torch.add(entity_res_emb[:self.n_items, :], entity_embed_cf)

        return entity_res_emb, user_emb_f

    def cal_deg(self, edges):
        atr = torch.exp(torch.sum(torch.mul(self.W_R[edges.data['type']], self.atr_att), dim=1))
        return {'atr': atr}


class KGA00(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dim_flag1):
        super(KGA00, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dim_flag1 = dim_flag1

    def cal_attribute1(self, edges):
        edge_emb = edges.src['node'] * self.W_r
        return {'emb': edge_emb}

    def forward(self, graph, entity_emb):
        graph = graph.local_var()

        for i in range(self.n_relations):
            graph.nodes['item'].data['node{}'.format(i)] = entity_emb[:, self.dim_flag1[i]:self.dim_flag1[i + 1]]

        entity_emb_f = []
        for i in range(self.n_relations):
            self.r = i
            """mean_edge"""
            graph.update_all(
                dgl.function.copy_u('node{}'.format(i), 'send'),
                dgl.function.mean('send', 'node0{}'.format(i)), etype=i)

            entity_emb_f.append(graph.nodes['item'].data['node0{}'.format(i)])

        entity_emb_f = torch.cat(entity_emb_f, dim=1)

        return entity_emb_f


class KGA1(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(KGA1, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute1(self, edges):

        edge_emb = edges.src['node'] * self.W_r

        return {'emb': edge_emb}

    def cal_attribute2(self, edges):
        att = edges.data['att'] / edges.dst['nodeatt']
        return {'att1': att}

    def e_mul_e(self, edges):
        att = edges.data['att1'].unsqueeze(1) * edges.data['emb']
        return {'nei': att}

    def forward(self, graph, entity_emb):
        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb

        entity_emb_f = 0
        for i in range(self.n_relations):
            graph.update_all(
                dgl.function.copy_u('node', 'send'),
                dgl.function.mean('send', 'node{}'.format(i)), etype=i)

            entity_emb_f = torch.add(entity_emb_f, graph.nodes['item'].data['node{}'.format(i)])

        return entity_emb_f


class KGA2_atr2(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dim_flag, dim_flag1, k_att):
        super(KGA2_atr2, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dim_flag = dim_flag
        self.dim_flag1 = dim_flag1
        self.k_att = k_att

        self.dk = math.pow(dim_flag1[-1], 0.5)

    def cal_attribute_UI0(self, edges):
        atr1 = torch.exp(torch.cosine_similarity(edges.src['node'], edges.dst['node'].unsqueeze(1), dim=2))
        atr1 = nn.Softmax(dim=-1)(atr1).unsqueeze(1)
        atr1 = torch.matmul(atr1, edges.src['node']).squeeze(1)
        return {'atr': atr1}

    def cal_attribute_UI1(self, edges):
        atr1 = edges.data['att'] / edges.dst['sumatt']
        return {'att1': atr1}

    def u_mul_e(self, edges):
        atr1 = edges.src['node'] * edges.data['att1'].unsqueeze(2)
        return {'nei': atr1}

    def cal_cluster(self, edges):
        att = []
        for i in range(self.n_relations):
            att.append(torch.cosine_similarity(edges.dst['node0'][:, self.dim_flag1[i]:self.dim_flag1[i + 1]],
                                               edges.src['node'][:, self.dim_flag1[i]:self.dim_flag1[i + 1]],
                                               dim=1).unsqueeze(1))
        att = torch.cat(att, dim=1)
        return {'att_cluster': att}

    def forward(self, graph_UIS, entity_embed0, entity_embed, user_embed):
        """cal user"""
        graph_UIS = graph_UIS.local_var()

        graph_UIS.nodes['item'].data['node0'] = entity_embed0[:self.n_items]
        graph_UIS.nodes['item'].data['node'] = entity_embed[:self.n_items]

        graph_UIS.update_all(
            dgl.function.copy_u('node', 'temp4'),
            dgl.function.mean('temp4', 'node0'), etype=1)

        user_emb_f = graph_UIS.nodes['user'].data['node0']


        graph_UIS.update_all(
            dgl.function.copy_u('node0', 'temp5'),
            dgl.function.mean('temp5', 'node00'), etype=1)

        user_emb_f0 = graph_UIS.nodes['user'].data['node00']

        att = []
        for i in range(self.n_relations):
            user_att = torch.sum(user_emb_f[:, self.dim_flag1[i]:self.dim_flag1[i + 1]] *
                                 user_embed[:, self.dim_flag1[i]:self.dim_flag1[i + 1]], dim=1).unsqueeze(1)
            user_att = nn.ReLU()(user_att) + 1e-10

            mean_att = torch.sum(
                torch.mean(entity_embed[:self.n_items][:, self.dim_flag1[i]:self.dim_flag1[i + 1]], dim=0) *
                user_embed[:, self.dim_flag1[i]:self.dim_flag1[i + 1]], dim=1).unsqueeze(1)
            mean_att = nn.ReLU()(mean_att) + 1e-8
            att.append(self.k_att * nn.ReLU()(user_att / mean_att - 1) + 0.01)

        att = torch.cat(att, dim=1)
        score = nn.Tanh()(att)
        """------------user interest attention----------------"""
        self.out = score
        a = []
        for i in range(self.n_relations):
            a.append(score[:, i].unsqueeze(1).expand(-1, self.dim_flag[i]))
        score = torch.cat(a, dim=1)
        user_emb_f = score * user_emb_f

        user_emb_f = user_emb_f + user_emb_f0

        return user_emb_f
