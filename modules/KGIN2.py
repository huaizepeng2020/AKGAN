'''
PyTorch Implementation of AKGAN
@author: Zepeng Huai
'''
__author__ = "Zepeng Huai"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter_mean
import dgl
import math
from sklearn.cluster import KMeans, MiniBatchKMeans

kkk = 1


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_factors):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors

    def forward(self, entity_emb, user_emb, latent_emb,
                edge_index, edge_type, interact_mat,
                weight, disen_weight_att):
        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        n_factors = self.n_factors

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """cul user->latent factor attention"""
        score_ = torch.mm(user_emb, latent_emb.t())
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
                                weight).expand(n_users, n_factors, channel)
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

        return entity_agg, user_agg


class Aggregator_KGUI(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_factors):
        super(Aggregator_KGUI, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors

    def forward(self, entity_emb,
                edge_index, edge_type, edge_index_user, edge_type_user,
                weight):
        n_entities = entity_emb.shape[0]
        dim = entity_emb.shape[1]
        n_users = self.n_users

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        head_user, tail_user = edge_index_user
        edge_relation_emb_user = weight[
            edge_type_user - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb_user = entity_emb[tail_user] * edge_relation_emb_user  # [-1, channel]
        user_agg = scatter_mean(src=neigh_relation_emb_user, index=head_user, dim_size=n_users, dim=0)

        return entity_agg, user_agg


class Aggregator_KGUI_linear(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_items, n_relations, n_new_node, dropuot):
        super(Aggregator_KGUI_linear, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_relations = n_relations
        self.n_new_node = n_new_node
        self.dropout_rate = dropuot

    def cal_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        # KGIN
        # att_tail = self.entity_embed(edges.dst['id']) * self.W_r
        return {'atr': att_tail}

    def cal_user_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        return {'atr': att_tail}

    def cal_att_user_attribute(self, edges):
        atr_tail = self.all_embed[edges.src['id']] * self.W_r.unsqueeze(0)
        center_embed = self.all_embed[edges.dst['id']]
        att_atr = torch.matmul(center_embed.unsqueeze(1), atr_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_atr}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att_atr'] * edges.data['atr']}

    def udf_u_mul_u(self, nodes):
        return {'node1': torch.matmul(self.user_embed[nodes.data['user']], self.W_atr2pre)}

    def cal_att2pre_attention(self, edges):
        att = torch.matmul(edges.src['node'].unsqueeze(1), edges.dst['node1'].unsqueeze(2)).squeeze(1)
        return {'att_atr': att}

    def cal_pre2user1(self, edges):
        # att_tail = torch.matmul(edges.dst['node'], self.W_pre2user)
        att_tail = edges.dst['node']
        att_tail = torch.matmul(edges.data['atr'].unsqueeze(1), att_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_tail}

    def edge_softmax_fix(self, graph, score):
        graph = graph.local_var()
        graph.edata['out'] = score
        graph.edata['out'] = torch.exp(graph.edata['out'])
        graph.update_all(dgl.function.copy_e('out', 'temp'), self.reduce_sum)
        graph.apply_edges(dgl.function.e_div_v('out', 'out_sum', 'out'))
        out = graph.edata['out']
        return out

    def reduce_sum(self, nodes):
        accum = torch.sum(nodes.mailbox['temp'], 1)
        return {'out_sum': accum}

    def forward11(self, graph, graph_atr2pre, graph_pre2user, g1, g2, user_emb, entity_emb,
                  edge_index, edge_type, edge_index_user, edge_type_user,
                  interact_mat,
                  weight, bias, dropout):
        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users

        if dropout:
            """KG aggregate"""
            graph = graph.local_var()

            a = graph.edges()
            a1 = torch.randint(0, a[0].shape[0], (1, int(a[0].shape[0] * self.dropout_rate))).squeeze(0).to(
                graph.device)
            a2 = graph.edata['type']

            # g1 = dgl.DGLGraph()
            # g1 = g1.to(graph.device)
            g1 = g1.local_var()
            g1.add_nodes(n_entities)
            g1.ndata['id'] = torch.arange(n_entities, dtype=torch.long, device=graph.device)
            g1.ndata['node'] = entity_emb
            g1.add_edges(a[0][a1], a[1][a1])
            g1.edata['type'] = a2[a1]

            for i in range(self.n_relations):
                edge_idxs = g1.filter_edges(lambda edge: edge.data['type'] == i)
                self.W_r = weight[i]
                self.b_r = bias[i].unsqueeze(0)
                self.entity_embed = entity_emb
                g1.apply_edges(self.cal_attribute, edge_idxs)
            g1.update_all(dgl.function.copy_edge('atr', 'neib'), dgl.function.mean('neib', 'node'))
            entity_emb_f = g1.ndata['node']

            user_emb = torch.sparse.mm(interact_mat, entity_emb_f)[:self.n_items]

            """user aggregate"""
            graph_user = graph_user.local_var()

            a = graph_user.edges()
            a1 = torch.randint(0, a[0].shape[0], (1, int(a[0].shape[0] * self.dropout_rate))).squeeze(0).to(
                graph.device)
            a2 = graph_user.edata['type']

            # g2 = dgl.DGLGraph()
            # g2 = g2.to(graph.device)
            g2 = g2.local_var()
            g2.add_nodes(n_entities + n_users)
            g2.ndata['id'] = torch.arange(n_entities + n_users, dtype=torch.long, device=graph.device)

            g2.ndata['node'] = torch.cat([user_emb, entity_emb], dim=0)
            g2.add_edges(a[0][a1], a[1][a1])
            g2.edata['type'] = a2[a1]

            for i in range(self.n_relations):
                edge_idxs = g2.filter_edges(lambda edge: edge.data['type'] == i)
                self.W_r = weight[i]
                self.b_r = bias[i].unsqueeze(0)
                self.entity_embed = entity_emb
                self.all_embed = torch.cat([user_emb, entity_emb], dim=0)
                g2.apply_edges(self.cal_user_attribute, edge_idxs)
                g2.apply_edges(self.cal_att_user_attribute, edge_idxs)
            g2.edata['att_atr'] = self.edge_softmax_fix(g2, g2.edata.pop('att_atr'))
            # g2.update_all(dgl.function.copy_edge('atr', 'neib'), dgl.function.mean('neib', 'node'))
            # g2.update_all(dgl.function.u_mul_e('atr', 'att_atr', 'neib'), dgl.function.sum('neib', 'node'))
            g2.update_all(self.udf_e_mul_e, dgl.function.sum('neib', 'node'))
            user_emb_f = g2.ndata['node'][:self.n_users]
        else:
            """KG aggregate"""
            graph = graph.local_var()
            graph.ndata['node'] = entity_emb
            for i in range(self.n_relations):
                edge_idxs = graph.filter_edges(lambda edge: edge.data['type'] == i)
                self.W_r = weight[i]
                self.b_r = bias[i].unsqueeze(0)
                self.entity_embed = entity_emb
                graph.apply_edges(self.cal_attribute, edge_idxs)
            graph.update_all(dgl.function.copy_edge('atr', 'neib'), dgl.function.mean('neib', 'node'))
            entity_emb_f = graph.ndata['node']

            user_emb = torch.sparse.mm(interact_mat, entity_emb_f)[:self.n_items]

            """user aggregate"""
            graph_user = graph_user.local_var()
            graph_user.ndata['node'] = torch.cat([user_emb, entity_emb], dim=0)
            for i in range(self.n_relations):
                edge_idxs = graph_user.filter_edges(lambda edge: edge.data['type'] == i)
                self.W_r = weight[i]
                self.b_r = bias[i].unsqueeze(0)
                self.entity_embed = entity_emb
                self.all_embed = torch.cat([user_emb, entity_emb], dim=0)
                graph_user.apply_edges(self.cal_user_attribute, edge_idxs)
                graph_user.apply_edges(self.cal_att_user_attribute, edge_idxs)
            # graph_user.update_all(dgl.function.copy_edge('atr', 'neib'), dgl.function.mean('neib', 'node'))
            graph_user.update_all(self.udf_e_mul_e, dgl.function.sum('neib', 'node'))
            user_emb_f = graph_user.ndata['node'][:self.n_users]

        return entity_emb_f, user_emb_f

    def forward(self, graph, graph_atr2pre, graph_pre2user, user_emb, entity_emb,
                edge_index, edge_type, edge_index_user, edge_type_user,
                interact_mat,
                weight, bias,
                weight1, bias1,
                weight2, bias2,
                dropout):
        self.user_embed = user_emb
        n_entities = entity_emb.shape[0]
        dim = entity_emb.shape[1]

        """KG aggregate"""
        graph = graph.local_var()
        graph.ndata['node'] = entity_emb
        for i in range(self.n_relations):
            edge_idxs = graph.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = weight[i]
            self.b_r = bias[i].unsqueeze(0)
            self.entity_embed = entity_emb
            graph.apply_edges(self.cal_attribute, edge_idxs)
        graph.update_all(dgl.function.copy_edge('atr', 'neib'), dgl.function.mean('neib', 'node'))
        entity_emb_f = graph.ndata['node']

        """atr2pre aggregate"""
        # attention
        graph_atr2pre = graph_atr2pre.local_var()
        new_node_emb = torch.zeros(size=(self.n_new_node, dim), device=graph.device)
        graph_atr2pre.ndata['node'] = torch.cat([new_node_emb, entity_emb], dim=0)

        # for i in range(self.n_relations):
        #     node_idxs = graph_atr2pre.filter_nodes(lambda node: node.data['relation'] == i)
        #     self.W_atr2pre = weight1[i]
        #     self.b_atr2pre = bias1[i].unsqueeze(0)
        #     graph_atr2pre.apply_nodes(self.udf_u_mul_u, node_idxs)
        # graph_atr2pre.apply_edges(self.cal_att2pre_attention)
        # graph_atr2pre.edata['att_atr'] = self.edge_softmax_fix(graph_atr2pre, graph_atr2pre.edata.pop('att_atr'))
        # graph_atr2pre.update_all(dgl.function.u_mul_e('node', 'att_atr', 'neib'), dgl.function.sum('neib', 'node'))
        # mean pool
        graph_atr2pre.update_all(dgl.function.copy_u('node', 'neib'), dgl.function.mean('neib', 'node'))
        user_pre_embed = graph_atr2pre.ndata['node'][:self.n_new_node]

        """pre2user aggregate"""
        graph_pre2user = graph_pre2user.local_var()
        graph_pre2user.ndata['node'] = torch.cat([user_emb, user_pre_embed], dim=0)
        for i in range(self.n_relations):
            edge_idxs = graph_pre2user.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = weight[i]
            self.b_r = bias[i].unsqueeze(0)
            self.user_pre_embed = torch.cat([user_emb, user_pre_embed], dim=0)
            graph_pre2user.apply_edges(self.cal_user_attribute, edge_idxs)

        for i in range(self.n_relations):
            edge_idxs = graph_pre2user.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_pre2user = weight2[i]
            self.b_pre2user = bias2[i].unsqueeze(0)
            graph_pre2user.apply_edges(self.cal_pre2user1, edge_idxs)

        graph_pre2user.edata['att_atr'] = self.edge_softmax_fix(graph_pre2user, graph_pre2user.edata.pop('att_atr'))
        graph_pre2user.update_all(self.udf_e_mul_e, dgl.function.sum('neib', 'node'))
        user_emb_f = graph_pre2user.ndata['node'][:self.n_users]

        return entity_emb_f, user_emb_f


class Aggregator_KGUI1(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_items, n_relations, dropuot):
        super(Aggregator_KGUI1, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_relations = n_relations
        self.dropout_rate = dropuot

    def cal_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        # KGIN
        # att_tail = self.entity_embed(edges.dst['id']) * self.W_r
        return {'atr': att_tail}

    def cal_user_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        att_tail = torch.matmul(self.weight_user_pre_c, att_tail.unsqueeze(1))
        return {'atr': att_tail}

    def cal_att_user_attribute(self, edges):
        atr_tail = self.all_embed[edges.src['id']] * self.W_r.unsqueeze(0)
        center_embed = self.all_embed[edges.dst['id']]
        att_atr = torch.matmul(center_embed.unsqueeze(1), atr_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_atr}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att_atr'] * edges.data['atr']}

    def udf_u_mul_u(self, nodes):
        return {'node1': torch.matmul(self.user_embed[nodes.data['user']], self.W_atr2pre)}

    def cal_att2pre_attention(self, edges):
        att = torch.matmul(edges.src['node'].unsqueeze(1), edges.dst['node1'].unsqueeze(2)).squeeze(1)
        return {'att_atr': att}

    def cal_pre2user1(self, edges):
        # att_tail = torch.matmul(edges.dst['node'], self.W_pre2user)
        att_tail = edges.dst['node']
        att_tail = torch.matmul(edges.data['atr'].unsqueeze(1), att_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_tail}

    def edge_softmax_fix(self, graph, score):
        graph = graph.local_var()
        graph.edata['out'] = score
        graph.edata['out'] = torch.exp(graph.edata['out'])
        graph.update_all(dgl.function.copy_e('out', 'temp'), self.reduce_sum)
        graph.apply_edges(dgl.function.e_div_v('out', 'out_sum', 'out'))
        out = graph.edata['out']
        return out

    def reduce_sum(self, nodes):
        accum = torch.sum(nodes.mailbox['temp'], 1)
        return {'out_sum': accum}

    def forward(self, graph, graph_user, user_emb, entity_emb,
                edge_index, edge_type, edge_index_user, edge_type_user,
                interact_mat,
                weight, bias,
                weight1, bias1,
                weight2, bias2,
                weight_user_pre,
                dropout):
        self.user_embed = user_emb
        n_entities = entity_emb.shape[0]
        dim = entity_emb.shape[1]
        # self.weight_user_pre = weight_user_pre

        """KG aggregate"""
        graph = graph.local_var()
        graph.ndata['node'] = entity_emb
        for i in range(self.n_relations):
            edge_idxs = graph.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = weight[i]
            self.b_r = bias[i].unsqueeze(0)
            self.entity_embed = entity_emb
            graph.apply_edges(self.cal_attribute, edge_idxs)
        graph.update_all(dgl.function.copy_edge('atr', 'neib'), dgl.function.mean('neib', 'node'))
        entity_emb_f = graph.ndata['node']

        """user aggregate"""
        # attention
        graph_user = graph_user.local_var()
        graph_user.ndata['node'] = torch.cat([user_emb, entity_emb], dim=0)
        for i in range(self.n_relations):
            edge_idxs = graph_user.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = weight[i]
            self.b_r = bias[i].unsqueeze(0)
            self.weight_user_pre_c = weight_user_pre[:, i].unsqueeze(1)
            graph_user.apply_edges(self.cal_user_attribute, edge_idxs)

        graph_user.update_all(dgl.function.copy_edge('atr', 'neib'), dgl.function.mean('neib', 'node'))
        user_emb_f1 = graph_user.ndata['node'][:self.n_users]
        user_emb_f = []
        for i in range(self.n_users):
            user_emb_f.append(user_emb_f1[i, i, :].unsqueeze(0))
        user_emb_f = torch.cat(user_emb_f, dim=0)

        return entity_emb_f, user_emb_f


class Aggregator_KGUI2(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_items, n_relations, dropuot):
        super(Aggregator_KGUI2, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_relations = n_relations
        self.dropout_rate = dropuot

    def cal_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        return {'atr': att_tail}

    def cal_user_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        att_tail = torch.matmul(self.weight_user_pre_c, att_tail.unsqueeze(1))
        return {'atr': att_tail}

    def cal_att_user_attribute(self, edges):
        atr_tail = self.all_embed[edges.src['id']] * self.W_r.unsqueeze(0)
        center_embed = self.all_embed[edges.dst['id']]
        att_atr = torch.matmul(center_embed.unsqueeze(1), atr_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_atr}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att_atr'] * edges.data['atr']}

    def udf_u_mul_u(self, nodes):
        return {'node1': torch.matmul(self.user_embed[nodes.data['user']], self.W_atr2pre)}

    def cal_att2pre_attention(self, edges):
        att = torch.matmul(edges.src['node'].unsqueeze(1), edges.dst['node1'].unsqueeze(2)).squeeze(1)
        return {'att_atr': att}

    def cal_pre2user1(self, edges):
        # att_tail = torch.matmul(edges.dst['node'], self.W_pre2user)
        att_tail = edges.dst['node']
        att_tail = torch.matmul(edges.data['atr'].unsqueeze(1), att_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_tail}

    def edge_softmax_fix(self, graph, score):
        graph = graph.local_var()
        graph.edata['out'] = score
        graph.edata['out'] = torch.exp(graph.edata['out'])
        graph.update_all(dgl.function.copy_e('out', 'temp'), self.reduce_sum)
        graph.apply_edges(dgl.function.e_div_v('out', 'out_sum', 'out'))
        out = graph.edata['out']
        return out

    def reduce_sum(self, nodes):
        accum = torch.sum(nodes.mailbox['temp'], 1)
        return {'out_sum': accum}

    def forward(self, graph, entity_emb,
                edge_index, edge_type, edge_index_user, edge_type_user,
                interact_mat,
                weight, bias,
                weight1, bias1,
                weight2, bias2,
                weight_user_pre,
                dropout):
        """KG aggregate"""
        graph = graph.local_var()
        graph.ndata['node'] = entity_emb
        for i in range(self.n_relations):
            edge_idxs = graph.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = weight[i]
            self.b_r = bias[i].unsqueeze(0)
            self.entity_embed = entity_emb
            graph.apply_edges(self.cal_attribute, edge_idxs)
        graph.update_all(dgl.function.copy_edge('atr', 'neib'), dgl.function.mean('neib', 'node'))
        entity_emb_f = graph.ndata['node']

        return entity_emb_f


class Aggregator_KGUI3(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dropuot):
        super(Aggregator_KGUI3, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot

        initializer = nn.init.xavier_uniform_
        user_pre = initializer(torch.empty(n_users, n_relations, dim))  # not include interact
        self.user_pre = nn.Parameter(user_pre)  # [n_relations - 1, dim,dim]

        weight = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, dim,dim]

    def cal_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        return {'atr': att_tail}

    def cal_att_i2u(self, edges):
        att_tail = torch.sum(torch.mul(edges.src['node'], edges.dst['node']), axis=1)
        att_tail = torch.exp(att_tail)
        return {'att': att_tail}

    def cal_att_i2u_1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']
        return {'att1': att_tail}

    def cal_hete_attribute(self, edges):
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        return {'atr': att_tail}

    def cal_user_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        att_tail = torch.matmul(self.weight_user_pre_c, att_tail.unsqueeze(1))
        return {'atr': att_tail}

    def cal_att_user_attribute(self, edges):
        atr_tail = self.all_embed[edges.src['id']] * self.W_r.unsqueeze(0)
        center_embed = self.all_embed[edges.dst['id']]
        att_atr = torch.matmul(center_embed.unsqueeze(1), atr_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_atr}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att_atr'] * edges.data['atr']}

    def udf_u_mul_u(self, nodes):
        return {'node1': torch.matmul(self.user_embed[nodes.data['user']], self.W_atr2pre)}

    def cal_att2pre_attention(self, edges):
        att = torch.matmul(edges.src['node'].unsqueeze(1), edges.dst['node1'].unsqueeze(2)).squeeze(1)
        return {'att_atr': att}

    def cal_pre2user1(self, edges):
        # att_tail = torch.matmul(edges.dst['node'], self.W_pre2user)
        att_tail = edges.dst['node']
        att_tail = torch.matmul(edges.data['atr'].unsqueeze(1), att_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_tail}

    def edge_softmax_fix(self, graph, score):
        graph = graph.local_var()
        graph.edata['out'] = score
        graph.edata['out'] = torch.exp(graph.edata['out'])
        graph.update_all(dgl.function.copy_e('out', 'temp'), self.reduce_sum)
        graph.apply_edges(dgl.function.e_div_v('out', 'out_sum', 'out'))
        out = graph.edata['out']
        return out

    def reduce_sum(self, nodes):
        accum = torch.sum(nodes.mailbox['temp'], 1)
        return {'out_sum': accum}

    def forward(self, graph, graph_i2u, graph_u2u, entity_emb,
                edge_index, edge_type, edge_index_user, edge_type_user,
                interact_mat,
                weight,
                dropout):
        """KG aggregate"""
        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb
        for i in range(self.n_relations):
            self.W_r = self.weight[i]
            graph.apply_edges(self.cal_attribute, etype=i)
        update_dict = {}
        for graph_type in graph.canonical_etypes:
            srctype, etype, dsttype = graph_type
            if etype < 100:
                update_dict[etype] = (dgl.function.copy_edge('atr', 'm'), dgl.function.mean('m', 'node'))
        graph.multi_update_all(update_dict, 'stack')
        entity_emb_f = graph.nodes['item'].data['node']
        item_emb_f = entity_emb_f[:self.n_items]

        # graph_i2u = graph_i2u.local_var()
        # for i in range(self.n_relations):
        #     graph_i2u.nodes['item{}'.format(i)].data['node'] = item_emb_f[:, i, :]
        #     graph_i2u.nodes['user{}'.format(i)].data['node'] = self.user_pre[:, i, :]
        # for i in range(self.n_relations):
        #     graph_i2u.apply_edges(self.cal_att_i2u, etype=i)
        #     graph_i2u.update_all(dgl.function.copy_e('att', 'temp'), dgl.function.sum('temp', 'sum_att'), etype=i)
        #     graph_i2u.apply_edges(self.cal_att_i2u_1, etype=i)
        #     graph_i2u.update_all(dgl.function.u_mul_e('node', 'att1', 'send'), dgl.function.sum('send', 'node'),
        #                          etype=i)

        graph_i2u = graph_i2u.local_var()
        for i in range(self.n_relations):
            graph_i2u.nodes['item{}'.format(i)].data['node'] = item_emb_f[:, i, :]

        for i in range(self.n_relations):
            graph_i2u.update_all(dgl.function.copy_u('node', 'temp'), dgl.function.sum('temp', 'node'), etype=i)

        user = []
        for i in range(self.n_relations):
            user.append(graph_i2u.nodes['user{}'.format(i)].data['node'])
        user = torch.cat(user, dim=1).reshape(self.n_users, self.n_relations, self.dim)

        # graph_u2u = graph_u2u.local_var()
        # update_dict = {}
        # for graph_type in graph_u2u.canonical_etypes:
        #     srctype, etype, dsttype = graph_type
        #     if etype < 100:
        #         update_dict[etype] = (dgl.function.copy_u('node', 'm'), dgl.function.mean('m', 'node'))
        entity_next = torch.sum(entity_emb_f, dim=1)
        entity = F.normalize(entity_emb_f, dim=2).reshape(self.n_entities, -1)
        user = F.normalize(user, dim=2).reshape(self.n_users, -1)

        return entity, entity_next, user


class Aggregator_KGUI3_1(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dropuot):
        super(Aggregator_KGUI3_1, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot

        initializer = nn.init.xavier_uniform_
        user_pre = initializer(torch.empty(n_users, dim))  # not include interact
        self.user_pre = nn.Parameter(user_pre)  # [n_relations - 1, dim,dim]

        weight = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, dim,dim]

        # self.entity_linear = nn.Linear(int(n_relations * dim), dim, bias=True)  # [n_relations - 1, dim,dim]
        # initializer(self.entity_linear.weight)

    def cal_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        return {'atr': att_tail}

    def cal_att_i2u(self, edges):
        att_tail = torch.sum(torch.mul(edges.src['node'], edges.dst['node']), axis=1)
        att_tail = torch.exp(att_tail)
        return {'att': att_tail}

    def cal_att_i2u_1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']
        return {'att1': att_tail}

    def cal_hete_attribute(self, edges):
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        return {'atr': att_tail}

    def cal_user_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        att_tail = torch.matmul(self.weight_user_pre_c, att_tail.unsqueeze(1))
        return {'atr': att_tail}

    def cal_att_user_attribute(self, edges):
        atr_tail = self.all_embed[edges.src['id']] * self.W_r.unsqueeze(0)
        center_embed = self.all_embed[edges.dst['id']]
        att_atr = torch.matmul(center_embed.unsqueeze(1), atr_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_atr}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att_atr'] * edges.data['atr']}

    def udf_u_mul_u(self, nodes):
        return {'node1': torch.matmul(self.user_embed[nodes.data['user']], self.W_atr2pre)}

    def cal_att2pre_attention(self, edges):
        att = torch.matmul(edges.src['node'].unsqueeze(1), edges.dst['node1'].unsqueeze(2)).squeeze(1)
        return {'att_atr': att}

    def cal_pre2user1(self, edges):
        # att_tail = torch.matmul(edges.dst['node'], self.W_pre2user)
        att_tail = edges.dst['node']
        att_tail = torch.matmul(edges.data['atr'].unsqueeze(1), att_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_tail}

    def edge_softmax_fix(self, graph, score):
        graph = graph.local_var()
        graph.edata['out'] = score
        graph.edata['out'] = torch.exp(graph.edata['out'])
        graph.update_all(dgl.function.copy_e('out', 'temp'), self.reduce_sum)
        graph.apply_edges(dgl.function.e_div_v('out', 'out_sum', 'out'))
        out = graph.edata['out']
        return out

    def reduce_sum(self, nodes):
        accum = torch.sum(nodes.mailbox['temp'], 1)
        return {'out_sum': accum}

    def forward(self, sub_idx1, graph, graph_i2u, graph_u2u, graph_i2u_0, entity_emb,
                interact_mat,
                dropout):
        """KG aggregate"""
        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb
        for i in range(self.n_relations):
            self.W_r = self.weight[i]
            graph.apply_edges(self.cal_attribute, etype=i)
        update_dict = {}
        for graph_type in graph.canonical_etypes:
            srctype, etype, dsttype = graph_type
            if etype < 100:
                update_dict[etype] = (dgl.function.copy_edge('atr', 'm'), dgl.function.mean('m', 'node'))
        graph.multi_update_all(update_dict, 'stack')
        entity_emb_f = graph.nodes['item'].data['node']
        # entity_emb_f = self.entity_linear(entity_emb_f.reshape(self.n_entities, -1))
        item_emb_f = entity_emb_f[:self.n_items]

        graph_i2u_0 = graph_i2u_0.local_var()
        graph_i2u_0.nodes['item'].data['node'] = item_emb_f.reshape(self.n_items, -1)
        graph_i2u_0.update_all(dgl.function.copy_u('node', 'temp'), dgl.function.mean('temp', 'node'), etype=0)
        user = graph_i2u_0.nodes['user'].data['node']

        entity_next = torch.sum(entity_emb_f, dim=1)
        entity = F.normalize(entity_emb_f, dim=2).reshape(self.n_entities, -1)
        # user = F.normalize(user.reshape(self.n_users, self.n_relations, self.dim), dim=2).reshape(self.n_users, -1)
        user = F.normalize(user.reshape(len(sub_idx1[0]), self.n_relations, self.dim), dim=2).reshape(len(sub_idx1[0]),
                                                                                                      -1)
        user = user[sub_idx1[1]]

        return entity, entity_next, user


class Aggregator_upat_atr(nn.Module):
    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dropuot):
        super(Aggregator_upat_atr, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot

        self.IN = nn.InstanceNorm1d(self.n_relations)

    def cal_attribute(self, edges):
        atr = edges.src['node'] * edges.dst['node']
        return {'atr': atr}

    def forward(self, graph, entity_emb, W_R):
        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb
        for i in range(self.n_relations):
            self.W_r = W_R[i]
            self.r = i
            "att"
            "mean1"
            # graph.apply_edges(self.cal_attribute_noatt, etype=i)
            # graph.update_all(
            #     dgl.function.copy_e('atr', 'temp'),
            #     dgl.function.mean('temp', 'node{}'.format(i)), etype=i)
            "mean2"
            graph.update_all(
                dgl.function.copy_u('node', 'temp'),
                dgl.function.sum('temp', 'node{}'.format(i)), etype=i)

            # graph.apply_edges(self.cal_attribute, etype=i)
            # graph.update_all(
            #     dgl.function.copy_e('atr', 'temp'),
            #     dgl.function.sum('temp', 'node{}'.format(i)), etype=i)

        entity_emb_f = []
        for i in range(self.n_relations):
            entity_emb_f.append(graph.nodes['item'].data['node{}'.format(i)])
        entity_emb_f = torch.cat(entity_emb_f, dim=1) \
            .reshape(self.n_entities, self.n_relations, self.dim)

        entity_emb_f1 = self.IN(entity_emb_f)

        return entity_emb_f


class Aggregator_upat_mulatr(nn.Module):
    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dropuot):
        super(Aggregator_upat_mulatr, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot

        self.IN = nn.InstanceNorm1d(self.n_relations)

    def cal_attribute(self, edges):

        # """useful1"""
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) \
        #            * torch.matmul(edges.dst['node'].unsqueeze(1), self.W_r).squeeze(1)
        # att_tail_1 = torch.matmul(edges.dst['node'].unsqueeze(1), self.W_r).squeeze(1)
        # att_tai2 = torch.exp(torch.cosine_similarity(att_tail, att_tail_1, dim=1))
        # return {'atr': att_tail, 'att': att_tai2}
        # """useful1"""

        """useful1"""
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        att_tail_1 = torch.matmul(edges.dst['node'].unsqueeze(1), self.W_r).squeeze(1)
        att_tai2 = torch.exp(torch.cosine_similarity(att_tail, att_tail_1, dim=1))
        return {'atr': att_tail, 'att': att_tai2}
        """useful1"""

        """2"""
        # att_tail = edges.src['node']
        # att_tail_1 = edges.dst['node']
        # # att_tai2 = torch.exp(torch.cosine_similarity(
        # #     torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1), att_tail_1, dim=1))
        # att_tai2 = torch.exp(torch.cosine_similarity(att_tail, att_tail_1, dim=1))
        # return {'atr': att_tail, 'att': att_tai2}
        """2"""

    def cal_attribute_noatt(self, edges):
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        return {'atr': att_tail}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def forward(self, graph, entity_emb, W_R):

        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb
        for i in range(self.n_relations):
            self.W_r = W_R[i]
            self.r = i
            "att"
            # graph.apply_edges(self.cal_attribute, etype=i)
            # graph.update_all(
            #     dgl.function.copy_e('att', 'temp'),
            #     dgl.function.sum('temp', 'sum_att{}'.format(i)), etype=i)
            # graph.apply_edges(self.cal_attribute1, etype=i)
            # graph.update_all(
            #     self.udf_e_mul_e,
            #     dgl.function.sum('neib', 'node{}'.format(i)), etype=i)
            "mean1"
            # graph.apply_edges(self.cal_attribute_noatt, etype=i)
            # graph.update_all(
            #     dgl.function.copy_e('atr', 'temp'),
            #     dgl.function.mean('temp', 'node{}'.format(i)), etype=i)
            "mean2"
            graph.update_all(
                dgl.function.copy_u('node', 'temp'),
                dgl.function.sum('temp', 'node{}'.format(i)), etype=i)

        entity_emb_f = graph.nodes['item'].data['node0']
        for i in range(1, self.n_relations):
            entity_emb_f = entity_emb_f + graph.nodes['item'].data['node{}'.format(i)]

        entity_emb_f = self.IN(entity_emb_f)

        return entity_emb_f


class Aggregator_NFGNN_item(nn.Module):
    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dropuot):
        super(Aggregator_NFGNN_item, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot

        self.IN = nn.InstanceNorm1d(self.n_relations)

    def cal_attribute_noatt(self, edges):
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r[edges.data['type']]).squeeze(1)
        att_tail = edges.src['node'] * self.W_r[edges.data['type']]
        att_tail = att_tail / \
                   edges.dst['r_deg'][list(range(len(edges))),
                                      edges.data['type'].cpu().numpy().tolist()].unsqueeze(1)
        return {'atr': att_tail}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def forward(self, graph, entity_emb, W_R):
        graph = graph.local_var()
        graph.ndata['node'] = entity_emb

        self.W_r = W_R
        graph.apply_edges(self.cal_attribute_noatt)
        graph.update_all(
            dgl.function.copy_e('atr', 'temp'),
            dgl.function.sum('temp', 'node'))
        entity_emb_f = graph.ndata['node']

        # for i in range(self.n_relations):
        #     self.W_r = W_R[i]
        #     self.r = i
        #     "att"
        #     # graph.apply_edges(self.cal_attribute, etype=i)
        #     # graph.update_all(
        #     #     dgl.function.copy_e('att', 'temp'),
        #     #     dgl.function.sum('temp', 'sum_att{}'.format(i)), etype=i)
        #     # graph.apply_edges(self.cal_attribute1, etype=i)
        #     # graph.update_all(
        #     #     self.udf_e_mul_e,
        #     #     dgl.function.sum('neib', 'node{}'.format(i)), etype=i)
        #     "mean1"
        #     graph.apply_edges(self.cal_attribute_noatt, etype=i)
        #     graph.update_all(
        #         dgl.function.copy_e('atr', 'temp'),
        #         dgl.function.mean('temp', 'node{}'.format(i)), etype=i)
        #     "mean2"
        #     # graph.update_all(
        #     #     dgl.function.copy_u('node', 'temp'),
        #     #     dgl.function.sum('temp', 'node{}'.format(i)), etype=i)
        #
        # entity_emb_f = graph.nodes['item'].data['node0']
        # for i in range(1, self.n_relations):
        #     entity_emb_f = entity_emb_f + graph.nodes['item'].data['node{}'.format(i)]

        return entity_emb_f


class Aggregator_NFGNN_user(nn.Module):
    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dropuot):
        super(Aggregator_NFGNN_user, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot

        self.act = nn.Sigmoid()

    def cal_attribute_noatt(self, edges):
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r[edges.data['type']]).squeeze(1)
        att_tail = edges.src['node'] * self.W_r[edges.data['type']]

        # att = torch.exp(torch.sum(self.user_filter * att_tail, dim=1))
        att = torch.exp(torch.cosine_similarity(self.user_filter, att_tail, dim=1))
        # att = torch.exp(torch.sum(att_tail, dim=1))

        att_tail = att_tail * edges.dst['deg_all'].unsqueeze(1)
        att_tail = att_tail / \
                   edges.dst['r_deg'][list(range(len(edges))),
                                      edges.data['type'].cpu().numpy().tolist()].unsqueeze(1)
        # return {'atr': att_tail}
        return {'atr': att_tail, 'att': att}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['attsum']
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def forward(self, graph1, entity_emb, W_R, user_filter, user_inter, sub_edges):
        self.user_filter = user_filter.unsqueeze(0)

        graph1 = graph1.local_var()

        graph1.ndata['node'] = entity_emb

        self.W_r = W_R

        graph1.apply_edges(self.cal_attribute_noatt)
        graph1.update_all(
            dgl.function.copy_e('att', 'temp'),
            dgl.function.sum('temp', 'attsum'))
        graph1.apply_edges(self.cal_attribute1)
        graph1.update_all(
            dgl.function.u_mul_e('node', 'att1', 'temp1'),
            dgl.function.sum('temp1', 'node'))

        # graph1.update_all(
        #     dgl.function.copy_e('atr', 'temp'),
        #     dgl.function.sum('temp', 'node'))
        entity_emb_f = graph1.ndata['node']

        """attention"""
        atr = entity_emb_f[user_inter]
        # att = torch.exp(torch.sum(atr * user_filter.unsqueeze(0), dim=1))
        att = torch.exp(torch.cosine_similarity(atr, user_filter.unsqueeze(0)))
        att_sum = torch.sum(att)
        att = att / att_sum
        user_f = torch.matmul(att.unsqueeze(0), atr)
        """attention"""

        """mean"""
        # user_f = torch.mean(entity_emb_f[user_inter], dim=0).unsqueeze(0)
        """mean"""

        return user_f, entity_emb_f


class Aggregator_NFGNN_user0(nn.Module):
    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dropuot):
        super(Aggregator_NFGNN_user0, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot

        self.act = nn.Sigmoid()

    def cal_attribute_noatt(self, edges):
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r[edges.data['type']]).squeeze(1)
        att_tail = edges.src['node'] * self.W_r[edges.data['type']]
        att_tail = att_tail * self.user_filter
        att_tail = att_tail / \
                   edges.dst['r_deg'][list(range(len(edges))),
                                      edges.data['type'].cpu().numpy().tolist()].unsqueeze(1)
        return {'atr': att_tail}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def forward(self, graph1, entity_emb, W_R, user_filter, user_node, user_inter, sub_edges):
        self.user_node = user_node
        self.user_filter = user_filter

        graph1 = graph1.local_var()

        graph1.ndata['node'] = entity_emb

        self.W_r = W_R

        graph1.apply_edges(self.cal_attribute_noatt)
        graph1.update_all(
            dgl.function.copy_e('atr', 'temp'),
            dgl.function.sum('temp', 'node'))
        entity_emb_f = graph1.ndata['node']

        """attention"""
        atr = entity_emb_f[user_inter]
        # att=torch.exp(torch.mm(atr,user_node.unsqueeze(1)))
        # att = torch.exp(torch.cosine_similarity(atr, user_node.unsqueeze(0)))
        att = torch.exp(torch.cosine_similarity(atr, user_filter.unsqueeze(0)))
        att_sum = torch.sum(att)
        att = att / att_sum
        user_f = torch.matmul(att.unsqueeze(0), atr)
        """attention"""

        """mean"""
        # user_f = torch.mean(entity_emb_f[user_inter], dim=0).unsqueeze(0)
        """mean"""

        return user_f, entity_emb_f


class Aggregator_KGIT_mulatr0(nn.Module):
    def __init__(self, n_users, n_entities, n_items, n_relations, dim, new_r_all, dropuot):
        super(Aggregator_KGIT_mulatr0, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot
        self.new_r_all = new_r_all

        initializer = nn.init.xavier_uniform_
        user_pre = initializer(torch.empty(n_users, len(self.new_r_all), dim))  # not include interact
        self.user_pre = nn.Parameter(user_pre)  # [n_relations - 1, dim,dim]

    def cal_attribute1(self, edges):
        # att_tail = torch.matmul(
        #     edges.src['node'].unsqueeze(1) * edges.dst['node'].unsqueeze(1),
        #     self.W_r).squeeze(1)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) \
        #            * torch.matmul(edges.dst['node'].unsqueeze(1), self.W_r).squeeze(1)
        att_tail = edges.src['node']
        # att_tail = edges.src['node'] * self.w_r.unsqueeze(0)

        # att_tai2 = torch.exp(torch.cosine_similarity(att_tail, edges.dst['node'], dim=1))

        return {'atr': att_tail}
        # return {'atr': att_tail, 'att': att_tai2}

    def cal_attribute2(self, edges):
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)

        att_tai2 = torch.exp(torch.cosine_similarity(att_tail, edges.dst['node{}'.format(self.r)], dim=1))

        return {'atr': att_tail, 'att': att_tai2}

    def cal_iu_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def cal_att_i2u_het_diff(self, edges):
        att_tai1 = torch.cosine_similarity(
            edges.dst['node{}'.format(self.i)],
            edges.src['node{}'.format(self.i)], dim=1)
        # att_tai1 = torch.sum(
        #     torch.mul(
        #         edges.dst['node{}'.format(self.i)],
        #         edges.src['node{}'.format(self.i)]),
        #     dim=1)

        att_tail = torch.exp(att_tai1)
        return {'att{}'.format(self.i): att_tail}

    def cal_att_i2u_1_het_diff(self, edges):
        att_tail = edges.data['att{}'.format(self.i)] / edges.dst['sum_att{}'.format(self.i)]
        return {'att1{}'.format(self.i): att_tail}

    def node_sum(self, nodes):
        mean_atr = torch.mean(nodes.mailbox['send1'], dim=1)
        att_atr = torch.mean(torch.cosine_similarity(mean_atr.unsqueeze(1), nodes.mailbox['send1'], dim=2), dim=1)
        att_atr = (torch.tanh(att_atr) + 1) / 2
        return {'att_all{}'.format(self.i): att_atr}

    def forward(self, sub_idx1, graph, graph_i2u_0, entity_emb, W_R, w_R, b_R, new_r_all):
        # """KG aggregate"""
        entity_emb = F.normalize(entity_emb, dim=1)

        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb
        # for i, ii in enumerate(new_r_all):
        #     graph.nodes['user'].data['node{}'.format(ii)] = self.user_pre[sub_idx1[0], i, :]
        for i in range(self.n_relations):
            self.W_r = W_R[i]
            self.w_r = w_R[i]
            self.b_r = b_R[i]
            self.r = i
            graph.apply_edges(self.cal_attribute1, etype=i)
            """attention"""
            # graph.update_all(
            #     dgl.function.copy_e('att', 'temp'),
            #     dgl.function.sum('temp', 'sum_att{}'.format(i)), etype=i)
            # graph.apply_edges(self.cal_iu_attribute1, etype=i)
            # graph.update_all(
            #     self.udf_e_mul_e,
            #     dgl.function.sum('neib', 'node{}'.format(i)), etype=i)
            # """sum"""
            # graph.update_all(
            #     dgl.function.copy_u('node', 'send'),
            #     dgl.function.mean('send', 'node{}'.format(i)), etype=i)
            """mean_edge"""
            graph.update_all(
                dgl.function.copy_e('atr', 'send'),
                dgl.function.sum('send', 'node{}'.format(i)), etype=i)

            # """user"""
            # if i + self.n_relations in graph.etypes:
            #     graph.apply_edges(self.cal_attribute2, etype=i + self.n_relations)
            #     graph.update_all(
            #         dgl.function.copy_e('att', 'temp'),
            #         dgl.function.sum('temp', 'sum_att{}'.format(i)), etype=i + self.n_relations)
            #     graph.apply_edges(self.cal_iu_attribute1, etype=i + self.n_relations)
            #     graph.update_all(
            #         self.udf_e_mul_e,
            #         dgl.function.sum('neib', 'nodenew{}'.format(i)), etype=i + self.n_relations)
            #     graph.nodes['user'].data['nodenew{}'.format(i)] = \
            #         graph.nodes['user'].data['nodenew{}'.format(i)] + \
            #         graph.nodes['user'].data['node{}'.format(i)]

        entity_emb_f = []
        for i in range(self.n_relations):
            entity_emb_f.append(graph.nodes['item'].data['node{}'.format(i)])
        entity_emb_f = torch.cat(entity_emb_f, dim=1).reshape(self.n_entities, self.n_relations, self.dim)

        # entity_emb_f = F.normalize(entity_emb_f, dim=2)

        item_emb_f = entity_emb_f[:self.n_items]

        """user agg"""
        """user attention 2"""
        graph_i2u_0 = graph_i2u_0.local_var()
        for i, ii in enumerate(new_r_all):
            graph_i2u_0.nodes['item'].data['node{}'.format(ii)] = item_emb_f[:, ii, :]
            graph_i2u_0.nodes['user'].data['node{}'.format(ii)] = \
                self.user_pre[sub_idx1[0], i, :]
        for i in new_r_all:
            self.W_r = W_R[i]
            self.w_r = F.normalize(w_R[i], dim=0)
            self.i = i
            graph_i2u_0.apply_edges(self.cal_att_i2u_het_diff, etype=0)
            graph_i2u_0.update_all(dgl.function.copy_e('att{}'.format(self.i), 'temp'),
                                   dgl.function.sum('temp', 'sum_att{}'.format(self.i)), etype=0)
            graph_i2u_0.apply_edges(self.cal_att_i2u_1_het_diff, etype=0)
            graph_i2u_0.update_all(
                dgl.function.u_mul_e('node{}'.format(self.i), 'att1{}'.format(self.i), 'send'),
                dgl.function.sum('send', 'nodenew{}'.format(self.i)),
                etype=0)
            graph_i2u_0.nodes['user'].data['nodenew{}'.format(self.i)] = \
                graph_i2u_0.nodes['user'].data['nodenew{}'.format(self.i)] + \
                graph_i2u_0.nodes['user'].data['node{}'.format(self.i)]

            graph_i2u_0.update_all(
                dgl.function.copy_u('node{}'.format(self.i), 'send1'),
                self.node_sum, etype=0)

        user = []
        for i in new_r_all:
            user.append(graph_i2u_0.nodes['user'].data['nodenew{}'.format(i)])
        user = torch.cat(user, dim=1).reshape(len(sub_idx1[0]), -1, self.dim)

        user_att = []
        for i in new_r_all:
            user_att.append(graph_i2u_0.nodes['user'].data['att_all{}'.format(i)].unsqueeze(1))
        user_att = torch.cat(user_att, dim=1)

        """output"""
        entity = F.normalize(entity_emb_f[:, self.new_r_all, :], dim=2).reshape(self.n_entities, -1)

        user = F.normalize(user, dim=2)
        user = user * user_att.unsqueeze(2)
        user = user.reshape(len(sub_idx1[0]), -1)
        user = user[sub_idx1[1]]

        return entity, entity_emb_f, user


class Aggregator_KGIT(nn.Module):
    def __init__(self, n_users, n_entities, n_items, n_relations, dim, new_r_all, dropuot):
        super(Aggregator_KGIT, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot
        self.new_r_all = new_r_all

        initializer = nn.init.xavier_uniform_

        user_pre = initializer(torch.empty(n_users, len(self.new_r_all), dim))  # not include interact
        # user_pre = torch.ones(n_users, len(self.new_r_all), dim)  # not include interact
        self.user_pre = nn.Parameter(user_pre)  # [n_relations - 1, dim,dim]

        user_pre1 = initializer(torch.empty(n_users, dim))  # not include interact
        self.user_pre1 = nn.Parameter(user_pre1)  # [n_relations - 1, dim,dim]
        #
        # user_pre2 = initializer(torch.empty(n_users, int(n_relations * dim)))  # not include interact
        # self.user_pre2 = nn.Parameter(user_pre2)  # [n_relations - 1, dim,dim]
        #

        item_pre = initializer(torch.empty(n_relations, dim))  # not include interact
        self.item_pre = nn.Parameter(item_pre)  # [n_relations - 1, dim,dim]

        item_pre1 = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.item_pre1 = nn.Parameter(item_pre1)  # [n_relations - 1, dim,dim]

        weight = initializer(torch.empty(dim, dim))  # not include interact
        # weight = initializer(torch.empty(n_relations, dim))  # not include interact
        # weight = initializer(torch.empty(n_relations, dim))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, dim,dim]

        bais = initializer(torch.empty(1, dim))  # not include interact
        self.bais = nn.Parameter(bais)  # [n_relations - 1, dim,dim]

        self.entity_linear = nn.Linear(int(n_relations * dim), dim, bias=True)  # [n_relations - 1, dim,dim]
        # self.entity_linear = nn.Linear(int(len(self.new_r_all) * dim), dim, bias=False)  # [n_relations - 1, dim,dim]
        initializer(self.entity_linear.weight)

        # self.entity_linear_user = nn.Linear(int(n_relations * dim), dim, bias=False)  # [n_relations - 1, dim,dim]
        self.entity_linear_user = nn.Linear(int(len(self.new_r_all) * dim), dim,
                                            bias=False)  # [n_relations - 1, dim,dim]
        initializer(self.entity_linear_user.weight)

        self.entity_linear_user1 = nn.Linear(int(len(self.new_r_all) * dim), dim,
                                             bias=False)  # [n_relations - 1, dim,dim]
        initializer(self.entity_linear_user1.weight)

        self.act_atr1 = nn.LeakyReLU()
        self.act_atr2 = nn.LeakyReLU()
        self.act_atr3 = nn.LeakyReLU()
        self.act_atr4 = nn.LeakyReLU()
        self.act_atr5 = nn.LeakyReLU()
        self.act_atr6 = nn.LeakyReLU()

    def cal_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        # att_tail = edges.src['node'] + self.W_r.unsqueeze(0)
        # att_tail = edges.src['node'] + self.W_r.unsqueeze(0)
        return {'atr': att_tail}

    def cal_attribute2(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def cal_att_i2u(self, edges):
        att_tai1 = torch.cosine_similarity(edges.dst['node'], edges.src['node'], dim=1)
        # att_tai1 = torch.cosine_similarity(torch.matmul(edges.dst['node'].unsqueeze(1), self.W_r).squeeze(1),
        #                                    edges.src['node'], dim=1)
        # att_tai1 = torch.cosine_similarity(edges.src['node'], edges.dst['node'], dim=1)
        # att_tai1 = torch.cosine_similarity(edges.src['node'], edges.dst['node'], dim=1)*5
        # att_tai1 = torch.cosine_similarity(edges.src['node'], edges.dst['node'], dim=1)
        att_tail = torch.exp(att_tai1)
        # a = torch.exp(att_tail)
        # a1 = torch.exp(att_tai2)
        # b = F.normalize(edges.src['node'], dim=1, p=1)
        # torch.sum(b[0,] * b[0,])
        return {'att': att_tail}

    def cal_att_i2u_1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']
        return {'att1': att_tail}

    def cal_att_i2u_het(self, edges):
        att_tai1 = torch.cosine_similarity(edges.dst['node'], edges.src['node{}'.format(self.i)], dim=1)
        # att_tai1_ = torch.cosine_similarity(edges.dst['node_'], edges.src['node{}'.format(self.i)], dim=1)
        # att_tail_1 = torch.matmul(edges.dst['node'].unsqueeze(1), self.W_r).squeeze(1)
        # att_tai1 = torch.cosine_similarity(att_tail_1, edges.src['node{}'.format(self.i)], dim=1)

        # att_tail_1 = self.act_atr5(edges.dst['node'] + self.W_r.unsqueeze(0)) + \
        #              self.act_atr6(edges.dst['node'] * self.W_r.unsqueeze(0))
        # att_tai1 = torch.cosine_similarity(att_tail_1, edges.src['node{}'.format(self.i)], dim=1)

        att_tail = torch.exp(att_tai1 * kkk)
        # att_tail_ = torch.exp(att_tai1_)
        return {'att{}'.format(self.i): att_tail}
        # return {'att{}'.format(self.i): att_tail, 'att_{}'.format(self.i): att_tail_}

    def cal_att_i2u_het_diff(self, edges):
        att_tai1 = torch.cosine_similarity(
            edges.dst['node{}'.format(self.i)],
            edges.src['node{}'.format(self.i)], dim=1)
        # att_tai1 = torch.sum(
        #     torch.mul(
        #         edges.dst['node{}'.format(self.i)],
        #         edges.src['node{}'.format(self.i)]),
        #     dim=1)

        att_tail = torch.exp(att_tai1)
        return {'att{}'.format(self.i): att_tail}

    def cal_att_i2u_1_het(self, edges):
        att_tail = edges.data['att{}'.format(self.i)] / edges.dst['sum_att{}'.format(self.i)]
        # att_tail_ = edges.data['att_{}'.format(self.i)] / edges.dst['sum_att_{}'.format(self.i)]
        # return {'att1{}'.format(self.i): att_tail, 'att1_{}'.format(self.i): att_tail_}
        return {'att1{}'.format(self.i): att_tail}

    def cal_att_i2u_1_het_diff(self, edges):
        att_tail = edges.data['att{}'.format(self.i)] / edges.dst['sum_att{}'.format(self.i)]
        return {'att1{}'.format(self.i): att_tail}

    def cal_iu_attribute(self, edges):
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        # att_tail = edges.src['node'] + self.W_r.unsqueeze(0)
        # att_tai2 = torch.exp(torch.cosine_similarity(att_tail, edges.dst['node{}'.format(self.r)], dim=1))
        # att_tai2 = torch.exp(torch.cosine_similarity(att_tail, edges.dst['node{}'.format(self.r)], dim=1))
        att_tai2 = torch.exp(torch.cosine_similarity(att_tail, edges.dst['node'], dim=1))
        # a = torch.exp(torch.cosine_similarity(att_tail, edges.dst['node{}'.format(self.r)], dim=1) * 10)
        # a = torch.cosine_similarity(att_tail, edges.dst['node{}'.format(self.r)], dim=1) * 10
        # att_tai2 = torch.exp(torch.sum(torch.mul(att_tail, edges.dst['node{}'.format(self.r)]), dim=1))
        return {'atr': att_tail, 'att': att_tai2}

    def cal_attribute1(self, edges):
        # att_tail = torch.matmul(
        #     edges.src['node'].unsqueeze(1) * edges.dst['node'].unsqueeze(1),
        #     self.W_r).squeeze(1)
        att_tail = torch.matmul(edges.src['node'], self.W_r)  # [edge,relation,dim]
        # att_tai2 = torch.exp(
        #     torch.cosine_similarity(att_tail, edges.dst['node'][:, self.r, :].unsqueeze(1), dim=2))  # [edge,relation]
        # return {'atr': att_tail, 'att': att_tai2}
        return {'atr': att_tail}

    def cal_iu_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]  # [edge,relation]
        return {'att1': att_tail}

    def cal_hete_attribute(self, edges):
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        return {'atr': att_tail}

    def cal_user_attribute(self, edges):
        # att_tail = edges.src['node'] * self.W_r.unsqueeze(0)
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) + self.b_r
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        att_tail = torch.matmul(self.weight_user_pre_c, att_tail.unsqueeze(1))
        return {'atr': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(2) * edges.data['atr']}

    def sum_e(self, nodes):
        return {'node{}'.format(self.r): torch.sum(nodes.mailbox['neib'], dim=1)}

    def udf_u_mul_u(self, nodes):
        return {'node1': torch.matmul(self.user_embed[nodes.data['user']], self.W_atr2pre)}

    def cal_att2pre_attention(self, edges):
        att = torch.matmul(edges.src['node'].unsqueeze(1), edges.dst['node1'].unsqueeze(2)).squeeze(1)
        return {'att_atr': att}

    def cal_pre2user1(self, edges):
        # att_tail = torch.matmul(edges.dst['node'], self.W_pre2user)
        att_tail = edges.dst['node']
        att_tail = torch.matmul(edges.data['atr'].unsqueeze(1), att_tail.unsqueeze(2)).squeeze(1)
        return {'att_atr': att_tail}

    def edge_softmax_fix(self, graph, score):
        graph = graph.local_var()
        graph.edata['out'] = score
        graph.edata['out'] = torch.exp(graph.edata['out'])
        graph.update_all(dgl.function.copy_e('out', 'temp'), self.reduce_sum)
        graph.apply_edges(dgl.function.e_div_v('out', 'out_sum', 'out'))
        out = graph.edata['out']
        return out

    def reduce_sum(self, nodes):
        accum = torch.sum(nodes.mailbox['temp'], 1)
        return {'out_sum': accum}

    def forward(self, sub_idx1, graph, graph_i2u, graph_u2u, graph_i2u_0, graph_i2u_1,
                user_emb, user_emb1, entity_emb,
                W_R, w_R, b_R, new_r_all, interact_mat,
                dropout):
        # """KG aggregate"""
        entity_emb = F.normalize(entity_emb, dim=2)

        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb
        for i in range(self.n_relations):
            self.W_r = W_R[i]
            self.w_r = w_R[i]
            self.b_r = b_R[i]
            self.r = i
            graph.apply_edges(self.cal_attribute1, etype=i)
            # graph.update_all(
            #     dgl.function.copy_e('att', 'temp'),
            #     dgl.function.sum('temp', 'sum_att{}'.format(i)), etype=i)
            # graph.apply_edges(self.cal_iu_attribute1, etype=i)
            # graph.update_all(
            #     self.udf_e_mul_e,
            #     self.sum_e, etype=i)
            graph.update_all(
                dgl.function.copy_e('atr', 'neib'),
                self.sum_e, etype=i)
        entity_emb_f = []
        for i in range(self.n_relations):
            entity_emb_f.append(graph.nodes['item'].data['node{}'.format(i)])
        entity_emb_f = torch.cat(entity_emb_f, dim=2) \
            .reshape(self.n_entities, self.n_relations, self.n_relations, self.dim)
        entity_emb_f = torch.sum(entity_emb_f, dim=2)  # [entity,relation,dim]

        item_emb_f = entity_emb_f[:self.n_items]

        graph_i2u_0 = graph_i2u_0.local_var()
        for i, ii in enumerate(new_r_all):
            graph_i2u_0.nodes['item'].data['node{}'.format(ii)] = item_emb_f[:, ii, :]
            graph_i2u_0.nodes['user'].data['node{}'.format(ii)] = \
                self.user_pre[sub_idx1[0], i, :]
        for i in new_r_all:
            self.W_r = W_R[i]
            self.w_r = w_R[i]
            self.i = i
            graph_i2u_0.apply_edges(self.cal_att_i2u_het_diff, etype=0)
            graph_i2u_0.update_all(dgl.function.copy_e('att{}'.format(self.i), 'temp'),
                                   dgl.function.sum('temp', 'sum_att{}'.format(self.i)), etype=0)
            graph_i2u_0.apply_edges(self.cal_att_i2u_1_het_diff, etype=0)
            graph_i2u_0.update_all(
                dgl.function.u_mul_e('node{}'.format(self.i), 'att1{}'.format(self.i), 'send'),
                dgl.function.sum('send', 'nodenew{}'.format(self.i)),
                etype=0)
            graph_i2u_0.nodes['user'].data['nodenew{}'.format(self.i)] = \
                graph_i2u_0.nodes['user'].data['nodenew{}'.format(self.i)] + \
                graph_i2u_0.nodes['user'].data['node{}'.format(self.i)]

        user = []
        for i in new_r_all:
            user.append(graph_i2u_0.nodes['user'].data['nodenew{}'.format(i)])
        user = torch.cat(user, dim=1).reshape(len(sub_idx1[0]), -1, self.dim)

        # a0 = graph_i2u_0.edges[('item', 0, 'user')].data['att10']
        # a1 = graph_i2u_0.edges[('item', 0, 'user')].data['att110']
        # a = graph_i2u_0.edges(etype=('item', 0, 'user'), order='eid')
        # a4 = torch.where(a[1] == 0)[0]
        # b = a1[a4]
        # b0 = a0[a4]
        # print(b)

        """next"""
        entity_next = entity_emb_f
        user_next = torch.sum(user, dim=1)

        entity = F.normalize(entity_emb_f[:, self.new_r_all, :], dim=2). \
            reshape(self.n_entities, -1)
        user = F.normalize(user.reshape(len(sub_idx1[0]), -1, self.dim), dim=2). \
            reshape(len(sub_idx1[0]), -1)

        user = user[sub_idx1[1]]

        return entity, entity_next, user, user_next


class Aggregator_KGUI3_11(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dropuot):
        super(Aggregator_KGUI3_11, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot

    def cal_attribute(self, edges):

        # """useful1"""
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1) \
        #            * torch.matmul(edges.dst['node'].unsqueeze(1), self.W_r).squeeze(1)
        # att_tail_1 = torch.matmul(edges.dst['node'].unsqueeze(1), self.W_r).squeeze(1)
        # att_tai2 = torch.exp(torch.cosine_similarity(att_tail, att_tail_1, dim=1))
        # return {'atr': att_tail, 'att': att_tai2}
        # """useful1"""

        """useful1"""
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        att_tail_1 = torch.matmul(edges.dst['node'].unsqueeze(1), self.W_r).squeeze(1)
        att_tai2 = torch.exp(torch.cosine_similarity(att_tail, att_tail_1, dim=1))
        return {'atr': att_tail, 'att': att_tai2}
        """useful1"""

        """2"""
        # att_tail = edges.src['node']
        # att_tail_1 = edges.dst['node']
        # # att_tai2 = torch.exp(torch.cosine_similarity(
        # #     torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1), att_tail_1, dim=1))
        # att_tai2 = torch.exp(torch.cosine_similarity(att_tail, att_tail_1, dim=1))
        # return {'atr': att_tail, 'att': att_tai2}
        """2"""

    def cal_attribute_noatt(self, edges):
        att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)
        return {'atr': att_tail}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def forward(self, graph, entity_emb, W_R, new_r_all):

        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb
        for i in range(self.n_relations):
            self.W_r = W_R[i]
            self.r = i
            "att"
            # graph.apply_edges(self.cal_attribute, etype=i)
            # graph.update_all(
            #     dgl.function.copy_e('att', 'temp'),
            #     dgl.function.sum('temp', 'sum_att{}'.format(i)), etype=i)
            # graph.apply_edges(self.cal_attribute1, etype=i)
            # graph.update_all(
            #     self.udf_e_mul_e,
            #     dgl.function.sum('neib', 'node{}'.format(i)), etype=i)
            "mean1"
            # graph.apply_edges(self.cal_attribute_noatt, etype=i)
            # graph.update_all(
            #     dgl.function.copy_e('atr', 'temp'),
            #     dgl.function.mean('temp', 'node{}'.format(i)), etype=i)
            "mean2"
            graph.update_all(
                dgl.function.copy_u('node', 'temp'),
                dgl.function.mean('temp', 'node{}'.format(i)), etype=i)

        entity_emb_f = []
        for i in range(self.n_relations):
            entity_emb_f.append(graph.nodes['item'].data['node{}'.format(i)])
        entity_emb_f = torch.cat(entity_emb_f, dim=1) \
            .reshape(self.n_entities, self.n_relations, self.dim)

        entity_next = torch.sum(entity_emb_f, dim=1)

        return entity_emb_f[self.n_items, new_r_all, :], entity_next


class Aggregator_KGUI3_12(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_entities, n_items, n_relations, dim, dropuot):
        super(Aggregator_KGUI3_12, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations
        self.dropout_rate = dropuot

    def cal_attribute(self, edges):
        att = torch.exp(torch.cosine_similarity(edges.src['node'], edges.dst['node'], dim=1))
        return {'att': att}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def cal_att_ii(self, edges):
        att = torch.cosine_similarity(self.item_emb_0[edges.src['id']], self.item_emb_0[edges.dst['id']], dim=1)
        att = torch.exp(att)
        return {'att': att}

    def cal_att_ii1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att0']
        return {'att1': att_tail}

    def cal_att_ui(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att1']
        return {'att1': att_tail}

    def cal_att_iu(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att0']
        return {'att1': att_tail}

    def forward(self, graph_UIS, item_emb, user_emb, item_emb_0):
        graph_UIS = graph_UIS.local_var()
        graph_UIS.nodes['user'].data['node'] = user_emb
        graph_UIS.nodes['item'].data['node'] = item_emb

        """item to item"""
        self.item_emb_0 = item_emb_0
        graph_UIS.apply_edges(self.cal_att_ii, etype=0)
        graph_UIS.update_all(dgl.function.copy_e('att', 'send'),
                             dgl.function.sum('send', 'sum_att0'), etype=0)
        graph_UIS.apply_edges(self.cal_att_ii1, etype=0)
        graph_UIS.update_all(dgl.function.u_mul_e('node', 'att1', 'send1'),
                             dgl.function.sum('send1', 'node0'), etype=0)

        # graph_UIS.apply_edges(self.cal_attribute, etype=0)
        # graph_UIS.update_all(dgl.function.copy_e('att', 'send'),
        #                      dgl.function.sum('send', 'sum_att0'), etype=0)
        # graph_UIS.apply_edges(self.cal_att_ii, etype=0)
        # graph_UIS.update_all(dgl.function.u_mul_e('node', 'att1', 'send1'),
        #                      dgl.function.sum('send1', 'node0'), etype=0)
        """user to item"""
        graph_UIS.apply_edges(self.cal_attribute, etype=1)
        graph_UIS.update_all(dgl.function.copy_e('att', 'send'),
                             dgl.function.sum('send', 'sum_att1'), etype=1)
        graph_UIS.apply_edges(self.cal_att_ui, etype=1)
        graph_UIS.update_all(dgl.function.u_mul_e('node', 'att1', 'send1'),
                             dgl.function.sum('send1', 'node1'), etype=1)
        """item to user"""
        graph_UIS.apply_edges(self.cal_attribute, etype=2)
        graph_UIS.update_all(dgl.function.copy_e('att', 'send'),
                             dgl.function.sum('send', 'sum_att0'), etype=2)
        graph_UIS.apply_edges(self.cal_att_iu, etype=2)
        graph_UIS.update_all(dgl.function.u_mul_e('node', 'att1', 'send1'),
                             dgl.function.sum('send1', 'node0'), etype=2)

        item = 0.5 * graph_UIS.nodes['item'].data['node0'] + 0.5 * graph_UIS.nodes['item'].data['node1']
        # item = graph_UIS.nodes['item'].data['node1']
        user = graph_UIS.nodes['user'].data['node0']

        return user, item


class Aggregator_UMIKGAN_item(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_UMIKGAN_item, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute(self, edges):
        att_tail = edges.src['node'] * self.W_r
        # att_tail_1 = edges.dst['node'] * self.W_r
        # att_tai2 = torch.exp(torch.cosine_similarity(att_tail, att_tail_1, dim=1))

        att_tai2 = torch.exp(torch.cosine_similarity(att_tail, edges.dst['node'], dim=1))

        return {'atr': att_tail, 'att': att_tai2}

    def cal_attribute_noatt(self, edges):
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)

        att_tail = edges.src['node'] * self.W_r
        # att_tail = edges.src['node'] * self.W_r * edges.dst['node'] * self.W_r

        # att_tail = edges.dst['node'] * edges.src['node'] * self.W_r
        return {'atr': att_tail}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def forward(self, graph, entity_emb, W_R):

        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb
        for i in range(self.n_relations):
            self.W_r = W_R[i].unsqueeze(0)
            self.r = i
            "att"
            # graph.apply_edges(self.cal_attribute, etype=i)
            # graph.update_all(
            #     dgl.function.copy_e('att', 'temp'),
            #     dgl.function.sum('temp', 'sum_att{}'.format(i)), etype=i)
            # graph.apply_edges(self.cal_attribute1, etype=i)
            # graph.update_all(
            #     self.udf_e_mul_e,
            #     dgl.function.sum('neib', 'node{}'.format(i)), etype=i)
            "mean1"
            graph.apply_edges(self.cal_attribute_noatt, etype=i)
            graph.update_all(
                dgl.function.copy_e('atr', 'temp'),
                dgl.function.mean('temp', 'node{}'.format(i)), etype=i)

        """out"""
        entity_emb_f = graph.nodes['item'].data['node0']
        for i in range(1, self.n_relations):
            entity_emb_f = entity_emb_f + graph.nodes['item'].data['node{}'.format(i)]
        # entity_emb_f = entity_emb_f / self.n_relations

        return entity_emb_f


class Aggregator_UPKGAN_item(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_UPKGAN_item, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute_noatt(self, edges):
        att_tail = edges.src['node'] * self.W_r[edges.data['type']]

        return {'atr': att_tail}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def forward(self, graph, entity_emb, W_R):
        graph = graph.local_var()
        graph.ndata['node'] = entity_emb

        self.W_r = W_R
        graph.apply_edges(self.cal_attribute_noatt)
        graph.update_all(
            dgl.function.copy_e('atr', 'temp'),
            dgl.function.mean('temp', 'node'))

        """out"""
        entity_emb_f = graph.ndata['node']

        return entity_emb_f


class Aggregator_UMIKGAN_user(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_UMIKGAN_user, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute(self, edges):
        att_tail = edges.src['node'] * self.W_r
        # att_tail = torch.matmul(edges.src['node'].unsqueeze(1), self.W_r).squeeze(1)

        # att_tail2 = self.common_like * self.W_r

        # att_tai2 = torch.exp(torch.sum(att_tail * self.common_like, dim=1))
        att_tai2 = torch.exp(torch.cosine_similarity(att_tail, self.common_like, dim=1))
        # att_tai2 = torch.exp(torch.cosine_similarity(att_tail, att_tail2, dim=1))
        return {'atr': att_tail, 'att': att_tai2}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att'] \
                   * edges.dst['deg_all']
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def cal_attribute_UI(self, edges):
        # att_tai2 = torch.exp(torch.sum(edges.src['node'] * self.common_like, dim=1))
        att_tai2 = torch.exp(torch.cosine_similarity(edges.src['node'], self.common_like, dim=1))
        return {'att': att_tai2}

    def cal_attribute1_UI(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']
        return {'att1': att_tail}

    def forward(self, graph, graph_UIS, entity_emb, W_R, common_like):
        self.common_like = common_like.unsqueeze(0)

        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb

        for i in range(self.n_relations):
            self.W_r = W_R[i]
            self.r = i
            graph.apply_edges(self.cal_attribute, etype=i)
            graph.update_all(
                dgl.function.copy_e('att', 'temp'),
                dgl.function.sum('temp', 'sum_att{}'.format(i)), etype=i)

        graph.nodes['item'].data['sum_att'] = graph.nodes['item'].data['sum_att0']
        for i in range(1, self.n_relations):
            graph.nodes['item'].data['sum_att'] = \
                graph.nodes['item'].data['sum_att'] + \
                graph.nodes['item'].data['sum_att{}'.format(i)]

        for i in range(self.n_relations):
            graph.apply_edges(self.cal_attribute1, etype=i)
            graph.update_all(
                self.udf_e_mul_e,
                dgl.function.mean('neib', 'node{}'.format(i)), etype=i)
            # graph.update_all(
            #     dgl.function.copy_e('atr', 'neib'),
            #     dgl.function.mean('neib', 'node{}'.format(i)), etype=i)

        entity_emb_f = graph.nodes['item'].data['node0']
        for i in range(1, self.n_relations):
            entity_emb_f = entity_emb_f + graph.nodes['item'].data['node{}'.format(i)]

        """cal user"""
        graph_UIS = graph_UIS.local_var()
        graph_UIS.nodes['item'].data['node'] = entity_emb_f[:self.n_items]
        graph_UIS.apply_edges(self.cal_attribute_UI, etype=1)
        graph_UIS.update_all(
            dgl.function.copy_e('att', 'temp'),
            dgl.function.sum('temp', 'sum_att'), etype=1)
        graph_UIS.apply_edges(self.cal_attribute1_UI, etype=1)
        graph_UIS.update_all(
            dgl.function.u_mul_e('node', 'att1', 'neib'),
            dgl.function.sum('neib', 'node'), etype=1)
        user_emb_f = graph_UIS.nodes['user'].data['node']

        return entity_emb_f, user_emb_f


class Aggregator_UMIKGAN_user0(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_UMIKGAN_user0, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute01(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def cal_attribute(self, edges):
        "agg"
        att_tail = edges.src['node'] * self.W_r

        "att user_atr"
        att_tai2 = torch.exp(torch.sum(att_tail * self.common_like, dim=1))

        return {'atr': att_tail, 'att': att_tai2}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)] \
                   * self.att_r
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def cal_attribute_UI(self, edges):
        # att_tai2 = torch.exp(torch.sum(edges.src['node'] * self.common_like, dim=1))
        att_tai2 = torch.exp(torch.sum(edges.src['node'] * edges.dst['node'], dim=1))
        return {'att': att_tai2}

    def cal_attribute1_UI(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']
        return {'att1': att_tail}

    def forward(self, graph, graph_UIS, entity_emb, W_R, common_like, att_r, user_emb):
        self.common_like = common_like.unsqueeze(0)

        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb

        """1 entity agg through same realtion"""
        for i in range(self.n_relations):
            self.W_r = W_R[i]
            self.r = i
            self.att_r = att_r[i]

            graph.apply_edges(self.cal_attribute, etype=i)

            graph.update_all(
                dgl.function.copy_e('att', 'temp'),
                dgl.function.sum('temp', 'sum_att{}'.format(i)), etype=i)

            graph.apply_edges(self.cal_attribute1, etype=i)

            graph.update_all(
                self.udf_e_mul_e,
                dgl.function.sum('neib', 'node{}'.format(i)), etype=i)

        """1 entity agg through different realtion"""
        entity_emb_f = graph.nodes['item'].data['node0']
        for i in range(1, self.n_relations):
            entity_emb_f = entity_emb_f + graph.nodes['item'].data['node{}'.format(i)]

        """cal user"""
        graph_UIS = graph_UIS.local_var()
        graph_UIS.nodes['item'].data['node'] = entity_emb_f[:self.n_items]

        # graph_UIS.update_all(
        #     dgl.function.copy_u('node', 'temp0'),
        #     dgl.function.mean('temp0', 'node0'), etype=1)

        graph_UIS.nodes['user'].data['node'] = user_emb * self.common_like
        # graph_UIS.nodes['user'].data['node'] = graph_UIS.nodes['user'].data['node0'] \
        #                                        * self.common_like

        graph_UIS.apply_edges(self.cal_attribute_UI, etype=1)
        graph_UIS.update_all(
            dgl.function.copy_e('att', 'temp'),
            dgl.function.sum('temp', 'sum_att'), etype=1)
        graph_UIS.apply_edges(self.cal_attribute1_UI, etype=1)
        graph_UIS.update_all(
            dgl.function.u_mul_e('node', 'att1', 'neib'),
            dgl.function.sum('neib', 'node'), etype=1)
        user_emb_f = graph_UIS.nodes['user'].data['node']
        # user_emb_f = graph_UIS.nodes['user'].data['node'] + graph_UIS.nodes['user'].data['node0']

        return entity_emb_f, user_emb_f


class Aggregator_MUIGAN(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_MUIGAN, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute01(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def cal_attribute(self, edges):
        "agg"
        att_tail = edges.src['node'] * self.W_r * self.att_r[self.r]

        # att_tai2 = torch.exp(torch.sum(att_tail * edges.dst['node'] * self.common_like, dim=1))
        # att_tai2 = self.common_like[self.r].expand(att_tail.shape[0])

        return {'atr': att_tail}
        # return {'atr': att_tail, 'att': att_tai2}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)] \
                   * self.att_r
        # att_tail = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att'].unsqueeze(1) * edges.data['atr']}

    def cal_attribute_UI(self, edges):
        # att_tai2 = torch.exp(torch.sum(edges.src['node'] * self.common_like, dim=1))
        att_tai2 = torch.exp(torch.sum(edges.src['node'] * edges.dst['node'], dim=1))
        return {'att': att_tai2}

    def cal_attribute1_UI(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']
        return {'att1': att_tail}

    def forward(self, graph, graph_UIS, entity_emb, W_R, common_like, att_r, user_emb):
        self.att_r = att_r
        self.common_like = common_like.unsqueeze(0)

        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb

        """1 entity agg through same realtion"""
        for i in range(self.n_relations):
            self.W_r = W_R[i].unsqueeze(0)
            self.r = i

            graph.apply_edges(self.cal_attribute, etype=i)

            graph.update_all(
                dgl.function.copy_e('atr', 'neib'),
                dgl.function.mean('neib', 'node{}'.format(i)), etype=i)

        """1 entity agg through different realtion"""
        entity_emb_f = graph.nodes['item'].data['node0']
        for i in range(1, self.n_relations):
            entity_emb_f = entity_emb_f + graph.nodes['item'].data['node{}'.format(i)]

        """cal user"""
        graph_UIS = graph_UIS.local_var()
        graph_UIS.nodes['item'].data['node'] = entity_emb[:self.n_items, :]
        graph_UIS.nodes['user'].data['node'] = user_emb * self.common_like
        graph_UIS.apply_edges(self.cal_attribute_UI, etype=1)
        graph_UIS.update_all(
            dgl.function.copy_e('att', 'temp'),
            dgl.function.sum('temp', 'sum_att'), etype=1)
        graph_UIS.apply_edges(self.cal_attribute1_UI, etype=1)
        graph_UIS.update_all(
            dgl.function.u_mul_e('node', 'att1', 'neib'),
            dgl.function.sum('neib', 'node'), etype=1)
        user_emb_f = graph_UIS.nodes['user'].data['node']

        return entity_emb_f, user_emb_f


class Aggregator_MAKG(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_MAKG, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute(self, edges):
        att_tail = edges.src['node'] * self.W_R[edges.data['type']] \
                   * self.atr_att[edges.data['type']].unsqueeze(1)
        return {'atr': att_tail}

    def cal_attribute1(self, edges):
        # att_tail = edges.src['node']
        # att_tail = edges.src['node'] * self.r_att[self.r]

        # att_tail = edges.src['node'] * self.r_att[self.r]
        att_tail = (edges.src['node'] + self.W_r) * self.r_att[self.r]
        # att_tail = (edges.src['node'] * self.W_r) * self.r_att[self.r]
        # att_tail = edges.src['node'] * self.W_r

        # att_tail = edges.src['node'] / edges.dst['deg'].unsqueeze(-1) \
        #            * torch.exp(torch.sum(torch.mul(self.W_r, self.user_like)))
        # att_tail = edges.src['node'] * self.W_r / edges.dst['deg'].unsqueeze(-1) \
        #            * torch.exp(10 * torch.cosine_similarity(self.W_r, self.user_like, dim=1))
        # att_tail = edges.src['node'] * self.W_r \
        #            * torch.exp(10 * torch.cosine_similarity(self.W_r, self.user_like, dim=1))
        # att_tail = edges.src['node'] \
        #            * torch.exp(10 * torch.cosine_similarity(self.W_r, self.user_like, dim=1))

        # att1 = torch.exp(torch.sum(self.common_like * edges.src['node'], dim=1))
        # att1 = torch.exp(torch.cosine_similarity(self.common_like,
        #                                          edges.src['node'] * self.W_r, dim=1))
        # att1 = torch.exp(torch.cosine_similarity(self.common_like, edges.src['node'], dim=1))
        # att1 = torch.exp(torch.cosine_similarity(self.common_like * edges.dst['node'], edges.src['node'], dim=1))
        # att1 = torch.exp(torch.cosine_similarity(
        #     self.common_like * edges.dst['node'], self.common_like * edges.src['node'], dim=1))
        # att1 = torch.exp(torch.cosine_similarity(edges.dst['node'], edges.src['node'], dim=1))
        return {'atr': att_tail}
        # return {'atr': att_tail, 'att': att1}

    def cal_att(self, edges):
        att = edges.data['att'] / edges.dst['sum_att{}'.format(self.r)]
        return {'att1': att}

    def ud_e_mul_e(self, edges):
        node = edges.data['att1'].unsqueeze(1) * edges.data['atr']
        return {'neib': node}

    def forward(self, graph, entity_emb, W_R, exp_deg):

        """het"""
        # self.common_like = common_like.unsqueeze(0)
        self.r_att = exp_deg
        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb

        """1 entity agg through same realtion"""
        for i in range(self.n_relations):
            self.W_r = W_R[i].unsqueeze(0)
            self.r = i

            graph.apply_edges(self.cal_attribute1, etype=i)

            graph.update_all(
                dgl.function.copy_e('atr', 'temp0'),
                dgl.function.mean('temp0', 'node{}'.format(i)), etype=i)


            # graph.update_all(
            #     dgl.function.copy_e('att', 'temp0'),
            #     dgl.function.sum('temp0', 'sum_att{}'.format(i)), etype=i)
            #
            # graph.apply_edges(self.cal_att, etype=i)
            #
            # graph.update_all(
            #     self.ud_e_mul_e,
            #     dgl.function.sum('neib', 'node{}'.format(i)), etype=i)

        """1 entity agg through different realtion"""
        entity_emb_f = graph.nodes['item'].data['node0']
        for i in range(1, self.n_relations):
            entity_emb_f = entity_emb_f + graph.nodes['item'].data['node{}'.format(i)]

        # entity_emb_f = entity_emb_f
        # entity_emb_f = entity_emb_f + entity_emb

        return entity_emb_f


class Aggregator_MAKG1(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_MAKG1, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute1(self, edges):
        # att_tail = edges.src['node']
        att_tail = edges.src['node'] + self.W_r
        # att_tail = edges.src['node'] * self.W_r
        return {'atr': att_tail}

    def forward(self, graph, entity_emb, W_R):

        """het"""
        graph = graph.local_var()
        graph.nodes['item'].data['node'] = entity_emb

        """1 entity agg through same realtion"""
        for i in range(self.n_relations):
            self.W_r = W_R[i].unsqueeze(0)
            self.r = i

            graph.apply_edges(self.cal_attribute1, etype=i)

            graph.update_all(
                dgl.function.copy_e('atr', 'temp0'),
                dgl.function.mean('temp0', 'node{}'.format(i)), etype=i)

        """1 entity agg through different realtion"""
        entity_emb_f = graph.nodes['item'].data['node0']
        for i in range(1, self.n_relations):
            entity_emb_f = entity_emb_f + graph.nodes['item'].data['node{}'.format(i)]

        return entity_emb_f


class Aggregator_I2U(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_I2U, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute_UI0(self, edges):
        # atr1 = torch.cosine_similarity(edges.src['node'], edges.dst['node'].unsqueeze(1), dim=2)
        atr1 = torch.sum(edges.src['node'] * edges.dst['node'].unsqueeze(1), dim=2)
        return {'att': atr1}

    def cal_attribute_UI00(self, edges):
        # idx = torch.tensor(list(range(len(edges))), device=edges.src['node'].device)
        # edges.src['node'][idx, edges.dst['label'], :] = \
        #     edges.src['node'][idx, edges.dst['label'], :] + edges.dst['node']
        # edges.src['node'][idx, edges.dst['label'], :] = \
        #     edges.src['node'][idx, edges.dst['label'], :] + self.common_like[edges.dst['label'], :]
        # atr = edges.src['node'] * edges.dst['att'].unsqueeze(-1)

        # atr = F.normalize(edges.src['node'] + edges.dst['node'].unsqueeze(1), dim=2)
        # atr = atr * edges.dst['att'].unsqueeze(-1)
        # atr = (edges.src['node'] + edges.dst['node'].unsqueeze(1)) * edges.dst['att'].unsqueeze(-1)

        atr = torch.matmul(edges.dst['att'].unsqueeze(1), edges.src['node']).squeeze(1)

        # atr = atr.reshape(len(edges), -1)
        return {'atr': atr}

    def cal_attribute_UI(self, edges):
        # atr = torch.sum(edges.src['node'] * edges.dst['att'].unsqueeze(-1), dim=1)

        atr1 = torch.cosine_similarity(edges.src['node'], edges.dst['node'].unsqueeze(1), dim=2)
        atr2 = nn.Softmax(dim=1)(atr1)
        atr = (edges.src['node'] + edges.dst['node'].unsqueeze(1)) * atr2.unsqueeze(-1)
        atr = atr.reshape(len(edges), -1)

        # atr11 = torch.exp(torch.cosine_similarity(atr, edges.dst['node'], dim=1))

        return {'atr': atr, 'att': atr2}
        # return {'atr': atr, 'att': atr11}

    def cal_attribute_UI1(self, edges):
        att = edges.data['att'] / edges.dst['sumatt']

        return {'att1': att}

    def udf_e_mul_e(self, edges):
        node = edges.data['att1'].unsqueeze(1) * edges.data['atr']
        return {'neib': node}

    def forward(self, graph_UIS, entity_emb_atr, entity_embed, user_embed, user_label, common_like):
        """cal user"""
        # self.common_like = common_like
        graph_UIS = graph_UIS.local_var()
        graph_UIS.nodes['item'].data['node0'] = entity_embed[:self.n_items]
        graph_UIS.nodes['item'].data['node'] = entity_emb_atr[:self.n_items]
        graph_UIS.nodes['user'].data['node'] = user_embed
        # graph_UIS.nodes['user'].data['label'] = user_label

        graph_UIS.apply_edges(self.cal_attribute_UI0, etype=1)
        graph_UIS.update_all(
            dgl.function.copy_e('att', 'temp1'),
            dgl.function.mean('temp1', 'sumatt0'), etype=1)
        att_all = graph_UIS.nodes['user'].data['sumatt0']
        att_all = nn.Softmax(dim=1)(att_all)
        graph_UIS.nodes['user'].data['att'] = att_all
        graph_UIS.apply_edges(self.cal_attribute_UI00, etype=1)

        graph_UIS.update_all(
            dgl.function.copy_e('atr', 'temp0'),
            dgl.function.mean('temp0', 'node'), etype=1)

        # graph_UIS.update_all(
        #     dgl.function.copy_u('node0', 'temp2'),
        #     dgl.function.mean('temp2', 'node1'), etype=1)

        user_emb_f = graph_UIS.nodes['user'].data['node']
        # user_emb_f = graph_UIS.nodes['user'].data['node'] + graph_UIS.nodes['user'].data['node1']
        return user_emb_f


class Aggregator_I2U0(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_I2U0, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def forward(self, graph_UIS, entity_embed, user_embed):
        """cal user"""
        graph_UIS = graph_UIS.local_var()
        graph_UIS.nodes['item'].data['node'] = entity_embed[:self.n_items]
        graph_UIS.nodes['user'].data['node'] = user_embed

        graph_UIS.update_all(
            dgl.function.copy_u('node', 'temp1'),
            dgl.function.mean('temp1', 'node'), etype=1)

        user_emb_f = graph_UIS.nodes['user'].data['node']
        return user_emb_f


class Aggregator_UPKGAN_user(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_UPKGAN_user, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute_noatt(self, edges):
        att_tail = edges.src['node'] * self.W_r[edges.data['type']]

        att_tai2 = torch.exp(torch.cosine_similarity(att_tail, self.common_like, dim=1))

        # att_tai21 = torch.exp(torch.cosine_similarity(edges.src['node'], self.common_like, dim=1))
        # att_tai22 = torch.exp(torch.cosine_similarity(self.W_r[edges.data['type']], self.common_like, dim=1))

        return {'atr': att_tail, 'att': att_tai2}
        # return {'atr': att_tail, 'att21': att_tai21, 'att22': att_tai22}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']

        # att_tail = edges.data['att21'] / edges.dst['sum_att21'] * edges.data['att22'] / edges.dst['sum_att22']

        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def cal_attribute_UI(self, edges):
        # att_tai2 = torch.exp(torch.sum(edges.src['node'] * self.common_like, dim=1))

        # att_tai2 = torch.exp(torch.cosine_similarity(edges.src['node'], self.common_like, dim=1))

        att_tai2 = torch.exp(torch.cosine_similarity(edges.src['node'], edges.dst['node'], dim=1))
        return {'att': att_tai2}

    def cal_attribute1_UI(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']
        return {'att1': att_tail}

    def forward(self, graph, graph_UIS, entity_emb, W_R, common_like, user_like):
        self.common_like = common_like.unsqueeze(0)

        """KG agg"""
        graph = graph.local_var()
        graph.ndata['node'] = entity_emb

        """similarity 1"""
        self.W_r = W_R
        graph.apply_edges(self.cal_attribute_noatt)
        graph.update_all(
            dgl.function.copy_e('att', 'temp0'),
            dgl.function.sum('temp0', 'sum_att'))
        graph.apply_edges(self.cal_attribute1)
        graph.update_all(
            self.udf_e_mul_e,
            dgl.function.sum('neib', 'node'))
        entity_emb_f = graph.ndata['node']

        """similarity 2"""
        # self.W_r = W_R
        # graph.apply_edges(self.cal_attribute_noatt)
        # graph.update_all(
        #     dgl.function.copy_e('att21', 'temp21'),
        #     dgl.function.sum('temp21', 'sum_att21'))
        # graph.update_all(
        #     dgl.function.copy_e('att22', 'temp22'),
        #     dgl.function.sum('temp22', 'sum_att22'))
        # graph.apply_edges(self.cal_attribute1)
        # graph.update_all(
        #     self.udf_e_mul_e,
        #     dgl.function.sum('neib', 'node'))
        # entity_emb_f = graph.ndata['node']

        """cal user"""
        graph_UIS = graph_UIS.local_var()
        graph_UIS.nodes['item'].data['node'] = entity_emb_f[:self.n_items]
        graph_UIS.nodes['user'].data['node'] = user_like
        graph_UIS.apply_edges(self.cal_attribute_UI, etype=1)
        graph_UIS.update_all(
            dgl.function.copy_e('att', 'temp'),
            dgl.function.sum('temp', 'sum_att'), etype=1)
        graph_UIS.apply_edges(self.cal_attribute1_UI, etype=1)
        graph_UIS.update_all(
            dgl.function.u_mul_e('node', 'att1', 'neib'),
            dgl.function.sum('neib', 'node'), etype=1)
        user_emb_f = graph_UIS.nodes['user'].data['node']

        return entity_emb_f, user_emb_f


class cal_UIKGAN_user0(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(cal_UIKGAN_user0, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def forward(self, graph_UIS, entity_emb):
        """cal user 0"""
        graph_UIS = graph_UIS.local_var()

        graph_UIS.nodes['item'].data['node'] = entity_emb[:self.n_items]

        graph_UIS.update_all(
            dgl.function.copy_u('node', 'temp'),
            dgl.function.mean('temp', 'node'), etype=1)

        user_emb = graph_UIS.nodes['user'].data['node']

        return user_emb


class Aggregator_UIKGAN_item(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_UIKGAN_item, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute_noatt(self, edges):
        att_tail = edges.src['node'] * self.W_r[edges.data['type']]

        att = torch.exp(torch.sum(torch.mul(att_tail, edges.dst['node']), dim=1))
        return {'atr': att_tail, 'att': att}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']

        return {'att1': att_tail}

    def forward(self, graph, entity_emb, W_R):
        graph = graph.local_var()
        graph.ndata['node'] = entity_emb

        self.W_r = W_R
        # graph.apply_edges(self.cal_attribute_noatt)
        # graph.update_all(
        #     dgl.function.copy_e('atr', 'temp'),
        #     dgl.function.mean('temp', 'node'))

        graph.apply_edges(self.cal_attribute_noatt)
        graph.update_all(
            dgl.function.copy_e('att', 'temp0'),
            dgl.function.mean('temp0', 'sum_att'))
        graph.apply_edges(self.cal_attribute1)
        graph.update_all(
            dgl.function.u_mul_e('node', 'att1', 'temp'),
            dgl.function.mean('temp', 'node'))

        """out"""
        entity_emb_f = graph.ndata['node']

        return entity_emb_f


class Aggregator_UIKGAN_user(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Aggregator_UIKGAN_user, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute_noatt(self, edges):
        att_tail = edges.src['node'] * self.W_r[edges.data['type']]

        # att_tai2 = torch.exp(torch.cosine_similarity(att_tail, self.common_like, dim=1))
        att_tai2 = torch.exp(torch.sum(att_tail * self.common_like, dim=1))

        return {'atr': att_tail, 'att': att_tai2}
        # return {'atr': att_tail, 'att21': att_tai21, 'att22': att_tai22}

    def cal_attribute1(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']

        # att_tail = edges.data['att21'] / edges.dst['sum_att21'] * edges.data['att22'] / edges.dst['sum_att22']

        return {'att1': att_tail}

    def udf_e_mul_e(self, edges):
        return {'neib': edges.data['att1'].unsqueeze(1) * edges.data['atr']}

    def cal_attribute_UI(self, edges):
        # att_tai2 = torch.exp(torch.sum(edges.src['node'] * self.common_like, dim=1))

        # att_tai2 = torch.exp(torch.cosine_similarity(edges.src['node'], self.common_like, dim=1))

        # att_tai2 = torch.exp(torch.cosine_similarity(edges.src['node'], edges.dst['node'], dim=1))
        att_tai2 = torch.exp(torch.sum(edges.src['node'] * edges.dst['node'], dim=1))
        return {'att': att_tai2}

    def cal_attribute1_UI(self, edges):
        att_tail = edges.data['att'] / edges.dst['sum_att']
        return {'att1': att_tail}

    def forward(self, graph, graph_UIS, entity_emb, W_R, common_like, user_like):
        self.common_like = common_like.unsqueeze(0)

        """KG agg"""
        graph = graph.local_var()
        graph.ndata['node'] = entity_emb

        """atr to item"""
        self.W_r = W_R
        graph.apply_edges(self.cal_attribute_noatt)
        graph.update_all(
            dgl.function.copy_e('att', 'temp0'),
            dgl.function.sum('temp0', 'sum_att'))
        graph.apply_edges(self.cal_attribute1)
        graph.update_all(
            self.udf_e_mul_e,
            dgl.function.sum('neib', 'node'))
        entity_emb_f = graph.ndata['node']

        """item to user"""
        graph_UIS = graph_UIS.local_var()
        graph_UIS.nodes['item'].data['node'] = entity_emb_f[:self.n_items]
        graph_UIS.nodes['user'].data['node'] = user_like
        graph_UIS.apply_edges(self.cal_attribute_UI, etype=1)
        graph_UIS.update_all(
            dgl.function.copy_e('att', 'temp'),
            dgl.function.sum('temp', 'sum_att'), etype=1)
        graph_UIS.apply_edges(self.cal_attribute1_UI, etype=1)
        graph_UIS.update_all(
            dgl.function.u_mul_e('node', 'att1', 'neib'),
            dgl.function.sum('neib', 'node'), etype=1)
        user_emb_f = graph_UIS.nodes['user'].data['node']

        return user_emb_f


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users,
                 n_factors, n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_factors=n_factors))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    # def _cul_cor_pro(self):
    #     # disen_T: [num_factor, dimension]
    #     disen_T = self.disen_weight_att.t()
    #
    #     # normalized_disen_T: [num_factor, dimension]
    #     normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)
    #
    #     pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
    #     ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)
    #
    #     pos_scores = torch.exp(pos_scores / self.temperature)
    #     ttl_scores = torch.exp(ttl_scores / self.temperature)
    #
    #     mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
    #     return mi_score

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = self._cul_cor()
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, latent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, self.disen_weight_att)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor


class GraphConv_KGUI(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users,
                 n_factors, n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv_KGUI, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        # disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        # self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_hops):
            self.convs.append(Aggregator_KGUI(n_users=n_users, n_factors=n_factors))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type, edge_index_user, edge_type_user,
                interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            edge_index_user, edge_type_user = self._edge_sampling(edge_index_user, edge_type_user,
                                                                  self.node_dropout_rate)
            # interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        # cor = self._cul_cor()
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb,
                                                 edge_index, edge_type, edge_index_user, edge_type_user,
                                                 self.weight)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb


class GraphConv_KGUI_linear(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, dim, n_hops, n_users, n_items, n_entities,
                 n_new_node, n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv_KGUI_linear, self).__init__()

        self.convs = nn.ModuleList()
        self.dim = dim
        self.n_hops = n_hops
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_new_node = n_new_node
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        # atr feature exaction
        # weight = initializer(torch.empty(n_relations, dim))  # not include interact
        # self.weight = nn.Parameter(weight)  # [n_relations - 1, dim,dim]
        # bias = initializer(torch.empty(n_relations, dim))  # not include interact
        # self.bias = nn.Parameter(bias)  # [n_relations - 1, dim,dim]

        weight = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, dim,dim]
        bias = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias = nn.Parameter(bias)  # [n_relations - 1, dim,dim]
        # atr_node to pre_node
        weight1 = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight1 = nn.Parameter(weight1)  # [n_relations - 1, dim,dim]
        bias1 = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias1 = nn.Parameter(bias1)  # [n_relations - 1, dim,dim]
        # atr_node to pre_node
        weight2 = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight2 = nn.Parameter(weight2)  # [n_relations - 1, dim,dim]
        bias2 = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias2 = nn.Parameter(bias2)  # [n_relations - 1, dim,dim]

        for i in range(n_hops):
            self.convs.append(
                Aggregator_KGUI_linear(n_users=n_users, n_items=n_items, n_relations=n_relations, n_new_node=n_new_node,
                                       dropuot=node_dropout_rate))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, graph, graph_atr2pre, graph_pre2user,
                user_emb, entity_emb, latent_emb,
                edge_index, edge_type,
                edge_index_user, edge_type_user,
                interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        entity_res_emb = []
        user_res_emb = []
        # entity_res_emb.append(entity_emb)
        # user_res_emb.append(user_emb)
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](graph, graph_atr2pre, graph_pre2user, user_emb, entity_emb,
                                                 edge_index, edge_type, edge_index_user, edge_type_user,
                                                 interact_mat,
                                                 self.weight, self.bias,
                                                 self.weight1, self.bias1,
                                                 self.weight2, self.bias2,
                                                 node_dropout)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            # entity_res_emb = torch.add(entity_res_emb, entity_emb)
            # user_res_emb = torch.add(user_res_emb, user_emb)

            entity_res_emb.append(entity_emb)
            user_res_emb.append(user_emb)
        entity_res_emb = \
            torch.sum(torch.cat(entity_res_emb, dim=0).reshape(self.n_hops, self.n_entities, self.dim), dim=0)
        user_res_emb = torch.sum(torch.cat(user_res_emb, dim=0).reshape(self.n_hops, self.n_users, self.dim), dim=0)

        return entity_res_emb, user_res_emb
        # return entity_emb, user_emb


class GraphConv_KGUI1(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, dim, n_hops, n_users, n_items, n_entities,
                 n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv_KGUI1, self).__init__()

        self.convs = nn.ModuleList()
        self.dim = dim
        self.n_hops = n_hops
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_

        weight = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, dim,dim]
        bias = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias = nn.Parameter(bias)  # [n_relations - 1, dim,dim]
        # atr_node to pre_node
        weight1 = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight1 = nn.Parameter(weight1)  # [n_relations - 1, dim,dim]
        bias1 = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias1 = nn.Parameter(bias1)  # [n_relations - 1, dim,dim]
        # atr_node to pre_node
        weight2 = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight2 = nn.Parameter(weight2)  # [n_relations - 1, dim,dim]
        bias2 = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias2 = nn.Parameter(bias2)  # [n_relations - 1, dim,dim]
        # user_pre_weight
        weight_user_pre = initializer(torch.empty(n_users, n_relations))
        self.weight_user_pre = nn.Parameter(weight_user_pre)

        for i in range(n_hops):
            self.convs.append(
                Aggregator_KGUI1(n_users=n_users, n_items=n_items, n_relations=n_relations,
                                 dropuot=node_dropout_rate))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, graph, graph_user,
                user_emb, entity_emb, latent_emb,
                edge_index, edge_type,
                edge_index_user, edge_type_user,
                interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        weight_user_pre_soft = torch.softmax(self.weight_user_pre, dim=1)
        entity_res_emb = []
        user_res_emb = []
        # entity_res_emb.append(entity_emb)
        # user_res_emb.append(user_emb)
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](graph, graph_user, user_emb, entity_emb,
                                                 edge_index, edge_type, edge_index_user, edge_type_user,
                                                 interact_mat,
                                                 self.weight, self.bias,
                                                 self.weight1, self.bias1,
                                                 self.weight2, self.bias2,
                                                 weight_user_pre_soft,
                                                 node_dropout)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            # entity_res_emb = torch.add(entity_res_emb, entity_emb)
            # user_res_emb = torch.add(user_res_emb, user_emb)

            entity_res_emb.append(entity_emb)
            user_res_emb.append(user_emb)
        entity_res_emb = \
            torch.sum(torch.cat(entity_res_emb, dim=0).reshape(self.n_hops, self.n_entities, self.dim), dim=0)
        user_res_emb = torch.sum(torch.cat(user_res_emb, dim=0).reshape(self.n_hops, self.n_users, self.dim), dim=0)

        return entity_res_emb, user_res_emb
        # return entity_emb, user_emb


class GraphConv_KGUI2(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, dim, n_hops, n_users, n_items, n_entities,
                 n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv_KGUI2, self).__init__()

        self.convs = nn.ModuleList()
        self.dim = dim
        self.n_hops = n_hops
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_

        weight = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, dim,dim]
        bias = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias = nn.Parameter(bias)  # [n_relations - 1, dim,dim]
        # atr_node to pre_node
        weight1 = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight1 = nn.Parameter(weight1)  # [n_relations - 1, dim,dim]
        bias1 = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias1 = nn.Parameter(bias1)  # [n_relations - 1, dim,dim]
        # atr_node to pre_node
        weight2 = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight2 = nn.Parameter(weight2)  # [n_relations - 1, dim,dim]
        bias2 = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias2 = nn.Parameter(bias2)  # [n_relations - 1, dim,dim]
        # user_pre_weight
        weight_user_pre = initializer(torch.empty(n_users, n_relations))
        self.weight_user_pre = nn.Parameter(weight_user_pre)

        for i in range(n_hops):
            self.convs.append(
                Aggregator_KGUI2(n_users=n_users, n_items=n_items, n_relations=n_relations,
                                 dropuot=node_dropout_rate))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, graph, user_pre_r, user_pre_node,
                user_emb, entity_emb, latent_emb,
                edge_index, edge_type,
                edge_index_user, edge_type_user,
                interact_mat, mess_dropout=True, node_dropout=False):

        all_embed = torch.cat([user_emb, entity_emb], dim=0)

        """node dropout"""
        weight_user_pre_soft = torch.softmax(self.weight_user_pre, dim=1)
        entity_res_emb = []
        user_res_emb = []
        # entity_res_emb.append(entity_emb)
        # user_res_emb.append(user_emb)
        for i in range(len(self.convs)):
            entity_emb = self.convs[i](graph, entity_emb,
                                       edge_index, edge_type, edge_index_user, edge_type_user,
                                       interact_mat,
                                       self.weight, self.bias,
                                       self.weight1, self.bias1,
                                       self.weight2, self.bias2,
                                       weight_user_pre_soft,
                                       node_dropout)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)
            entity_res_emb.append(entity_emb)

        """user agg"""
        user_emb_agg = []
        for i in range(self.n_users):
            user_emb_agg.append([])
            for j in user_pre_node[i]:
                user_emb_agg[i].append(torch.mean(all_embed[j], dim=0).unsqueeze(0))
            for j, jj in enumerate(user_pre_r[i]):
                for k in jj:
                    user_emb_agg[i][j] = torch.matmul(user_emb_agg[i][j], self.weight[k])
            user_emb_agg[i] = torch.cat(user_emb_agg[i], dim=0)
            # attention
            user_att = torch.matmul(user_emb_agg[i], user_emb[i].unsqueeze(1))
            user_att = F.softmax(user_att, dim=0).permute(1, 0)
            user_emb_agg[i] = torch.matmul(user_att, user_emb_agg[i])
        user_emb_agg = torch.cat(user_emb_agg, dim=0)
        if mess_dropout:
            user_emb_agg = self.dropout(user_emb_agg)
        user_res_emb = F.normalize(user_emb_agg)

        """result emb"""
        entity_res_emb = \
            torch.sum(torch.cat(entity_res_emb, dim=0).reshape(self.n_hops, self.n_entities, self.dim), dim=0)

        return entity_res_emb, user_res_emb
        # return entity_emb, user_emb


class GraphConv_KGUI3(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, dim, n_hops, n_users, n_items, n_entities,
                 n_relations, n_pre, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv_KGUI3, self).__init__()

        self.convs = nn.ModuleList()
        self.dim = dim
        self.n_hops = n_hops
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_pre = n_pre
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_

        weight = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, dim,dim]
        bias = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias = nn.Parameter(bias)  # [n_relations - 1, dim,dim]
        # user_pre
        self.user_pre = nn.ParameterList()
        for i in range(n_hops):
            self.user_pre.append(nn.Parameter(initializer(torch.empty(n_users, n_relations, dim))))
            # self.user_pre.append(nn.Parameter(initializer(torch.empty(n_relations, dim))))
            # self.user_pre.append(nn.Parameter(initializer(torch.empty(n_users, dim))))
            # self.user_pre.append(nn.Parameter(initializer(torch.empty(n_users, int(dim * self.n_relations)))))

        # for i in range(n_hops):
        #     if i == 0:
        #         self.convs.append(
        #             Aggregator_KGUI3(n_users=n_users, n_entities=n_entities, n_items=n_items, n_relations=n_relations,
        #                              dim=dim,
        #                              dropuot=node_dropout_rate))
        #     else:
        #         self.convs.append(
        #             Aggregator_KGUI3(n_users=n_users, n_entities=n_entities, n_items=n_items, n_relations=n_relations,
        #                              dim=dim * n_relations ^ (i - 1),
        #                              dropuot=node_dropout_rate))
        for i in range(n_hops):
            self.convs.append(
                Aggregator_KGUI3(n_users=n_users, n_entities=n_entities, n_items=n_items, n_relations=n_relations,
                                 dim=dim,
                                 dropuot=node_dropout_rate))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, graph, graph_i2u, graph_u2u, user_dict,
                user_emb, entity_emb, latent_emb,
                edge_index, edge_type,
                edge_index_user, edge_type_user,
                interact_mat, mess_dropout=True, node_dropout=False):
        """node dropout"""
        entity_emb_f_all = []
        user_res_emb = []
        for i in range(len(self.convs)):
            """item agg"""
            entity, entity_emb, user = self.convs[i](graph, graph_i2u, graph_u2u, entity_emb,
                                                     edge_index, edge_type, edge_index_user, edge_type_user,
                                                     interact_mat,
                                                     self.weight,
                                                     node_dropout)
            # entity_emb = entity
            entity_emb_f_all.append(entity)
            user_res_emb.append(user)
            # entity_emb = torch.sum(entity_emb_f, dim=1)
            # """item agg 1"""
            # entity_emb_f_all.append(F.normalize(entity_emb_f, dim=2).reshape(self.n_entities, -1))
            # if mess_dropout:
            #     entity_emb_f = self.dropout(entity_emb_f)
            #     # entity_emb = self.dropout(entity_emb)
            # entity_emb_f_all.append(entity_emb_f.reshape(self.n_entities, -1))
            """item agg 2"""
            # entity_emb_f = F.normalize(entity_emb, dim=1)
            # entity_emb_f_all.append(entity_emb_f)

            """user agg 1"""
            # for j in range(self.n_users):
            #     user_atr = entity_emb_f[user_dict[j]].permute(1, 0, 2)
            #     # user_att_atr = torch.matmul(user_atr, self.user_pre[i][j].unsqueeze(2)).squeeze(2)
            #     user_att_atr = torch.matmul(user_atr,
            #                                 (user_emb[j].unsqueeze(0) * self.user_pre[i]).unsqueeze(2)). \
            #         squeeze(2)
            #     user_att_atr = F.softmax(torch.exp(user_att_atr), dim=1)
            #     user_res_emb[i].append(
            #         F.normalize(torch.bmm(user_att_atr.unsqueeze(1), user_atr).squeeze(1), dim=1).reshape(1, -1))
            # user_res_emb[i] = torch.cat(user_res_emb[i], dim=0)
            """user agg 2"""
            # for j in range(self.n_users):
            #     user_atr = torch.mean(entity_emb_f[user_dict[j]], dim=0)
            #     user_att_atr = torch.matmul(user_atr, self.user_pre[i][j].unsqueeze(1))
            #     user_att_atr = F.softmax(torch.exp(user_att_atr), dim=0)
            #     user_res_emb[i].append(F.normalize((user_atr * user_att_atr).reshape(1, -1), dim=1))
            # user_res_emb[i] = torch.cat(user_res_emb[i], dim=0)
            # if mess_dropout:
            #     user_res_emb[i] = self.dropout(user_res_emb[i])
            """user agg 3"""
            # for j in range(self.n_users):
            #     user_atr = entity_emb_f[user_dict[j]].permute(1, 0, 2)
            #     user_att_atr = torch.matmul(user_atr, self.user_pre[i][j].unsqueeze(2)).squeeze(2)
            #     user_att_atr = F.softmax(torch.exp(user_att_atr), dim=1).unsqueeze(1)
            #     user_res_emb[i].append(F.normalize(torch.matmul(user_att_atr, user_atr).squeeze(1), dim=1))
            # user_res_emb[i] = torch.cat(user_res_emb[i], dim=0).reshape(self.n_users, -1)
            """user agg 4"""
            # for j in range(self.n_users):
            #     user_atr = entity_emb_f_all[i][user_dict[j]]
            #     user_att_atr = torch.matmul(user_atr, self.user_pre[i][j].unsqueeze(1))
            #     user_att_atr = F.softmax(torch.exp(user_att_atr), dim=0).permute(1, 0)
            #     user_res_emb[i].append(F.normalize(torch.matmul(user_att_atr, user_atr), dim=1))
            # user_res_emb[i] = torch.cat(user_res_emb[i], dim=0)

        """result emb"""
        """user agg 2"""

        """agg 3"""
        # entity = \
        #     torch.sum(torch.cat(entity_emb_f_all, dim=0).reshape(self.n_hops, self.n_entities, self.dim), dim=0)
        # user = \
        #     torch.sum(torch.cat(user_res_emb, dim=0).reshape(self.n_hops, self.n_users, self.dim), dim=0)
        # user = torch.sparse.mm(interact_mat, entity)

        entity_f = torch.cat(entity_emb_f_all, dim=1)
        user_f = torch.cat(user_res_emb, dim=1)
        return entity_f, user_f
        # return entity_emb, user_emb


class GraphConv_KGIT(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, dim, n_hops, n_users, n_items, n_entities,
                 n_relations, n_pre, interact_mat,
                 ind, new_r_all, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv_KGIT, self).__init__()

        self.convs = nn.ModuleList()
        self.dim = dim
        self.n_hops = n_hops
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_pre = n_pre
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.new_r_all = new_r_all

        self.temperature = 0.2

        self.item_atr = Aggregator_KGIT_atr(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                            n_relations=n_relations,
                                            dim=dim, new_r_all=self.new_r_all,
                                            dropuot=node_dropout_rate)
        for i in range(n_hops):
            self.convs.append(
                Aggregator_KGIT(n_users=n_users, n_entities=n_entities, n_items=n_items, n_relations=n_relations,
                                dim=dim, new_r_all=self.new_r_all,
                                dropuot=node_dropout_rate))
            # self.convs.append(
            #     Aggregator_KGUI_sub(n_users=n_users, n_entities=n_entities, n_items=n_items, n_relations=n_relations,
            #                         dim=dim,
            #                         dropuot=node_dropout_rate))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        self.act = nn.LeakyReLU()

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, sub_idx1, graph, graph_i2u, graph_u2u, graph_i2u_0, graph_i2u_1, user_dict,
                user_emb, user_emb1, entity_emb,
                W_R, w_R, b_R, use_pre, new_r_all,
                interact_mat, mess_dropout=True, node_dropout=False):
        """node dropout"""
        entity_emb_f_all = []
        user_res_emb = []
        user_emb = user_emb[sub_idx1[0], :]
        user_emb1 = user_emb1[sub_idx1[0], :]
        entity, entity_atr, user = self.item_atr(sub_idx1, graph, entity_emb, W_R, w_R, b_R, new_r_all, graph_i2u_0)
        for i in range(len(self.convs)):
            """item agg"""
            entity, entity_atr, user, user_emb = \
                self.convs[i](sub_idx1, graph, graph_i2u, graph_u2u,
                              graph_i2u_0,
                              graph_i2u_1,
                              user_emb, user_emb1, entity_atr,
                              W_R, w_R, b_R, new_r_all,
                              interact_mat,
                              node_dropout)
            entity_emb_f_all.append(entity)
            user_res_emb.append(user)

        entity_f = torch.cat(entity_emb_f_all, dim=1)
        user_f = torch.cat(user_res_emb, dim=1)

        # use_pre = use_pre[sub_idx1[0]][sub_idx1[1]]
        # user_f = user_f.reshape(len(sub_idx1[1]), -1, self.dim)
        # user_f = user_f * use_pre.unsqueeze(2)
        # user_f = user_f.reshape(len(sub_idx1[1]), -1)

        # use_pre = use_pre[sub_idx1[0]][sub_idx1[1]]
        # user_f = user_f.reshape(len(sub_idx1[1]), -1, self.dim)
        # # use_pre = torch.matmul(user_f, use_pre.unsqueeze(2)).squeeze(2)
        # use_pre = F.softmax(
        #     self.act(torch.cosine_similarity(user_f, use_pre.unsqueeze(1), dim=2)),
        #     dim=1)
        # use_pre = F.softmax(use_pre, dim=1)
        # user_f = user_f * use_pre.unsqueeze(2)
        # user_f = user_f.reshape(len(sub_idx1[1]), -1)

        return entity_f, user_f
        # return entity_emb, user_emb


class GraphConv_KGUI3_1(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, dim, n_hops, n_users, n_items, n_entities,
                 n_relations, new_r_all, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv_KGUI3_1, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()
        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.temperature = 0.2

        for i in range(n_hops):
            self.convs.append(
                Aggregator_KGUI3_11(n_users=n_users, n_entities=n_entities, n_items=n_items, n_relations=n_relations,
                                    dim=dim, dropuot=node_dropout_rate))

        for i in range(n_hops):
            self.convs_ui.append(
                Aggregator_KGUI3_12(n_users=n_users, n_entities=n_entities, n_items=n_items, n_relations=n_relations,
                                    dim=dim, dropuot=node_dropout_rate))

        self.KG2UIG = nn.Linear(n_hops * len(new_r_all) * self.dim, self.dim, bias=False)
        nn.init.xavier_uniform_(self.KG2UIG.weight)
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        self.act = nn.LeakyReLU()

    def forward(self, graph, graph_UIS,
                entity_embed,
                user_embed_ui, item_embed_ui,
                W_R, new_r_all,
                mess_dropout=True,
                node_dropout=False):
        """KG agg"""
        entity_emb_all = []
        for i in range(len(self.convs)):
            entity_emb_f, entity_embed = self.convs[i](graph, entity_embed, W_R, new_r_all)
            # entity_emb_all.append(F.normalize(entity_embed, dim=1))
            entity_emb_all.append(F.normalize(entity_emb_f, dim=2).reshape(self.n_items, -1))
        entity_emb_all = torch.cat(entity_emb_all, dim=1)
        entity_emb_all = self.KG2UIG(entity_emb_all)

        """UIG agg"""
        # item_embed_ui = self.KG2UIG(entity_emb_all)
        user_f = []
        item_f = []
        for i in range(len(self.convs_ui)):
            user_embed_ui, item_embed_ui = self.convs_ui[i](graph_UIS, item_embed_ui, user_embed_ui, entity_emb_all)
            user_f.append(F.normalize(user_embed_ui, dim=1))
            item_f.append(F.normalize(item_embed_ui, dim=1))

        user_f = torch.cat(user_f, dim=1)
        item_f = torch.cat(item_f, dim=1)

        return item_f, user_f


class GraphConv_KGUI3_1_mulatr(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, dim, n_hops, n_users, n_items, n_entities,
                 n_relations, n_pre, interact_mat,
                 ind, new_r_all, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv_KGUI3_1_mulatr, self).__init__()

        self.convs = nn.ModuleList()
        self.dim = dim
        self.n_hops = n_hops
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_pre = n_pre
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.new_r_all = new_r_all

        self.temperature = 0.2

        self.entity_atr = Aggregator_KGIT_mulatr0(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                                  n_relations=n_relations,
                                                  dim=dim, new_r_all=self.new_r_all,
                                                  dropuot=node_dropout_rate)

        for i in range(n_hops):
            self.convs.append(
                Aggregator_KGUI3_11_mulatr(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                           n_relations=n_relations,
                                           dim=dim, new_r_all=self.new_r_all,
                                           dropuot=node_dropout_rate))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        self.act = nn.LeakyReLU()

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, sub_idx1, graph, graph_1, graph_i2u, graph_u2u, graph_i2u_0, graph_i2u_1, user_dict,
                user_emb, user_emb1, entity_emb,
                W_R, w_R, b_R, use_pre, new_r_all,
                interact_mat, mess_dropout=True, node_dropout=False):
        """node dropout"""
        entity_emb_f_all = []
        user_res_emb = []
        user_emb = user_emb[sub_idx1[0], :]
        user_emb1 = user_emb1[sub_idx1[0], :]

        entity0, entity_atr, user0 = self.entity_atr(sub_idx1, graph, graph_i2u_0, entity_emb, W_R, w_R, b_R, new_r_all)
        entity_emb_f_all.append(entity0)
        user_res_emb.append(user0)

        for i in range(len(self.convs)):
            """item agg"""
            entity, entity_atr, user = \
                self.convs[i](sub_idx1, graph, graph_i2u, graph_u2u,
                              graph_i2u_0,
                              graph_i2u_1,
                              user_emb, user_emb1, entity_atr,
                              W_R, w_R, b_R, new_r_all,
                              interact_mat,
                              node_dropout)
            # entity, entity_emb, user = self.convs[i](sub_idx1, user_dict, graph, graph_i2u, graph_u2u, graph_i2u_0,
            #                                          entity_emb,
            #                                          interact_mat,
            #                                          node_dropout)
            # entity_emb = entity
            entity_emb_f_all.append(entity)
            user_res_emb.append(user)

        entity_f = torch.cat(entity_emb_f_all, dim=1)
        user_f = torch.cat(user_res_emb, dim=1)

        # use_pre = F.normalize(use_pre[sub_idx1[0]], dim=1, p=1)[sub_idx1[1]]
        # user_f = user_f.reshape(len(sub_idx1[1]), -1, self.dim)
        # user_f = user_f * use_pre.unsqueeze(2)
        # user_f = user_f.reshape(len(sub_idx1[1]), -1)

        # use_pre = use_pre[sub_idx1[0]][sub_idx1[1]]
        # user_f = user_f.reshape(len(sub_idx1[1]), -1, self.dim)
        # use_pre = F.softmax(
        #     torch.cosine_similarity(user_f, use_pre.unsqueeze(1), dim=2),
        #     dim=1)
        # self.out_use_pre = use_pre
        # user_f = user_f * use_pre.unsqueeze(2)
        # user_f = user_f.reshape(len(sub_idx1[1]), -1)

        return entity_f, user_f
        # return entity_emb, user_emb


class GraphConv_UMIKGAN(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like):
        super(GraphConv_UMIKGAN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like

        for i in range(n_hops):
            self.convs.append(
                Aggregator_UMIKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim))

        for i in range(n_hops):
            self.convs_ui.append(
                Aggregator_UMIKGAN_user(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim))

    def forward(self, graph, graph_UIS, entity_embed, W_R, common_like, common_dislike, user_like):
        """KG agg"""
        entity_embed0 = entity_embed

        # entity_emb_all = []
        entity_emb_all = entity_embed[:self.n_items, :]

        for i in range(self.n_hops):
            entity_embed0 = self.convs[i](graph, entity_embed0, W_R)

            entity_emb_all = torch.add(entity_emb_all, F.normalize(entity_embed0[:self.n_items, :], dim=1))
            # entity_emb_all = torch.add(entity_emb_all, entity_embed0[:self.n_items, :])
        # entity_emb_all = torch.cat(entity_emb_all, dim=1)

        """user agg"""
        # att = torch.exp(torch.matmul(user_like, common_like.permute(1, 0)))
        att = torch.exp(torch.cosine_similarity(user_like.unsqueeze(1), common_like.unsqueeze(0), dim=2))
        att_sum = torch.sum(att, dim=1)
        att = att / att_sum.unsqueeze(1)

        common_user_embed = []
        for i in range(self.num_like):
            entity_embed0 = entity_embed

            user_f = []
            for j in range(self.n_hops):
                entity_embed0, user_embed = self.convs_ui[j](graph, graph_UIS, entity_embed0, W_R, common_like[i])
                # user_f.append(user_f)

                user_f.append(F.normalize(user_embed, dim=1))
                # user_f.append(user_embed)

            user_f = torch.cat(user_f, dim=1).reshape(self.n_users, self.n_hops, -1)
            user_f = torch.sum(user_f, dim=1)

            user_f = user_f + common_like[i].unsqueeze(0)

            common_user_embed.append(user_f)

        common_user_embed = torch.cat(common_user_embed, dim=1). \
            reshape(self.n_users, self.num_like, -1)

        """all user"""
        user_emb_all = torch.matmul(att.unsqueeze(1), common_user_embed).squeeze(1)

        # """user dislike agg"""
        # att = torch.exp(torch.cosine_similarity(user_like.unsqueeze(1), common_dislike.unsqueeze(0), dim=2))
        # att_sum = torch.sum(att, dim=1)
        # att = att / att_sum.unsqueeze(1)
        #
        # common_user_embed = []
        # for i in range(self.num_like):
        #     entity_embed0 = entity_embed
        #
        #     user_f = []
        #     for j in range(self.n_hops):
        #         entity_embed0, user_embed = self.convs_ui[j](graph, graph_UIS, entity_embed0, W_R, common_dislike[i])
        #         # user_f.append(user_f)
        #
        #         user_f.append(F.normalize(user_embed, dim=1))
        #
        #     user_f = torch.cat(user_f, dim=1).reshape(self.n_users, self.n_hops, -1)
        #     user_f = torch.sum(user_f, dim=1)
        #
        #     user_f = user_f + common_dislike[i].unsqueeze(0)
        #
        #     common_user_embed.append(user_f)
        #
        # common_user_embed = torch.cat(common_user_embed, dim=1). \
        #     reshape(self.n_users, self.num_like, -1)
        #
        # """all user"""
        # user_emb_all_dislike = torch.matmul(att.unsqueeze(1), common_user_embed).squeeze(1)

        return entity_emb_all, user_emb_all


class GraphConv_UMIKGAN0(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like):
        super(GraphConv_UMIKGAN0, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like

        for i in range(n_hops):
            self.convs.append(
                Aggregator_UMIKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim))

        for i in range(n_hops):
            self.convs_ui.append(
                Aggregator_UMIKGAN_user0(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                         n_relations=n_relations,
                                         dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

        self.item2user = Aggregator_I2U(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim)

    def forward(self, graph, graph_UIS, all_embed, W_R, common_like):

        user_embed = all_embed[:self.n_users, :]
        entity_embed = all_embed[self.n_users:, :]

        """user agg"""
        common_user_embed = []
        common_entity_embed = []

        for i in range(self.num_like):
            entity_embed0 = entity_embed
            user_embed0 = user_embed

            att_r = torch.exp(torch.sum(W_R * common_like[i].unsqueeze(0), dim=1))
            att_r_sum = torch.sum(att_r)
            att_r = att_r / att_r_sum

            user_f = []
            entity_f = []
            for j in range(self.n_hops):
                entity_embed0, user_embed0 = \
                    self.convs_ui[j](graph, graph_UIS, entity_embed0, W_R, common_like[i], att_r, user_embed0)

                if False:
                    entity_embed0 = self.dropout(entity_embed0)
                    user_embed0 = self.dropout(user_embed0)

                entity_embed0 = F.normalize(entity_embed0, dim=1)
                user_embed0 = F.normalize(user_embed0, dim=1)

                user_f.append(user_embed0)
                entity_f.append(entity_embed0)

            user_f = torch.cat(user_f, dim=1).reshape(self.n_users, self.n_hops, -1)
            user_f = torch.sum(user_f, dim=1)

            entity_f = torch.cat(entity_f, dim=1).reshape(self.n_entities, self.n_hops, -1)
            entity_f = torch.sum(entity_f, dim=1)

            common_user_embed.append(user_f)
            common_entity_embed.append(entity_f)

        common_user_embed = torch.cat(common_user_embed, dim=1). \
            reshape(self.n_users, self.num_like, -1)

        common_entity_embed = torch.cat(common_entity_embed, dim=1). \
            reshape(self.n_entities, self.num_like, -1)

        """OUTPUT item"""
        entity_emb_all = torch.mean(common_entity_embed, dim=1)
        entity_emb_all = entity_emb_all + entity_embed

        """OUTPUT user"""
        att = torch.exp(torch.matmul(user_embed, common_like.permute(1, 0)))
        att_sum = torch.sum(att, dim=1)
        att = att / att_sum.unsqueeze(1)

        user_emb_all = torch.matmul(att.unsqueeze(1), common_user_embed).squeeze(1)

        user_emb_all = user_emb_all + user_embed

        """all user"""
        # att = torch.exp(torch.matmul(user_embed, common_like.permute(1, 0)))
        # att_sum = torch.sum(att, dim=1)
        # att = att / att_sum.unsqueeze(1)
        #
        # common_user_embed = torch.matmul(att.unsqueeze(1), common_user_embed).squeeze(1)
        # user_emb_all = user_emb_all + common_user_embed

        return entity_emb_all, user_emb_all


class GraphConv_MUIGAN(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like):
        super(GraphConv_MUIGAN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like

        for i in range(n_hops):
            self.convs.append(
                Aggregator_UMIKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim))

        for i in range(n_hops):
            self.convs_ui.append(
                Aggregator_MUIGAN(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                  n_relations=n_relations,
                                  dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

        self.item2user = Aggregator_I2U(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim)

    def forward(self, graph, graph_UIS, all_embed, W_R, common_like):
        user_embed = all_embed[:self.n_users, :]
        entity_embed = all_embed[self.n_users:, :]

        att_r = torch.mm(common_like, W_R.permute(1, 0))
        att_r = nn.Softmax(dim=1)(att_r)

        user_emb_all = []
        entity_emb_all = []

        user_emb_all.append(user_embed)
        entity_emb_all.append(entity_embed)

        for j in range(self.n_hops):
            att = torch.mm(user_embed, common_like.permute(1, 0))
            att = nn.Softmax(dim=1)(att)

            user_f = []
            entity_f = []
            for i in range(self.num_like):

                entity_embed0, user_embed0 = \
                    self.convs_ui[j](graph, graph_UIS, entity_embed, W_R,
                                     common_like[i], att_r[i], user_embed)

                if False:
                    entity_embed0 = self.dropout(entity_embed0)
                    user_embed0 = self.dropout(user_embed0)

                user_f.append(user_embed0)
                entity_f.append(entity_embed0)

            user_f = torch.cat(user_f, dim=1).reshape(self.n_users, self.num_like, self.dim)
            entity_f = torch.cat(entity_f, dim=1).reshape(self.n_entities, self.num_like, self.dim)

            entity_embed = torch.mean(entity_f, dim=1)

            user_embed = torch.matmul(att.unsqueeze(1), user_f).squeeze(1)

            entity_embed = F.normalize(entity_embed, dim=1)
            user_embed = F.normalize(user_embed, dim=1)

            entity_emb_all.append(entity_embed)
            user_emb_all.append(user_embed)

        """OUTPUT"""
        entity_emb_all = torch.cat(entity_emb_all, dim=1). \
            reshape(self.n_entities, (1 + self.n_hops), -1)
        entity_emb_all = torch.sum(entity_emb_all, dim=1)

        user_emb_all = torch.cat(user_emb_all, dim=1). \
            reshape(self.n_users, (1 + self.n_hops), -1)
        user_emb_all = torch.sum(user_emb_all, dim=1)

        return entity_emb_all, user_emb_all


class GraphConv_MAKG(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like):
        super(GraphConv_MAKG, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like

        for i in range(n_hops):
            self.convs.append(
                Aggregator_UMIKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim))

        for i in range(n_hops):
            self.convs_ui.append(
                Aggregator_MAKG(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                n_relations=n_relations,
                                dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

        self.item2user = Aggregator_I2U(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim)

        self.kmeans = []
        for i in range(n_hops):
            self.kmeans.append(KMeans(n_clusters=self.num_like))
            self.kmeans[i].num = 1

        self.kmeans_r = KMeans(n_clusters=self.num_like + 1)
        self.kmeans_r.num = 1

    def forward(self, graph, graph_UIS, graph_r, all_embed, W_R, common_like, user_like):
        user_embed = all_embed[:self.n_users, :]
        entity_embed = all_embed[self.n_users:, :]

        # W_R = W_R_user_like[:self.n_relations]
        # user_like = W_R_user_like[self.n_relations:]

        """relation attention in common like"""
        # atr_att = torch.mm(W_R, user_like.permute(1, 0))
        # atr_att = nn.Softmax(dim=1)(atr_att).permute(1, 0)
        #
        # atr_att = nn.Softmax(dim=0)(common_like)

        # att_r = nn.Softmax(dim=1)(10 * common_like)
        # user_like = torch.mm(att_r, W_R)

        # att_r = torch.mm(user_like, W_R.permute(1, 0))
        # att_r = nn.Softmax(dim=1)(att_r)
        # user_like = torch.mm(att_r, W_R)

        # index = np.arange(self.n_relations)
        # np.random.shuffle(index)
        # num_one = self.n_relations // self.num_like + 1
        # user_like = []
        # for i in range(self.num_like):
        #     user_like.append(torch.mean(W_R[index[i * num_one:(i + 1) * num_one]], dim=0).unsqueeze(0))
        # user_like = torch.cat(user_like, dim=0)

        "kmeans"
        if self.training:
            if self.kmeans_r.num == 1:
                self.kmeans_r = self.kmeans_r.fit(W_R.detach().cpu().numpy())
                self.kmeans_r.num = 2
            else:
                self.kmeans_r = KMeans(n_clusters=self.num_like + 1, init=self.kmeans_r.cluster_centers_)
                self.kmeans_r.num = 2
                self.kmeans_r = self.kmeans_r.fit(W_R.detach().cpu().numpy())
        user_like = torch.tensor(self.kmeans_r.cluster_centers_, device=user_embed.device)

        R_att = torch.cosine_similarity(user_like.unsqueeze(1), W_R.unsqueeze(0), dim=2)
        exp_deg = nn.Softmax(dim=1)(2 * R_att)

        # exp_deg = []
        # self.W_R = W_R
        # for i in range(self.num_like):
        #     self.atr_att = user_like[i].unsqueeze(0)
        #     graph_r = graph_r.local_var()
        #     graph_r.apply_edges(self.cal_deg)
        #     graph_r.update_all(
        #         dgl.function.copy_e('atr', 'temp'),
        #         dgl.function.sum('temp', 'exp_deg'))
        #     exp_deg.append(graph_r.ndata['exp_deg'][:-1])

        user_emb_all = []
        entity_emb_all = []

        user_emb_all.append(user_embed)
        entity_emb_all.append(entity_embed)

        entity_f = []
        for i in range(self.num_like):
            entity_f.append(entity_embed)

        user_embed1 = user_embed
        for j in range(self.n_hops):

            "kmeans"
            if self.training:
                if self.kmeans[j].num == 1:
                    self.kmeans[j] = self.kmeans[j].fit(user_embed1.detach().cpu().numpy())
                    self.kmeans[j].num = 2
                else:
                    self.kmeans[j] = KMeans(n_clusters=self.num_like, init=self.kmeans[j].cluster_centers_)
                    self.kmeans[j].num = 2
                    self.kmeans[j] = self.kmeans[j].fit(user_embed1.detach().cpu().numpy())
            common_like = torch.tensor(self.kmeans[j].cluster_centers_, device=user_embed.device)

            entity_f1 = []
            for i in range(self.num_like):

                entity_embed0 = self.convs_ui[j](graph, entity_f[i], W_R,
                                                 user_like[i], exp_deg[i], common_like[i])
                if True:
                    entity_embed0 = self.dropout(entity_embed0)
                entity_f1.append(entity_embed0)
                # entity_f1.append(F.normalize(entity_embed0, dim=1))

            entity_f = entity_f1

            entity_f1 = torch.cat(entity_f1, dim=1).reshape(self.n_entities, self.num_like, self.dim)
            # entity_f1 = F.normalize(entity_f1, dim=2)

            """agg user from diffirent atr"""
            user_att = torch.cosine_similarity(user_embed1.unsqueeze(1), common_like.unsqueeze(0), dim=2)
            # user_att = torch.cosine_similarity(user_embed1.unsqueeze(1), common_like[:self.num_like].unsqueeze(0),
            #                                    dim=2)

            user_att = nn.Softmax(dim=1)(2 * user_att)

            user_embed = self.item2user(graph_UIS, entity_f1, user_att, entity_embed)
            # user_embed0 = torch.matmul(user_att, common_like)
            # user_embed = user_embed + user_embed0

            """agg item by mean pooling"""
            entity_embed = torch.mean(entity_f1, dim=1)

            entity_embed = F.normalize(entity_embed, dim=1)
            user_embed = F.normalize(user_embed, dim=1)

            entity_emb_all.append(entity_embed)
            user_emb_all.append(user_embed)

            user_embed1 = user_embed1 + user_emb_all[-1]

        """OUTPUT"""
        entity_emb_all = torch.cat(entity_emb_all, dim=1). \
            reshape(self.n_entities, (1 + self.n_hops), -1)
        entity_emb_all = torch.sum(entity_emb_all, dim=1)

        user_emb_all = torch.cat(user_emb_all, dim=1). \
            reshape(self.n_users, (1 + self.n_hops), -1)
        user_emb_all = torch.sum(user_emb_all, dim=1)

        # entity_emb_all = entity_emb_all[-1]
        # user_emb_all = user_emb_all[-1]

        return entity_emb_all, user_emb_all

    def cal_deg(self, edges):
        atr = torch.exp(torch.sum(torch.mul(self.W_R[edges.data['type']], self.atr_att), dim=1))
        # atr = torch.exp(10 * torch.cosine_similarity(self.W_R[edges.data['type']], self.atr_att, dim=1))
        return {'atr': atr}


class GraphConv_MAKG1(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like):
        super(GraphConv_MAKG1, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like

        for i in range(n_hops):
            self.convs.append(
                Aggregator_UMIKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim))

        for i in range(n_hops):
            self.convs_ui.append(
                Aggregator_MAKG(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                n_relations=n_relations,
                                dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

        self.item2user = Aggregator_I2U(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim)

        self.kmeans = KMeans(n_clusters=self.num_like)
        self.kmeans.num = 1

        self.kmeans_r = KMeans(n_clusters=self.num_like)
        self.kmeans_r.num = 1

    def forward(self, graph, graph_UIS, graph_r, all_embed, W_R, common_like, user_like):
        user_embed = all_embed[:self.n_users, :]
        entity_embed = all_embed[self.n_users:, :]

        "kmeans"
        if self.training:
            if self.kmeans_r.num == 1:
                self.kmeans_r = self.kmeans_r.fit(W_R.detach().cpu().numpy())
                self.kmeans_r.num = 2
            else:
                self.kmeans_r = KMeans(n_clusters=self.num_like, init=self.kmeans_r.cluster_centers_)
                self.kmeans_r.num = 2
                self.kmeans_r = self.kmeans_r.fit(W_R.detach().cpu().numpy())
        user_like = torch.tensor(self.kmeans_r.cluster_centers_, device=user_embed.device)

        R_att = torch.cosine_similarity(user_like.unsqueeze(1), W_R.unsqueeze(0), dim=2)
        exp_deg = nn.Softmax(dim=1)(2 * R_att)

        "kmeans"
        if self.training:
            if self.kmeans.num == 1:
                self.kmeans = self.kmeans.fit(user_embed.detach().cpu().numpy())
                self.kmeans.num = 2
            else:
                self.kmeans = KMeans(n_clusters=self.num_like, init=self.kmeans.cluster_centers_)
                self.kmeans.num = 2
                self.kmeans = self.kmeans.fit(user_embed.detach().cpu().numpy())
        common_like = torch.tensor(self.kmeans.cluster_centers_, device=user_embed.device)

        user_emb_all = []
        entity_emb_all = []

        user_emb_all.append(user_embed)
        entity_emb_all.append(entity_embed)

        entity_f = []
        for i in range(self.num_like):
            entity_f.append(entity_embed)

        user_embed1 = user_embed
        for j in range(self.n_hops):

            entity_f1 = []
            for i in range(self.num_like):

                entity_embed0 = self.convs_ui[j](graph, entity_f[i], W_R,
                                                 user_like[i], exp_deg[i], common_like[i])
                if True:
                    entity_embed0 = self.dropout(entity_embed0)
                entity_f1.append(entity_embed0)

            entity_f = entity_f1

            entity_f1 = torch.cat(entity_f1, dim=1).reshape(self.n_entities, self.num_like, self.dim)

            """agg user from diffirent atr"""
            user_att = torch.cosine_similarity(user_embed1.unsqueeze(1), common_like.unsqueeze(0), dim=2)

            user_att = nn.Softmax(dim=1)(2 * user_att)

            user_embed = self.item2user(graph_UIS, entity_f1, user_att, entity_embed)

            """agg item by mean pooling"""
            entity_embed = torch.mean(entity_f1, dim=1)

            entity_embed = F.normalize(entity_embed, dim=1)
            user_embed = F.normalize(user_embed, dim=1)

            entity_emb_all.append(entity_embed)
            user_emb_all.append(user_embed)

            user_embed1 = user_embed1 + user_emb_all[-1]

        """OUTPUT"""
        entity_emb_all = torch.cat(entity_emb_all, dim=1). \
            reshape(self.n_entities, (1 + self.n_hops), -1)
        entity_emb_all = torch.sum(entity_emb_all, dim=1)

        user_emb_all = torch.cat(user_emb_all, dim=1). \
            reshape(self.n_users, (1 + self.n_hops), -1)
        user_emb_all = torch.sum(user_emb_all, dim=1)

        return entity_emb_all, user_emb_all

    def cal_deg(self, edges):
        atr = torch.exp(torch.sum(torch.mul(self.W_R[edges.data['type']], self.atr_att), dim=1))
        # atr = torch.exp(10 * torch.cosine_similarity(self.W_R[edges.data['type']], self.atr_att, dim=1))
        return {'atr': atr}


class GraphConv_UIFGAN(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like, n_epoch):
        super(GraphConv_UIFGAN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()
        self.convs_ui0 = nn.ModuleList()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like
        self.n_epoch = n_epoch

        for i in range(n_hops):
            self.convs.append(
                Aggregator_UMIKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim))

        for j in range(n_hops):
            self.convs_ui0.append(
                Aggregator_MAKG1(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                 n_relations=n_relations,
                                 dim=dim))

        for i in range(num_like):
            self.convs_ui.append(nn.ModuleList())
            for j in range(n_hops):
                self.convs_ui[i].append(
                    Aggregator_MAKG(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                    n_relations=n_relations,
                                    dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

        self.item2user = Aggregator_I2U(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim)
        self.item2user0 = Aggregator_I2U0(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                          n_relations=n_relations,
                                          dim=dim)

        self.kmeans = KMeans(n_clusters=self.num_like)
        self.kmeans.num = 1

        self.kmeans_r = KMeans(n_clusters=self.num_like)
        self.kmeans_r.num = 1

        self.pos = nn.LeakyReLU()

    def forward(self, graph, graph_UIS, all_embed):
        user_embed = all_embed[:self.n_users, :]
        entity_embed = all_embed[self.n_users:self.n_users + self.n_entities, :]
        W_R = all_embed[self.n_users + self.n_entities:, :]

        """agg"""
        entity_f0 = []
        # entity_f0.append(entity_embed)
        entity_f_agg = entity_embed
        for j in range(self.n_hops):
            entity_f_agg = self.convs_ui0[j](graph, entity_f_agg, W_R)
            # entity_f_agg = F.normalize(entity_f_agg, dim=1)
            if True:
                entity_f_agg = self.dropout(entity_f_agg)
            entity_f0.append(entity_f_agg)
        entity_f0 = torch.cat(entity_f0, dim=1). \
            reshape(self.n_entities, self.n_hops, -1)
        entity_f0 = torch.sum(entity_f0, dim=1)
        # entity_f0 = F.normalize(entity_f0, dim=1)
        entity_f0 = entity_f0 + entity_embed
        entity_f0 = F.normalize(entity_f0, dim=1)
        # user_emb0 = self.item2user0(graph_UIS, entity_f0, user_embed)
        """agg"""

        # if self.training:
        #     if self.kmeans.num == 1:
        #         self.kmeans = KMeans(n_clusters=self.num_like)
        #         self.kmeans.num = 1
        #         # self.kmeans = self.kmeans.fit(user_embed.detach().cpu().numpy())
        #         self.kmeans = self.kmeans.fit(user_embed.detach().cpu().numpy() + user_emb0.detach().cpu().numpy())
        #     self.kmeans.num += 1
        #     if self.kmeans.num > self.n_epoch:
        #         # if self.kmeans.num > 100:
        #         self.kmeans.num = 1
        # self.common_like = torch.tensor(self.kmeans.cluster_centers_, device=user_embed.device)
        # user_label = torch.LongTensor(self.kmeans.labels_).to(user_embed.device)

        # if self.training:
        #     if self.kmeans_r.num == 1:
        #         self.kmeans_r = KMeans(n_clusters=self.num_like)
        #         self.kmeans_r.num = 1
        #         self.kmeans_r = self.kmeans.fit(W_R.detach().cpu().numpy())
        #     self.kmeans_r.num += 1
        #     if self.kmeans_r.num > self.n_epoch:
        #         if self.kmeans.num > 100:
                # self.kmeans_r.num = 1

        # if self.training:
        #     self.kmeans_r = self.kmeans_r.fit(W_R.detach().cpu().numpy())
        # self.common_like = []
        # # exp_deg = []
        # r_label = self.kmeans_r.labels_
        # for i in range(self.num_like):
        #     idx = np.where(r_label == i)[0]
        #     # att_rr = torch.zeros(size=(1, self.n_relations))
        #     # att_rr[0, idx] = 1 / len(idx)
        #     # exp_deg.append(att_rr)
        #     self.common_like.append(torch.mean(W_R[idx], dim=0).unsqueeze(0))
        # # exp_deg = torch.cat(exp_deg, dim=0).to(user_embed.device)
        # self.common_like = torch.cat(self.common_like, dim=0)
        #
        # R_att = torch.cosine_similarity(self.common_like.unsqueeze(1), W_R.unsqueeze(0), dim=2)
        # # R_att = torch.mm(self.common_like, W_R.permute(1, 0))
        # exp_deg = nn.Softmax(dim=1)(R_att) * self.n_relations
        # self.out = exp_deg

        """test"""
        R_att = torch.cosine_similarity(user_embed.unsqueeze(1), W_R.unsqueeze(0), dim=2)
        # R_att = torch.mm(user_embed, W_R.permute(1, 0))
        R_att = nn.Softmax(dim=1)(R_att)
        self.kmeans_r = self.kmeans_r.fit(R_att.detach().cpu().numpy())
        exp_deg = torch.tensor(self.kmeans_r.cluster_centers_, device=user_embed.device) * self.n_relations
        self.out = exp_deg
        """test"""

        # """test hot relation"""
        # R_att = torch.mm(user_embed, W_R.permute(1, 0))
        # # R_att = torch.cosine_similarity(user_embed.unsqueeze(1), W_R.unsqueeze(0), dim=2)
        # # R_att = torch.cosine_similarity(user_embed.unsqueeze(1), W_R.unsqueeze(0), dim=2)[:, :int(self.n_relations / 2)]
        # R_att = nn.Softmax(dim=1)(R_att)
        # R_att_hot = torch.mean(R_att, dim=0)
        # idx = torch.argsort(R_att_hot)
        # self.out = idx[:10]
        # exp_deg = []
        # for i in range(self.num_like):
        #     att_rr = 0.1 * torch.ones(size=(1, self.n_relations))
        #     att_rr[0, idx[i]] = 1
        #     exp_deg.append(att_rr)
        # # exp_deg.append(torch.ones(size=(1, self.n_relations)))
        # exp_deg = torch.cat(exp_deg, dim=0).to(user_embed.device)
        # """test"""

        entity_emb_all1 = []
        # entity_emb_all1.append(entity_f0)

        for i in range(self.num_like):

            entity_f1 = []
            # entity_f1.append(entity_embed)

            entity_f_agg = entity_embed

            for j in range(self.n_hops):
                entity_f_agg = self.convs_ui[i][j](graph, entity_f_agg, W_R, exp_deg[i])
                # entity_f_agg = F.normalize(entity_f_agg, dim=1)

                if True:
                    entity_f_agg = self.dropout(entity_f_agg)

                entity_f1.append(entity_f_agg)

            entity_f1 = torch.cat(entity_f1, dim=1). \
                reshape(self.n_entities, self.n_hops, -1)
            entity_f1 = torch.sum(entity_f1, dim=1)
            # entity_f1 = F.normalize(entity_f1, dim=1)
            entity_f1 = entity_f1 + entity_embed
            entity_f1 = F.normalize(entity_f1, dim=1)

            entity_emb_all1.append(entity_f1)

        entity_emb_all1 = torch.cat(entity_emb_all1, dim=1).reshape(self.n_entities, self.num_like, self.dim)
        # entity_emb_all1 = torch.cat(entity_emb_all1, dim=1).reshape(self.n_entities, self.num_like + 1, self.dim)

        """agg user from diffirent atr"""
        # user_embed_agg = self.item2user(graph_UIS, entity_emb_all1, entity_embed,
        #                                 self.user_embed_f)
        user_embed_agg = self.item2user(graph_UIS, entity_emb_all1, entity_embed,
                                        user_embed, 1, 1)

        """OUTPUT"""
        # user_emb_f = user_embed + user_embed_agg
        # entity_emb_f = torch.mean(entity_emb_all1, dim=1)

        user_emb_f = user_embed_agg + user_embed
        entity_emb_f = entity_f0

        return entity_emb_f, user_emb_f

    def cal_deg(self, edges):
        atr = torch.exp(torch.sum(torch.mul(self.W_R[edges.data['type']], self.atr_att), dim=1))
        # atr = torch.exp(10 * torch.cosine_similarity(self.W_R[edges.data['type']], self.atr_att, dim=1))
        return {'atr': atr}


class GraphConv_UIFGANcp(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like):
        super(GraphConv_UIFGANcp, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like

        for i in range(n_hops):
            self.convs.append(
                Aggregator_UMIKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim))

        for i in range(num_like):
            self.convs_ui.append(nn.ModuleList())
            for j in range(n_hops):
                self.convs_ui[i].append(
                    Aggregator_MAKG(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                    n_relations=n_relations,
                                    dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

        self.item2user = Aggregator_I2U(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim)
        self.item2user0 = Aggregator_I2U0(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                          n_relations=n_relations,
                                          dim=dim)

        self.kmeans = KMeans(n_clusters=self.num_like)
        self.kmeans.num = 1

        self.kmeans_r = KMeans(n_clusters=self.num_like)
        self.kmeans_r.num = 1

        self.pos = nn.LeakyReLU()

    def forward(self, graph, graph_UIS, graph_r, all_embed, W_R, common_like, user_like):
        user_embed = all_embed[:self.n_users, :]
        entity_embed = all_embed[self.n_users:self.n_users + self.n_entities, :]
        W_R = all_embed[self.n_users + self.n_entities:, :]
        #
        # "kmeans_r"
        # if self.training:
        #     if self.kmeans_r.num == 1:
        #         self.kmeans_r = self.kmeans_r.fit(W_R.detach().cpu().numpy())
        #         self.kmeans_r.num = 2
        #     else:
        #         self.kmeans_r = KMeans(n_clusters=self.num_like, init=self.kmeans_r.cluster_centers_)
        #         self.kmeans_r.num = 2
        #         self.kmeans_r = self.kmeans_r.fit(W_R.detach().cpu().numpy())
        # user_like = torch.tensor(self.kmeans_r.cluster_centers_, device=user_embed.device)
        #
        # R_att = torch.cosine_similarity(user_like.unsqueeze(1), W_R.unsqueeze(0), dim=2)
        # exp_deg = nn.Softmax(dim=1)(2 * R_att)

        # user_embed0 = self.item2user0(graph_UIS, entity_embed, user_embed)

        "kmeans_node"
        if not hasattr(self, 'common_like'):
            if self.training:
                if self.kmeans.num == 1:
                    self.kmeans = self.kmeans.fit(user_embed.detach().cpu().numpy())
                    self.kmeans.num = 2
                else:
                    self.kmeans = KMeans(n_clusters=self.num_like, init=self.kmeans.cluster_centers_)
                    self.kmeans.num = 2
                    self.kmeans = self.kmeans.fit(user_embed.detach().cpu().numpy())
            self.common_like = torch.tensor(self.kmeans.cluster_centers_, device=user_embed.device)

        # from Bio.Cluster import kcluster
        # from Bio.Cluster import clustercentroids
        # clusterid, error, nfound = kcluster(user_embed.detach().cpu().numpy(), nclusters=3, method='m',dist='u')
        # a=np.where(clusterid==0)[0]
        # a1=torch.mean(user_embed[a],dim=0)
        # torch.cosine_similarity(user_embed[a[0]],user_embed[a[500]],dim=0)
        # torch.cosine_similarity(user_embed[0],user_embed[2],dim=0)
        # torch.cosine_similarity(common_like[2],common_like[1],dim=0)

        from sklearn.cluster import spectral_clustering

        # torch.sum(user_embed[0] * user_embed[0])
        # torch.sum(user_embed[1] * user_embed[1])
        # torch.sum(user_embed[1] * user_embed[0])
        # torch.sum(common_like[2] * common_like[2])
        # torch.sum(common_like[2] * entity_embed[2])

        R_att = torch.cosine_similarity(self.common_like.unsqueeze(1), W_R.unsqueeze(0), dim=2)
        exp_deg = nn.Softmax(dim=1)(R_att)
        # exp_deg = nn.Softmax(dim=1)(3 * R_att)
        # exp_deg = self.pos(R_att) + 0.01

        user_emb_all = []
        entity_emb_all = []
        entity_emb_all1 = []

        user_emb_all.append(user_embed)

        for i in range(self.num_like):
            entity_f = []
            entity_f.append(entity_embed)

            entity_f1 = []
            entity_f1.append(entity_embed)

            entity_f_agg = entity_embed

            for j in range(self.n_hops):
                entity_f_agg = self.convs_ui[i][j](graph, entity_f_agg, W_R,
                                                   user_like[i], exp_deg[i], self.common_like[i])
                entity_f_agg = F.normalize(entity_f_agg, dim=1)

                if True:
                    entity_f_agg = self.dropout(entity_f_agg)

                entity_f.append(F.normalize(entity_f_agg))
                entity_f1.append(entity_f_agg)

            entity_f = torch.cat(entity_f, dim=1). \
                reshape(self.n_entities, (1 + self.n_hops), -1)
            entity_f = torch.sum(entity_f, dim=1)

            entity_f1 = torch.cat(entity_f1, dim=1). \
                reshape(self.n_entities, (1 + self.n_hops), -1)
            entity_f1 = torch.sum(entity_f1, dim=1)

            entity_emb_all.append(entity_f)
            entity_emb_all1.append(entity_f1)

        entity_emb_all = torch.cat(entity_emb_all, dim=1).reshape(self.n_entities, self.num_like, self.dim)
        entity_emb_all1 = torch.cat(entity_emb_all1, dim=1).reshape(self.n_entities, self.num_like, self.dim)

        """agg user from diffirent atr"""
        user_att = torch.cosine_similarity(user_embed.unsqueeze(1), self.common_like.unsqueeze(0), dim=2)

        user_att = nn.Softmax(dim=1)(user_att)

        user_embed_agg = self.item2user(graph_UIS, entity_emb_all1, user_att, entity_embed,
                                        user_embed)

        """OUTPUT"""
        # user_emb_f = user_embed + F.normalize(user_embed_agg, dim=1)
        # entity_emb_f = torch.mean(entity_emb_all, dim=1)

        user_emb_f = user_embed + user_embed_agg
        entity_emb_f = torch.sum(entity_emb_all1, dim=1)

        "kmeans_node"
        if self.training:
            if self.kmeans.num == 1:
                self.kmeans = self.kmeans.fit(user_embed.detach().cpu().numpy())
                self.kmeans.num = 2
            else:
                self.kmeans = KMeans(n_clusters=self.num_like, init=self.kmeans.cluster_centers_)
                self.kmeans.num = 2
                self.kmeans = self.kmeans.fit(user_embed.detach().cpu().numpy())
        self.common_like = torch.tensor(self.kmeans.cluster_centers_, device=user_embed.device)

        return entity_emb_f, user_emb_f

    def cal_deg(self, edges):
        atr = torch.exp(torch.sum(torch.mul(self.W_R[edges.data['type']], self.atr_att), dim=1))
        # atr = torch.exp(10 * torch.cosine_similarity(self.W_R[edges.data['type']], self.atr_att, dim=1))
        return {'atr': atr}


class GraphConv_UMIKGAN0_kmean(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like):
        super(GraphConv_UMIKGAN0_kmean, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like

        for i in range(n_hops):
            self.convs.append(
                Aggregator_UMIKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                        n_relations=n_relations,
                                        dim=dim))

        for i in range(n_hops):
            self.convs_ui.append(
                Aggregator_UMIKGAN_user0(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                         n_relations=n_relations,
                                         dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

    def forward(self, graph, graph_UIS, entity_embed, W_R, user_like):
        """KG agg"""
        entity_embed0 = entity_embed

        # entity_emb_all = []
        # entity_emb_all.append(entity_embed[:self.n_items, :])

        entity_emb_all = entity_embed[:self.n_items, :]
        # entity_emb_all = F.normalize(entity_embed[:self.n_items, :], dim=1)

        for i in range(self.n_hops):
            entity_embed0 = self.convs[i](graph, entity_embed0, W_R)

            entity_embed0 = F.normalize(entity_embed0, dim=1)

            # entity_emb_all.append(entity_embed0[:self.n_items, :])
            entity_emb_all = torch.add(entity_emb_all, entity_embed0[:self.n_items, :])

        """user agg"""
        kmeans = KMeans(n_clusters=self.num_like).fit(user_like.detach().cpu().numpy())
        common_like = torch.tensor(kmeans.cluster_centers_, device=entity_embed.device)
        user_cluster = kmeans.labels_.tolist()

        common_user_embed = []
        for i in range(self.num_like):
            entity_embed0 = entity_embed

            att_r = torch.exp(torch.cosine_similarity(W_R, common_like[i].unsqueeze(0), dim=1))
            att_r_sum = torch.sum(att_r)
            att_r = att_r / att_r_sum

            user_f = []
            for j in range(self.n_hops):
                entity_embed0, user_embed = \
                    self.convs_ui[j](graph, graph_UIS, entity_embed0, W_R, common_like[i], att_r)

                entity_embed0 = F.normalize(entity_embed0, dim=1)

                user_f.append(F.normalize(user_embed, dim=1))

            user_f = torch.cat(user_f, dim=1).reshape(self.n_users, self.n_hops, -1)
            user_f = torch.sum(user_f, dim=1)

            common_user_embed.append(user_f)

        common_user_embed = torch.cat(common_user_embed, dim=1). \
            reshape(self.n_users, self.num_like, -1)

        """all user"""
        user_emb_all = common_user_embed[list(range(self.n_users)), user_cluster, :]
        user_emb_all = user_like + user_emb_all

        return entity_emb_all, user_emb_all


class GraphConv_UPKGAN(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like):
        super(GraphConv_UPKGAN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like

        for i in range(n_hops):
            self.convs.append(
                Aggregator_UPKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                       n_relations=n_relations,
                                       dim=dim))

        for i in range(n_hops):
            self.convs_ui.append(
                Aggregator_UPKGAN_user(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                       n_relations=n_relations,
                                       dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

    def forward(self, graph, graph_UIS, entity_embed, W_R, common_like, user_like):
        """KG agg"""
        entity_embed0 = entity_embed

        # entity_emb_all = []
        # entity_emb_all.append(entity_embed[:self.n_items, :])

        entity_emb_all = entity_embed[:self.n_items, :]
        # entity_emb_all = F.normalize(entity_embed[:self.n_items, :], dim=1)

        for i in range(self.n_hops):
            entity_embed0 = self.convs[i](graph, entity_embed0, W_R)

            entity_embed0 = F.normalize(entity_embed0, dim=1)

            # entity_emb_all.append(entity_embed0[:self.n_items, :])
            entity_emb_all = torch.add(entity_emb_all, entity_embed0[:self.n_items, :])

        # entity_emb_all = torch.cat(entity_emb_all, dim=1)

        """user agg"""
        # att = torch.exp(torch.matmul(user_like, common_like.permute(1, 0)))
        att = torch.exp(torch.cosine_similarity(user_like.unsqueeze(1), common_like.unsqueeze(0), dim=2))
        # att = torch.cosine_similarity(user_like.unsqueeze(1), common_like.unsqueeze(0), dim=2)+1
        att_sum = torch.sum(att, dim=1)
        att = att / att_sum.unsqueeze(1)

        common_user_embed = []
        for i in range(self.num_like):
            entity_embed0 = entity_embed

            user_f = []
            # user_f.append(common_like[i].unsqueeze(0).expand(self.n_users, self.dim))
            for j in range(self.n_hops):
                entity_embed0, user_embed = \
                    self.convs_ui[j](graph, graph_UIS, entity_embed0, W_R, common_like[i], user_like)

                entity_embed0 = F.normalize(entity_embed0, dim=1)

                user_f.append(F.normalize(user_embed, dim=1))
                # user_f.append(user_embed)

            # user_f = torch.cat(user_f, dim=1)

            user_f = torch.cat(user_f, dim=1).reshape(self.n_users, self.n_hops, -1)
            user_f = torch.sum(user_f, dim=1)
            # user_f = user_f + common_like[i].unsqueeze(0)

            common_user_embed.append(user_f)

        common_user_embed = torch.cat(common_user_embed, dim=1). \
            reshape(self.n_users, self.num_like, -1)

        """all user"""
        user_emb_all = torch.matmul(att.unsqueeze(1), common_user_embed).squeeze(1)
        user_emb_all = user_like + user_emb_all
        # user_emb_all = F.normalize(user_like, dim=1) + user_emb_all

        # """user dislike agg"""
        # att = torch.exp(torch.cosine_similarity(user_like.unsqueeze(1), common_dislike.unsqueeze(0), dim=2))
        # att_sum = torch.sum(att, dim=1)
        # att = att / att_sum.unsqueeze(1)
        #
        # common_user_embed = []
        # for i in range(self.num_like):
        #     entity_embed0 = entity_embed
        #
        #     user_f = []
        #     for j in range(self.n_hops):
        #         entity_embed0, user_embed = self.convs_ui[j](graph, graph_UIS, entity_embed0, W_R, common_dislike[i])
        #         # user_f.append(user_f)
        #
        #         user_f.append(F.normalize(user_embed, dim=1))
        #
        #     user_f = torch.cat(user_f, dim=1).reshape(self.n_users, self.n_hops, -1)
        #     user_f = torch.sum(user_f, dim=1)
        #
        #     user_f = user_f + common_dislike[i].unsqueeze(0)
        #
        #     common_user_embed.append(user_f)
        #
        # common_user_embed = torch.cat(common_user_embed, dim=1). \
        #     reshape(self.n_users, self.num_like, -1)
        #
        # """all user"""
        # user_emb_all_dislike = torch.matmul(att.unsqueeze(1), common_user_embed).squeeze(1)

        return entity_emb_all, user_emb_all


class GraphConv_UPKGAN_kmean(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like):
        super(GraphConv_UPKGAN_kmean, self).__init__()

        self.convs = nn.ModuleList()
        self.convs_ui = nn.ModuleList()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like

        for i in range(n_hops):
            self.convs.append(
                Aggregator_UPKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                       n_relations=n_relations,
                                       dim=dim))

        for i in range(n_hops):
            self.convs_ui.append(
                Aggregator_UPKGAN_user(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                       n_relations=n_relations,
                                       dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

    def forward(self, graph, graph_UIS, entity_embed, W_R, user_like):
        """KG agg"""
        entity_embed0 = entity_embed

        # entity_emb_all = []
        # entity_emb_all.append(entity_embed[:self.n_items, :])

        entity_emb_all = entity_embed[:self.n_items, :]
        # entity_emb_all = F.normalize(entity_embed[:self.n_items, :], dim=1)

        for i in range(self.n_hops):
            entity_embed0 = self.convs[i](graph, entity_embed0, W_R)

            entity_embed0 = F.normalize(entity_embed0, dim=1)

            # entity_emb_all.append(entity_embed0[:self.n_items, :])
            entity_emb_all = torch.add(entity_emb_all, entity_embed0[:self.n_items, :])

        # entity_emb_all = torch.cat(entity_emb_all, dim=1)

        """user agg"""
        kmeans = KMeans(n_clusters=self.num_like, max_iter=100).fit(user_like.detach().cpu().numpy())
        common_like = torch.tensor(kmeans.cluster_centers_, device=entity_embed.device)
        user_cluster = kmeans.labels_.tolist()

        common_user_embed = []
        for i in range(self.num_like):
            entity_embed0 = entity_embed

            user_f = []
            # user_f.append(common_like[i].unsqueeze(0).expand(self.n_users, self.dim))
            for j in range(self.n_hops):
                entity_embed0, user_embed = \
                    self.convs_ui[j](graph, graph_UIS, entity_embed0, W_R, common_like[i], user_like)

                entity_embed0 = F.normalize(entity_embed0, dim=1)
                user_f.append(F.normalize(user_embed, dim=1))

            user_f = torch.cat(user_f, dim=1).reshape(self.n_users, self.n_hops, -1)
            user_f = torch.sum(user_f, dim=1)

            common_user_embed.append(user_f)

        common_user_embed = torch.cat(common_user_embed, dim=1). \
            reshape(self.n_users, self.num_like, -1)

        """all user"""
        user_emb_all = common_user_embed[list(range(self.n_users)), user_cluster, :]
        user_emb_all = user_like + user_emb_all

        return entity_emb_all, user_emb_all


class GraphConv_UIKGAN(nn.Module):

    def __init__(self, dim, n_hops, n_users, n_items, n_entities, n_relations, num_like):
        super(GraphConv_UIKGAN, self).__init__()

        self.dim = dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.num_like = num_like

        # self.cal_user0 = cal_UIKGAN_user0(n_users=n_users, n_entities=n_entities, n_items=n_items,
        #                                   n_relations=n_relations,
        #                                   dim=dim)

        self.kmeans = []
        for i in range(n_hops):
            self.kmeans.append(KMeans(n_clusters=self.num_like))
            self.kmeans[i].num = 1

        self.convs_i = nn.ModuleList()
        self.convs_u = nn.ModuleList()
        for i in range(n_hops):
            self.convs_i.append(
                Aggregator_UIKGAN_item(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                       n_relations=n_relations,
                                       dim=dim))

        for i in range(n_hops):
            self.convs_u.append(nn.ModuleList())
            for j in range(num_like):
                self.convs_u[i].append(
                    Aggregator_UIKGAN_user(n_users=n_users, n_entities=n_entities, n_items=n_items,
                                           n_relations=n_relations,
                                           dim=dim))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout

    def forward(self, graph, graph_UIS, entity_embed, W_R, user_embed, coe_list):
        # """cal user0"""
        # user_embed = self.cal_user0(graph_UIS, entity_embed)

        """KGAN agg"""
        entity_emb_all = []
        user_emb_all = []

        entity_emb_all.append(entity_embed)  # 0-hop
        user_emb_all.append(user_embed)  # 0-hop

        for i in range(self.n_hops):
            entity_embed0 = entity_embed

            "----------------------------agg item by KG----------------------------"
            entity_embed = self.convs_i[i](graph, entity_embed0, W_R)
            entity_embed = F.normalize(entity_embed, dim=1)

            entity_emb_all.append(entity_embed)  # (i+1)-hop

            "--------------------agg user by user intent-aware KGAN----------"

            "1 user-intent"
            if self.training:
                if self.kmeans[i].num == 1:
                    self.kmeans[i] = self.kmeans[i].fit(user_embed.detach().cpu().numpy())
                    self.kmeans[i].num = 2
                else:
                    self.kmeans[i] = KMeans(n_clusters=self.num_like, init=self.kmeans[i].cluster_centers_)
                    self.kmeans[i].num = 2
                    self.kmeans[i] = self.kmeans[i].fit(user_embed.detach().cpu().numpy())
            common_like = torch.tensor(self.kmeans[i].cluster_centers_, device=entity_embed.device)
            user_cluster = self.kmeans[i].labels_.tolist()

            "2 KG agg using user-intent"
            user_embed_mul = []
            for j in range(self.num_like):
                user_embed1 = \
                    self.convs_u[i][j](graph, graph_UIS, entity_embed0, W_R, common_like[j], user_embed)

                user_embed_mul.append(F.normalize(user_embed1, dim=1))

            user_embed_mul = torch.cat(user_embed_mul, dim=1). \
                reshape(self.n_users, self.num_like, -1)

            user_embed = user_embed_mul[list(range(self.n_users)), user_cluster, :]

            user_emb_all.append(user_embed)  # (i+1)-hop

        """OUTPUT item"""
        # entity_emb_all = torch.cat(entity_emb_all,dim=1)

        entity_emb_all = torch.cat(entity_emb_all, dim=1).reshape(self.n_entities, (1 + self.n_hops), self.dim)
        entity_emb_all = torch.sum(entity_emb_all, dim=1)

        # entity_emb_all_f = entity_emb_all[0] * coe_list[0]
        # for i in range(1, 1 + self.n_hops):
        #     entity_emb_all_f = entity_emb_all_f + entity_emb_all[i] * coe_list[i]
        """OUTPUT user"""
        # user_emb_all = torch.cat(user_emb_all,dim=1)

        user_emb_all = torch.cat(user_emb_all, dim=1).reshape(self.n_users, (1 + self.n_hops), self.dim)
        user_emb_all = torch.sum(user_emb_all, dim=1)

        # user_emb_all_f = user_emb_all[0] * coe_list[0]
        # for i in range(1, 1 + self.n_hops):
        #     user_emb_all_f = user_emb_all_f + user_emb_all[i] * coe_list[i]

        return entity_emb_all, user_emb_all
        # return entity_emb_all_f, user_emb_all_f


class GraphConv_NFGNN(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, dim, dim1, coe1, coe2, n_hops, n_users, n_items, n_entities,
                 n_relations, new_r_all, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv_NFGNN, self).__init__()

        self.convs_item = nn.ModuleList()
        self.convs_user = nn.ModuleList()
        self.dim = dim
        self.dim1 = dim1
        self.coe1 = coe1
        self.coe2 = coe2
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.temperature = 0.2

        for i in range(n_hops):
            self.convs_item.append(
                Aggregator_NFGNN_item(n_users=n_users,
                                      n_entities=n_entities,
                                      n_items=n_items,
                                      n_relations=n_relations,
                                      dim=dim,
                                      dropuot=node_dropout_rate))
            self.convs_user.append(
                Aggregator_NFGNN_user(n_users=n_users,
                                      n_entities=n_entities,
                                      n_items=n_items,
                                      n_relations=n_relations,
                                      dim=dim,
                                      dropuot=node_dropout_rate))

    def cal_attribute_noatt(self, edges):
        num_r = []
        for i in range(self.n_relations):
            num_r.append(torch.where(edges.data['type'] == i)[0].shape[0])
        num_r = torch.tensor(num_r, device=edges.data['type'].device)
        return {'num_homo': num_r}

    def forward(self, graph, graph_UIS,
                entity_embed,
                W_R, new_r_all,
                user_filter,
                user_dict, user_select, sub_edges,
                mess_dropout=True,
                node_dropout=False):

        # sub_edges_all = []
        # for user in user_select:
        #     sub_edges_all += sub_edges[user]
        # sub_edges_all = list(set(sub_edges_all))
        #
        # graph = dgl.edge_subgraph(graph, sub_edges_all, preserve_nodes=True)

        """KG agg"""
        entity_emb_f_all = []
        user_emb_f_all = []

        entity_embed0 = entity_embed
        for i in range(len(self.convs_item)):
            entity_embed0 = self.convs_item[i](graph, entity_embed0, W_R)
            entity_emb_f_all.append(F.normalize(entity_embed0, dim=1))
        entity_emb_f_all = torch.cat(entity_emb_f_all, dim=1)

        # """user agg 1"""
        for user in user_select:
            entity_embed0 = entity_embed
            user_c_emb = []
            sub_edges_c = sub_edges[user]

            graph1 = dgl.edge_subgraph(graph, sub_edges_c, preserve_nodes=True)
            # graph1 = graph1.local_var()

            for i in range(len(self.convs_user)):
                user_f, entity_embed0 = self.convs_user[i](
                    graph1, entity_embed0, W_R,
                    user_filter[user],
                    user_dict[user], sub_edges_c)
                user_c_emb.append(F.normalize(user_f, dim=1))

            user_c_emb = torch.cat(user_c_emb, dim=1)
            user_emb_f_all.append(user_c_emb)
        user_emb_f_all = torch.cat(user_emb_f_all, dim=0)

        """user agg 2"""
        # for i in range(len(self.convs_user)):
        #     user_f, entity_embed0 = self.convs_user[i](
        #         graph, entity_embed0, W_R,
        #         user_filter[user_select], user_node[user_select],
        #         user_dict[user_select])
        #     user_c_emb.append(F.normalize(user_f, dim=1))
        #
        # for user in user_select:
        #     entity_embed0 = entity_embed
        #     user_c_emb = []
        #     sub_edges_c = sub_edges[user]
        #
        #     for i in range(len(self.convs_user)):
        #         user_f, entity_embed0 = self.convs_user[i](
        #             graph, entity_embed0, W_R,
        #             user_filter[user], user_node[user],
        #             user_dict[user], sub_edges_c)
        #         user_c_emb.append(F.normalize(user_f, dim=1))
        #
        #     user_c_emb = torch.cat(user_c_emb, dim=1)
        #     user_emb_f_all.append(user_c_emb)
        # user_emb_f_all = torch.cat(user_emb_f_all, dim=0)

        return entity_emb_f_all, user_emb_f_all


class GraphConv_KGUI4(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, dim, n_hops, n_users, n_items, n_entities,
                 n_relations, n_pre, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv_KGUI4, self).__init__()

        self.convs = nn.ModuleList()
        self.dim = dim
        self.n_hops = n_hops
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_pre = n_pre
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_

        weight = initializer(torch.empty(n_relations, dim, dim))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, dim,dim]
        bias = initializer(torch.empty(n_relations, dim))  # not include interact
        self.bias = nn.Parameter(bias)  # [n_relations - 1, dim,dim]
        # user_pre
        self.item_atr = nn.ParameterList()
        self.user_pre = nn.ParameterList()
        for i in range(n_hops):
            # self.user_pre.append(nn.Parameter(initializer(torch.empty(n_users, n_relations, dim))))
            # self.user_pre.append(nn.Parameter(initializer(torch.empty(n_relations, dim))))
            self.item_atr.append(nn.Parameter(initializer(torch.empty(n_entities, dim))))
            self.user_pre.append(nn.Parameter(initializer(torch.empty(n_users, dim))))
            # self.user_pre.append(nn.Parameter(initializer(torch.empty(n_users, int(dim * self.n_relations)))))

        for i in range(n_hops):
            self.convs.append(nn.ModuleList())
            for j in range(n_pre + 1):
                self.convs[i].append(
                    Aggregator_KGUI4(n_users=n_users, n_items=n_items, n_relations=n_relations,
                                     dropuot=node_dropout_rate))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, graph, user_dict,
                user_emb, entity_emb, latent_emb,
                edge_index, edge_type,
                edge_index_user, edge_type_user,
                interact_mat, mess_dropout=True, node_dropout=False):
        """node dropout"""
        entity_emb_f_all = []
        user_res_emb = []
        for i in range(len(self.convs)):
            """item agg"""
            entity_emb, user_emb = self.convs[i][0](graph, entity_emb,
                                                    edge_index, edge_type, edge_index_user, edge_type_user,
                                                    interact_mat,
                                                    self.weight,
                                                    node_dropout)
            item_att_atr = torch.matmul(entity_emb, self.item_atr[i].unsqueeze(2)).squeeze(2)
            item_att_atr = F.softmax(torch.exp(item_att_atr), dim=1).unsqueeze(1)
            entity_emb = F.normalize(torch.matmul(item_att_atr, entity_emb).squeeze(1), dim=1)

            """user agg """
            user_att_atr = torch.matmul(user_emb, self.user_pre[i].unsqueeze(2)).squeeze(2)
            user_att_atr = F.softmax(torch.exp(user_att_atr), dim=1).unsqueeze(1)
            user_res_emb.append(F.normalize(torch.matmul(user_att_atr, user_emb).squeeze(1), dim=1))

            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_res_emb[i] = self.dropout(user_res_emb[i])
            entity_emb_f_all.append(entity_emb)

        entity = torch.cat(entity_emb_f_all, dim=1)
        user = torch.cat(user_res_emb, dim=1)
        """agg 3"""
        # entity = \
        #     torch.sum(torch.cat(entity_emb_f_all, dim=0).reshape(self.n_hops, self.n_entities, self.dim), dim=0)
        # user = \
        #     torch.sum(torch.cat(user_res_emb, dim=0).reshape(self.n_hops, self.n_users, self.dim), dim=0)
        # user = torch.sparse.mm(interact_mat, entity)
        return entity, user
        # return entity_emb, user_emb


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_factors=self.n_factors,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e, cor)

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)[:-1]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items, cor):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss + cor_loss, mf_loss, emb_loss, cor


class RS_KGUI(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(RS_KGUI, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.n_new_node = data_config['n_new_node']

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = adj_mat
        # self.graph = graph
        # self.edge_index, self.edge_type = self._get_edges(graph)
        # self.edge_index_user, self.edge_type_user = self._get_edges(graph_user)
        self.edge_index, self.edge_type = 1, 1
        self.edge_index_user, self.edge_type_user = 1, 1

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv_KGUI_linear(dim=self.emb_size,
                                     n_hops=self.context_hops,
                                     n_users=self.n_users,
                                     n_items=self.n_items,
                                     n_entities=self.n_entities,
                                     n_relations=self.n_relations,
                                     n_new_node=self.n_new_node,
                                     interact_mat=self.interact_mat,
                                     ind=self.ind,
                                     node_dropout_rate=self.node_dropout_rate,
                                     mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, graph_atr2pre, graph_pre2user):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb = self.gcn(graph, graph_atr2pre, graph_pre2user,
                                                user_emb, item_emb,
                                                self.latent_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.edge_index_user,
                                                self.edge_type_user,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_atr2pre, graph_pre2user):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(graph, graph_atr2pre, graph_pre2user,
                        user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.edge_index_user,
                        self.edge_type_user,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss, mf_loss, emb_loss


class RS_KGUI1(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(RS_KGUI1, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = adj_mat
        self.edge_index, self.edge_type = 1, 1
        self.edge_index_user, self.edge_type_user = 1, 1

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv_KGUI1(dim=self.emb_size,
                               n_hops=self.context_hops,
                               n_users=self.n_users,
                               n_items=self.n_items,
                               n_entities=self.n_entities,
                               n_relations=self.n_relations,
                               interact_mat=self.interact_mat,
                               ind=self.ind,
                               node_dropout_rate=self.node_dropout_rate,
                               mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, graph_user):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb = self.gcn(graph, graph_user,
                                                user_emb, item_emb,
                                                self.latent_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.edge_index_user,
                                                self.edge_type_user,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_user):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(graph, graph_user,
                        user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.edge_index_user,
                        self.edge_type_user,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss, mf_loss, emb_loss


class RS_KGUI2(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(RS_KGUI2, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = adj_mat
        self.edge_index, self.edge_type = 1, 1
        self.edge_index_user, self.edge_type_user = 1, 1

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv_KGUI2(dim=self.emb_size,
                               n_hops=self.context_hops,
                               n_users=self.n_users,
                               n_items=self.n_items,
                               n_entities=self.n_entities,
                               n_relations=self.n_relations,
                               interact_mat=self.interact_mat,
                               ind=self.ind,
                               node_dropout_rate=self.node_dropout_rate,
                               mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, user_pre_r, user_pre_node):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb = self.gcn(graph, user_pre_r, user_pre_node,
                                                user_emb, item_emb,
                                                self.latent_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.edge_index_user,
                                                self.edge_type_user,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, user_pre_r, user_pre_node):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(graph, user_pre_r, user_pre_node,
                        user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.edge_index_user,
                        self.edge_type_user,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss, mf_loss, emb_loss


class RS_KGUI3(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(RS_KGUI3, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = adj_mat
        self.edge_index, self.edge_type = 1, 1
        self.edge_index_user, self.edge_type_user = 1, 1

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv_KGUI3(dim=self.emb_size,
                               n_hops=self.context_hops,
                               n_users=self.n_users,
                               n_items=self.n_items,
                               n_entities=self.n_entities,
                               n_pre=self.n_pre,
                               n_relations=self.n_relations,
                               interact_mat=self.interact_mat,
                               ind=self.ind,
                               node_dropout_rate=self.node_dropout_rate,
                               mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, graph_i2u, graph_u2u, user_dict):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb = self.gcn(graph, graph_i2u, graph_u2u, user_dict,
                                                user_emb, item_emb,
                                                self.latent_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.edge_index_user,
                                                self.edge_type_user,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_i2u, graph_u2u, user_dict):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(graph, graph_i2u, graph_u2u, user_dict,
                        user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.edge_index_user,
                        self.edge_type_user,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss, mf_loss, emb_loss


class RS_KGUI3_1(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(RS_KGUI3_1, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = adj_mat
        self.edge_index, self.edge_type = 1, 1
        self.edge_index_user, self.edge_type_user = 1, 1

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()
        self.user_idx = torch.arange(self.n_users)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv_KGUI3_1(dim=self.emb_size,
                                 n_hops=self.context_hops,
                                 n_users=self.n_users,
                                 n_items=self.n_items,
                                 n_entities=self.n_entities,
                                 n_pre=self.n_pre,
                                 n_relations=self.n_relations,
                                 interact_mat=self.interact_mat,
                                 ind=self.ind,
                                 node_dropout_rate=self.node_dropout_rate,
                                 mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, graph_i2u, graph_u2u, graph_i2u_0, user_dict):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb = self.gcn(user, graph, graph_i2u, graph_u2u, graph_i2u_0, user_dict,
                                                user_emb, item_emb,
                                                self.latent_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.edge_index_user,
                                                self.edge_type_user,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        # u_e = user_gcn_emb[user]
        u_e = user_gcn_emb
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_i2u, graph_u2u, graph_i2u_0, user_dict):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(self.user_idx, graph, graph_i2u, graph_u2u, graph_i2u_0, user_dict,
                        user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.edge_index_user,
                        self.edge_type_user,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss, mf_loss, emb_loss


class RS_KGUI_sub(nn.Module):
    def __init__(self, data_config, args_config, new_r_all):
        super(RS_KGUI_sub, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.new_r_all = new_r_all

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.entity_embed = nn.Parameter(self.entity_embed)
        self.W_R = nn.Parameter(self.W_R)
        self.item_embed_ui = nn.Parameter(self.item_embed_ui)
        self.user_embed_ui = nn.Parameter(self.user_embed_ui)

        self.gcn = self._init_model()

    def _init_weight(self):
        # initializer = nn.init.xavier_uniform_
        initializer = nn.init.xavier_normal_
        kgain = 1.414
        self.entity_embed = initializer(torch.empty(self.n_entities, self.emb_size), gain=kgain)
        self.W_R = initializer(torch.empty(self.n_relations, self.emb_size, self.emb_size), gain=kgain)

        self.item_embed_ui = initializer(torch.empty(self.n_items, self.emb_size), gain=kgain)
        self.user_embed_ui = initializer(torch.empty(self.n_users, self.emb_size), gain=kgain)

    def _init_model(self):
        return GraphConv_KGUI3_1(dim=self.emb_size,
                                 n_hops=self.context_hops,
                                 n_users=self.n_users,
                                 n_items=self.n_items,
                                 n_entities=self.n_entities,
                                 n_relations=self.n_relations,
                                 new_r_all=self.new_r_all,
                                 node_dropout_rate=self.node_dropout_rate,
                                 mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, graph_UIS):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS,
                                              self.entity_embed,
                                              self.user_embed_ui, self.item_embed_ui,
                                              self.W_R, self.new_r_all,
                                              mess_dropout=self.mess_dropout,
                                              node_dropout=self.node_dropout)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_UIS):
        return self.gcn(graph, graph_UIS,
                        self.entity_embed,
                        self.user_embed_ui, self.item_embed_ui,
                        self.W_R, self.new_r_all,
                        mess_dropout=self.mess_dropout,
                        node_dropout=self.node_dropout)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]  # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.all_embed(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.all_embed(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.all_embed(neg_t)  # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


class RS_KGUI_sub_mulatr(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(RS_KGUI_sub_mulatr, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.new_r_all = torch.tensor(data_config['new_r_all'], device=self.device).type(torch.int64)

        self.adj_mat = adj_mat
        self.edge_index, self.edge_type = 1, 1
        self.edge_index_user, self.edge_type_user = 1, 1

        self._init_weight()
        self.item_embed = nn.Parameter(self.item_embed)
        self.user_embed = nn.Parameter(self.user_embed)
        self.user_embed1 = nn.Parameter(self.user_embed1)
        self.W_R = nn.Parameter(self.W_R)
        self.w_R = nn.Parameter(self.w_R)
        self.b_R = nn.Parameter(self.b_R)
        self.use_pre = nn.Parameter(self.use_pre)

        self.gcn = self._init_model()
        self.user_idx = (torch.arange(self.n_users), torch.arange(self.n_users))

    def _init_weight(self):
        # initializer = nn.init.xavier_uniform_
        initializer = nn.init.xavier_normal_
        kgain = 1.414
        self.item_embed = initializer(torch.empty(self.n_entities, self.emb_size), gain=kgain)
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size), gain=kgain)
        self.user_embed1 = initializer(torch.empty(self.n_users, self.emb_size), gain=kgain)
        self.W_R = initializer(torch.empty(self.n_relations, self.emb_size, self.emb_size), gain=kgain)
        self.w_R = initializer(torch.empty(self.n_relations, self.emb_size), gain=1)
        self.b_R = initializer(torch.empty(self.n_relations, self.emb_size), gain=1)
        # self.use_pre = torch.ones(self.n_users, int(len(self.new_r_all) * self.context_hops))
        self.use_pre = initializer(torch.empty(self.n_users, self.emb_size), gain=kgain)

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv_KGUI3_1_mulatr(dim=self.emb_size,
                                        n_hops=self.context_hops,
                                        n_users=self.n_users,
                                        n_items=self.n_items,
                                        n_entities=self.n_entities,
                                        n_pre=self.n_pre,
                                        n_relations=self.n_relations,
                                        interact_mat=self.interact_mat,
                                        ind=self.ind,
                                        new_r_all=self.new_r_all,
                                        node_dropout_rate=self.node_dropout_rate,
                                        mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, graph_1, graph_i2u, graph_u2u, graph_i2u_0, graph_i2u_1, sub_idx1, user_dict):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        # user_emb = self.all_embed[:self.n_users, :]
        # item_emb = self.all_embed[self.n_users:, :]
        item_emb = self.item_embed
        user_emb = self.user_embed
        user_emb1 = self.user_embed1

        entity_gcn_emb, user_gcn_emb = self.gcn(sub_idx1, graph, graph_1, graph_i2u, graph_u2u, graph_i2u_0,
                                                graph_i2u_1,
                                                user_dict, user_emb, user_emb1,
                                                item_emb,
                                                self.W_R, self.w_R, self.b_R,
                                                self.use_pre,
                                                self.new_r_all,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        # u_e = user_gcn_emb[user]
        u_e = user_gcn_emb
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e, sub_idx1)

    def generate(self, graph, graph_1, graph_i2u, graph_u2u, graph_i2u_0, graph_i2u_1, sub_idx1, user_dict):
        # user_emb = self.all_embed[:self.n_users, :]
        # item_emb = self.all_embed[self.n_users:, :]
        # item_emb = self.all_embed
        item_emb = self.item_embed
        user_emb = self.user_embed
        user_emb1 = self.user_embed1
        return self.gcn(sub_idx1, graph, graph_1, graph_i2u, graph_u2u, graph_i2u_0, graph_i2u_1,
                        user_dict, user_emb, user_emb1,
                        item_emb,
                        self.W_R, self.w_R, self.b_R,
                        self.use_pre,
                        self.new_r_all,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items, sub_idx1):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor

        pre_loss = torch.mean(
            torch.abs(
                torch.cosine_similarity(
                    self.user_embed[sub_idx1[0]], self.user_embed1[sub_idx1[0]], dim=1))
        )

        return mf_loss + emb_loss, mf_loss, pre_loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]  # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.all_embed(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.all_embed(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.all_embed(neg_t)  # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


class RS_upat(nn.Module):
    def __init__(self, data_config, args_config, new_r_all):
        super(RS_upat, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.new_r_all = new_r_all

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.coe1 = args_config.coe1
        self.coe2 = args_config.coe2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.emb_size1 = args_config.dim1
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.entity_embed = nn.Parameter(self.entity_embed)
        self.W_R = nn.Parameter(self.W_R)

        self.user_att = nn.Parameter(self.user_att)
        self.user_filter = nn.Parameter(self.user_filter)

        self.gcn = self._init_model()

        self.act_user_att = nn.Sigmoid()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        # initializer = nn.init.xavier_normal_
        kgain = 1
        self.entity_embed = initializer(torch.empty(self.n_entities, self.emb_size), gain=kgain)
        self.W_R = initializer(torch.empty(self.n_relations, self.emb_size, self.emb_size), gain=kgain)

        # self.user_att = initializer(torch.empty(self.n_users, self.emb_size1))
        self.user_att = initializer(torch.empty(self.n_users, (1 + self.context_hops) * self.n_relations))
        self.user_filter = initializer(
            torch.empty(self.n_users, (1 + self.context_hops) * self.n_relations, self.emb_size))

    def _init_model(self):
        return GraphConv_upat(dim=self.emb_size,
                              dim1=self.emb_size1,
                              coe1=self.coe1,
                              coe2=self.coe2,
                              n_hops=self.context_hops,
                              n_users=self.n_users,
                              n_items=self.n_items,
                              n_entities=self.n_entities,
                              n_relations=self.n_relations,
                              new_r_all=self.new_r_all,
                              node_dropout_rate=self.node_dropout_rate,
                              mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, graph_UIS, user_dict):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS,
                                              self.entity_embed,
                                              self.W_R, self.new_r_all, self.user_att, self.user_filter,
                                              user_dict,
                                              mess_dropout=self.mess_dropout,
                                              node_dropout=self.node_dropout)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss2(u_e, pos_e, neg_e, user)

    def generate(self, graph, graph_UIS, user_dict):
        return self.gcn(graph, graph_UIS,
                        self.entity_embed,
                        self.W_R, self.new_r_all, self.user_att, self.user_filter,
                        user_dict,
                        mess_dropout=self.mess_dropout,
                        node_dropout=self.node_dropout)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def rating1(self, u_g_embeddings, i_g_embeddings, user_idx):
        user_att_coe = self.act_user_att(self.coe1 * self.user_att)

        rank = []
        for idx, i in enumerate(user_idx):
            scores = torch.abs(u_g_embeddings[idx].unsqueeze(0) - i_g_embeddings)
            # scores = torch.abs(u_g_embeddings[idx].unsqueeze(0) - i_g_embeddings) * user_att_coe[i].unsqueeze(0)
            rank.append(torch.sum(scores, dim=1).unsqueeze(0))
        rank = -torch.cat(rank)

        a = rank[1]
        a11, a1 = torch.sort(a, descending=True)
        a1[:100]

        return rank

    def rating2(self, u_g_embeddings, i_g_embeddings, user_idx):
        user_att_coe = self.act_user_att(self.coe1 * self.user_att)

        rank = []
        for idx, i in enumerate(user_idx):
            score_c = torch.sum(torch.sum(u_g_embeddings[idx].unsqueeze(0) * i_g_embeddings, dim=2).unsqueeze(0), dim=2)
            rank.append(score_c)
        rank = torch.cat(rank)

        # a = rank[1]
        # a11, a1 = torch.sort(a, descending=True)
        # a1[:100]

        return rank

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # regularizer1 = -torch.mean(torch.exp(self.coe2 * (pos_scores - neg_scores)))
        # regularizer1 = -torch.mean((self.coe2 * (pos_scores - neg_scores)))

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_bpr_loss1(self, users, pos_items, neg_items, user_idx):
        batch_size = users.shape[0]

        user_att_coe = self.act_user_att(self.coe1 * self.user_att)[user_idx]

        pos_scores = torch.abs(users - pos_items)
        neg_scores = torch.abs(users - neg_items)
        # pos_scores = pos_scores * user_att_coe
        # neg_scores = neg_scores * user_att_coe
        pos_scores = torch.mean(torch.sum(pos_scores, dim=1))
        neg_scores = torch.mean(torch.sum(neg_scores, dim=1))

        mf_loss = pos_scores - neg_scores

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # regularizer1 = -torch.mean(torch.exp(self.coe2 * (pos_scores - neg_scores)))
        # regularizer1 = -torch.mean((self.coe2 * (pos_scores - neg_scores)))

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_bpr_loss2(self, users, pos_items, neg_items, user_idx):
        batch_size = users.shape[0]

        user_att_coe = self.act_user_att(self.coe1 * self.user_att)[user_idx]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=2)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=2)
        # pos_scores = pos_scores * user_att_coe
        # neg_scores = neg_scores * user_att_coe
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.sum(neg_scores, dim=1)
        # pos_scores = torch.mean(torch.sum(pos_scores, dim=1))
        # neg_scores = torch.mean(torch.sum(neg_scores, dim=1))
        # mf_loss = pos_scores - neg_scores
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # regularizer1 = -torch.mean(torch.exp(self.coe2 * (pos_scores - neg_scores)))
        # regularizer1 = -torch.mean((self.coe2 * (pos_scores - neg_scores)))

        return mf_loss + emb_loss, mf_loss, emb_loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]  # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.all_embed(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.all_embed(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.all_embed(neg_t)  # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


class RS_NFGNN(nn.Module):
    def __init__(self, data_config, args_config, new_r_all):
        super(RS_NFGNN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.new_r_all = new_r_all

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.coe1 = args_config.coe1
        self.coe2 = args_config.coe2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.emb_size1 = args_config.dim1
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.entity_embed = nn.Parameter(self.entity_embed)
        self.W_R = nn.Parameter(self.W_R)

        # self.user_att = nn.Parameter(self.user_att)
        self.user_filter = nn.Parameter(self.user_filter)
        # self.user_node = nn.Parameter(self.user_node)

        self.gcn = self._init_model()

        self.act_user_att = nn.Sigmoid()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        initializer1 = nn.init.normal_

        # kgain = 1
        # self.entity_embed = initializer(torch.empty(self.n_entities, self.emb_size), gain=kgain)
        # self.W_R = initializer(torch.empty(self.n_relations, self.emb_size), gain=kgain)
        # self.user_filter = initializer(
        #     torch.empty(self.n_users, self.n_relations, self.emb_size))

        std = 0.1
        std1 = 1
        self.entity_embed = initializer1(torch.empty(self.n_entities, self.emb_size), std=std)
        self.W_R = initializer1(torch.empty(self.n_relations, self.emb_size), std=std)
        # self.user_filter = initializer1(
        #     torch.empty(self.n_users, self.n_relations, self.emb_size), std=std)
        self.user_filter = initializer1(
            torch.empty(self.n_users, self.emb_size), std=std)
        self.user_node = 1
        # self.user_filter=nn.Embedding()

    def _init_model(self):
        return GraphConv_NFGNN(dim=self.emb_size,
                               dim1=self.emb_size1,
                               coe1=self.coe1,
                               coe2=self.coe2,
                               n_hops=self.context_hops,
                               n_users=self.n_users,
                               n_items=self.n_items,
                               n_entities=self.n_entities,
                               n_relations=self.n_relations,
                               new_r_all=self.new_r_all,
                               node_dropout_rate=self.node_dropout_rate,
                               mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, graph_UIS, user_dict, user_select, sub_edges, sub_idx):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS,
                                              self.entity_embed,
                                              self.W_R, self.new_r_all,
                                              self.user_filter,
                                              user_dict, user_select, sub_edges,
                                              mess_dropout=self.mess_dropout,
                                              node_dropout=self.node_dropout)

        # u_e = user_gcn_emb[user]
        u_e = user_gcn_emb[sub_idx[1]]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_UIS, user_dict, user_select, sub_edges):
        return self.gcn(graph, graph_UIS,
                        self.entity_embed,
                        self.W_R, self.new_r_all,
                        self.user_filter,
                        user_dict, user_select, sub_edges,
                        mess_dropout=self.mess_dropout,
                        node_dropout=self.node_dropout)

    def generate_item(self, graph):
        """KG agg"""
        entity_emb_f_all = []

        entity_embed0 = self.entity_embed
        for i in range(len(self.gcn.convs_item)):
            entity_embed0 = self.gcn.convs_item[i](graph, entity_embed0, self.W_R)
            entity_emb_f_all.append(F.normalize(entity_embed0, dim=1))
        entity_emb_f_all = torch.cat(entity_emb_f_all, dim=1)

        return entity_emb_f_all

    def generate_user(self, graph, user_dict, user_select, sub_edges):
        user_emb_f_all = []

        for user in user_select:
            entity_embed0 = self.entity_embed
            user_c_emb = []
            sub_edges_c = sub_edges[user]

            graph1 = dgl.edge_subgraph(graph, sub_edges_c, preserve_nodes=True)

            for i in range(len(self.gcn.convs_user)):
                user_f, entity_embed0 = self.gcn.convs_user[i](
                    graph1, entity_embed0, self.W_R,
                    self.user_filter[user], user_dict[user], sub_edges_c)
                user_c_emb.append(F.normalize(user_f, dim=1))
            user_c_emb = torch.cat(user_c_emb, dim=1)
            user_emb_f_all.append(user_c_emb)

        user_emb_f_all = torch.cat(user_emb_f_all, dim=0)

        return user_emb_f_all

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def rating1(self, u_g_embeddings, i_g_embeddings, user_idx):
        user_att_coe = self.act_user_att(self.coe1 * self.user_att)

        rank = []
        for idx, i in enumerate(user_idx):
            scores = torch.abs(u_g_embeddings[idx].unsqueeze(0) - i_g_embeddings)
            # scores = torch.abs(u_g_embeddings[idx].unsqueeze(0) - i_g_embeddings) * user_att_coe[i].unsqueeze(0)
            rank.append(torch.sum(scores, dim=1).unsqueeze(0))
        rank = -torch.cat(rank)

        a = rank[1]
        a11, a1 = torch.sort(a, descending=True)
        a1[:100]

        return rank

    def rating2(self, u_g_embeddings, i_g_embeddings, user_idx):
        user_att_coe = self.act_user_att(self.coe1 * self.user_att)

        rank = []
        for idx, i in enumerate(user_idx):
            score_c = torch.sum(torch.sum(u_g_embeddings[idx].unsqueeze(0) * i_g_embeddings, dim=2).unsqueeze(0), dim=2)
            rank.append(score_c)
        rank = torch.cat(rank)

        # a = rank[1]
        # a11, a1 = torch.sort(a, descending=True)
        # a1[:100]

        return rank

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # regularizer1 = -torch.mean(torch.exp(self.coe2 * (pos_scores - neg_scores)))
        # regularizer1 = -torch.mean((self.coe2 * (pos_scores - neg_scores)))

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_bpr_loss1(self, users, pos_items, neg_items, user_idx):
        batch_size = users.shape[0]

        user_att_coe = self.act_user_att(self.coe1 * self.user_att)[user_idx]

        pos_scores = torch.abs(users - pos_items)
        neg_scores = torch.abs(users - neg_items)
        # pos_scores = pos_scores * user_att_coe
        # neg_scores = neg_scores * user_att_coe
        pos_scores = torch.mean(torch.sum(pos_scores, dim=1))
        neg_scores = torch.mean(torch.sum(neg_scores, dim=1))

        mf_loss = pos_scores - neg_scores

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # regularizer1 = -torch.mean(torch.exp(self.coe2 * (pos_scores - neg_scores)))
        # regularizer1 = -torch.mean((self.coe2 * (pos_scores - neg_scores)))

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_bpr_loss2(self, users, pos_items, neg_items, user_idx):
        batch_size = users.shape[0]

        user_att_coe = self.act_user_att(self.coe1 * self.user_att)[user_idx]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=2)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=2)
        # pos_scores = pos_scores * user_att_coe
        # neg_scores = neg_scores * user_att_coe
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.sum(neg_scores, dim=1)
        # pos_scores = torch.mean(torch.sum(pos_scores, dim=1))
        # neg_scores = torch.mean(torch.sum(neg_scores, dim=1))
        # mf_loss = pos_scores - neg_scores
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # regularizer1 = -torch.mean(torch.exp(self.coe2 * (pos_scores - neg_scores)))
        # regularizer1 = -torch.mean((self.coe2 * (pos_scores - neg_scores)))

        return mf_loss + emb_loss, mf_loss, emb_loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]  # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.all_embed(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.all_embed(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.all_embed(neg_t)  # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


class RS_UMIKGAN(nn.Module):
    def __init__(self, data_config, args_config):
        super(RS_UMIKGAN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.ind = args_config.ind
        self.temperature = 0.2

        self.num_like = args_config.num_like
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.W_R = nn.Parameter(self.W_R)
        self.common_like = nn.Parameter(self.common_like)
        self.user_like = nn.Parameter(self.user_like)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.W_R = initializer(torch.empty(self.n_relations, self.emb_size))
        self.user_like = initializer(torch.empty(self.num_like, self.emb_size))
        self.common_like = initializer(torch.empty(self.num_like, self.emb_size))

    def _init_model(self):
        return GraphConv_UMIKGAN0(dim=self.emb_size,
                                  n_hops=self.context_hops,
                                  n_users=self.n_users,
                                  n_items=self.n_items,
                                  n_entities=self.n_entities,
                                  n_relations=self.n_relations,
                                  num_like=self.num_like)

    def forward(self, batch, graph, graph_UIS):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS, self.all_embed, self.W_R, self.common_like)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_UIS):
        return self.gcn(graph, graph_UIS, self.all_embed, self.W_R, self.common_like)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # loss1 = 0
        # for i in range(self.num_like - 1):
        #     for j in range(i + 1, self.num_like):
        #         loss1 = loss1 + torch.cosine_similarity(self.common_like[i], self.common_like[j], dim=0)
        # loss1 = self.sim_decay * loss1 / (self.num_like * (self.num_like - 1) / 2)

        loss1 = self._cul_cor()
        loss1 = self.sim_decay * loss1

        return mf_loss + emb_loss + loss1, mf_loss, loss1

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.num_like):
                for j in range(i + 1, self.num_like):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.common_like[i], self.common_like[j])
                    else:
                        cor += CosineSimilarity(self.common_like[i], self.common_like[j])
        return cor


class RS_MUIGAN(nn.Module):
    def __init__(self, data_config, args_config):
        super(RS_MUIGAN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.ind = args_config.ind
        self.temperature = 0.2

        self.num_like = args_config.num_like
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.W_R = nn.Parameter(self.W_R)
        self.common_like = nn.Parameter(self.common_like)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        gain = 1.414
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size), gain=gain)
        self.W_R = initializer(torch.empty(self.n_relations, self.emb_size), gain=gain)
        self.common_like = initializer(torch.empty(self.num_like, self.emb_size), gain=gain)

    def _init_model(self):
        return GraphConv_MUIGAN(dim=self.emb_size,
                                n_hops=self.context_hops,
                                n_users=self.n_users,
                                n_items=self.n_items,
                                n_entities=self.n_entities,
                                n_relations=self.n_relations,
                                num_like=self.num_like)

    def forward(self, batch, graph, graph_UIS):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS, self.all_embed, self.W_R, self.common_like)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_UIS):
        return self.gcn(graph, graph_UIS, self.all_embed, self.W_R, self.common_like)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # loss1 = 0
        # for i in range(self.num_like - 1):
        #     for j in range(i + 1, self.num_like):
        #         loss1 = loss1 + torch.cosine_similarity(self.common_like[i], self.common_like[j], dim=0)
        # loss1 = self.sim_decay * loss1 / (self.num_like * (self.num_like - 1) / 2)

        loss1 = self._cul_cor()
        loss2 = self.sim_decay * loss1

        return mf_loss + emb_loss + loss2, mf_loss, loss1

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.num_like):
                for j in range(i + 1, self.num_like):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.common_like[i], self.common_like[j])
                    else:
                        cor += CosineSimilarity(self.common_like[i], self.common_like[j])
        return cor


class RS_MAKG(nn.Module):
    def __init__(self, data_config, args_config):
        super(RS_MAKG, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.ind = args_config.ind
        self.temperature = 0.2

        self.num_like = args_config.num_like
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        # self.W_R_user_like = nn.Parameter(self.W_R_user_like)

        self.W_R = nn.Parameter(self.W_R)
        self.user_like = nn.Parameter(self.user_like)
        self.common_like = nn.Parameter(self.common_like)

        self.gcn = self._init_model()

        index = np.arange(self.n_relations)
        np.random.shuffle(index)
        self.index = index

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        initializer1 = torch.nn.init.xavier_normal_
        gain1 = 2
        gain = 1
        # gain = 2
        std = 1
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size), gain=gain)
        self.W_R = initializer(torch.empty(self.n_relations, self.emb_size), gain=gain)
        self.user_like = initializer(torch.empty(self.num_like, self.emb_size), gain=gain)
        self.common_like = initializer(torch.empty(self.num_like, self.emb_size), gain=gain1)

        # self.all_embed = initializer1(torch.empty(self.n_nodes, self.emb_size), gain=gain)
        # self.W_R = initializer1(torch.empty(self.n_relations, self.emb_size), gain=gain)
        # self.user_like = initializer1(torch.empty(self.num_like, self.emb_size), gain=gain)
        # self.common_like = initializer1(torch.empty(self.num_like, self.emb_size), gain=gain)

    def _init_model(self):
        return GraphConv_MAKG(dim=self.emb_size,
                              n_hops=self.context_hops,
                              n_users=self.n_users,
                              n_items=self.n_items,
                              n_entities=self.n_entities,
                              n_relations=self.n_relations,
                              num_like=self.num_like)

    def forward(self, batch, graph, graph_UIS, graph_r):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS, graph_r, self.all_embed,
                                              self.W_R,
                                              self.common_like, self.user_like)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_UIS, graph_r):
        return self.gcn(graph, graph_UIS, graph_r, self.all_embed,
                        self.W_R,
                        self.common_like, self.user_like)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # loss1 = self._cul_cor()
        loss1 = self._cul_cor() + self._cul_cor1()

        loss11 = -torch.mean(
            torch.cosine_similarity(self.common_like.unsqueeze(1),
                                    self.all_embed[:self.n_users, :].unsqueeze(0), dim=2))
        loss12 = -torch.mean(
            torch.cosine_similarity(self.user_like.unsqueeze(1),
                                    self.W_R.unsqueeze(0), dim=2))
        loss1 = loss1 + loss11 + loss12

        loss2 = self.sim_decay * loss1 * 1000

        return mf_loss + emb_loss, mf_loss, loss1

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0)
            # return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.common_like.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.num_like):
                for j in range(i + 1, self.num_like):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.user_like[i], self.user_like[j])
                    else:
                        cor += CosineSimilarity(self.user_like[i], self.user_like[j])
        return cor

    def _cul_cor1(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0)
            # return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.common_like.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.num_like):
                for j in range(i + 1, self.num_like):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.common_like[i], self.common_like[j])
                    else:
                        cor += CosineSimilarity(self.common_like[i], self.common_like[j])
        return cor

    def _cul_cor_r(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.W_R.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_relations):
                for j in range(i + 1, self.n_relations):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.W_R[i], self.W_R[j])
                    else:
                        cor += CosineSimilarity(self.W_R[i], self.W_R[j])
            # cor = cor / (self.n_relations * (self.n_relations - 1) / 2)
        return cor


class RS_MAKG1(nn.Module):
    def __init__(self, data_config, args_config):
        super(RS_MAKG1, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.ind = args_config.ind
        self.temperature = 0.2

        self.num_like = args_config.num_like
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        # self.W_R_user_like = nn.Parameter(self.W_R_user_like)

        self.W_R = nn.Parameter(self.W_R)
        self.user_like = nn.Parameter(self.user_like)
        self.common_like = nn.Parameter(self.common_like)

        self.gcn = self._init_model()

        index = np.arange(self.n_relations)
        np.random.shuffle(index)
        self.index = index

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        initializer1 = torch.nn.init.xavier_normal_
        gain1 = 2
        gain = 1
        # gain = 2
        std = 1
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size), gain=gain)
        self.W_R = initializer(torch.empty(self.n_relations, self.emb_size), gain=gain)
        self.user_like = initializer(torch.empty(self.num_like, self.emb_size), gain=gain)
        self.common_like = initializer(torch.empty(self.num_like, self.emb_size), gain=gain1)

        # self.all_embed = initializer1(torch.empty(self.n_nodes, self.emb_size), gain=gain)
        # self.W_R = initializer1(torch.empty(self.n_relations, self.emb_size), gain=gain)
        # self.user_like = initializer1(torch.empty(self.num_like, self.emb_size), gain=gain)
        # self.common_like = initializer1(torch.empty(self.num_like, self.emb_size), gain=gain)

    def _init_model(self):
        return GraphConv_MAKG(dim=self.emb_size,
                              n_hops=self.context_hops,
                              n_users=self.n_users,
                              n_items=self.n_items,
                              n_entities=self.n_entities,
                              n_relations=self.n_relations,
                              num_like=self.num_like)

    def forward(self, batch, graph, graph_UIS, graph_r):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS, graph_r, self.all_embed,
                                              self.W_R,
                                              self.common_like, self.user_like)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_UIS, graph_r):
        return self.gcn(graph, graph_UIS, graph_r, self.all_embed,
                        self.W_R,
                        self.common_like, self.user_like)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # loss1 = self._cul_cor()
        loss1 = self._cul_cor() + self._cul_cor1()

        loss11 = -torch.mean(
            torch.cosine_similarity(self.common_like.unsqueeze(1),
                                    self.all_embed[:self.n_users, :].unsqueeze(0), dim=2))
        loss12 = -torch.mean(
            torch.cosine_similarity(self.user_like.unsqueeze(1),
                                    self.W_R.unsqueeze(0), dim=2))
        loss1 = loss1 + loss11 + loss12

        loss2 = self.sim_decay * loss1 * 1000

        return mf_loss + emb_loss, mf_loss, loss1

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0)
            # return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.common_like.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.num_like):
                for j in range(i + 1, self.num_like):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.user_like[i], self.user_like[j])
                    else:
                        cor += CosineSimilarity(self.user_like[i], self.user_like[j])
        return cor

    def _cul_cor1(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0)
            # return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.common_like.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.num_like):
                for j in range(i + 1, self.num_like):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.common_like[i], self.common_like[j])
                    else:
                        cor += CosineSimilarity(self.common_like[i], self.common_like[j])
        return cor

    def _cul_cor_r(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.W_R.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_relations):
                for j in range(i + 1, self.n_relations):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.W_R[i], self.W_R[j])
                    else:
                        cor += CosineSimilarity(self.W_R[i], self.W_R[j])
            # cor = cor / (self.n_relations * (self.n_relations - 1) / 2)
        return cor


class RS_UIFKAN(nn.Module):
    def __init__(self, data_config, args_config):
        super(RS_UIFKAN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.n_epoch = data_config['epoch_num']

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.ind = args_config.ind
        self.temperature = 0.2

        self.num_like = args_config.num_like
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        # self.dis_embed = nn.Parameter(self.dis_embed)

        # self.W_R = nn.Parameter(self.W_R)
        # self.user_like = nn.Parameter(self.user_like)
        # self.common_like = nn.Parameter(self.common_like)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        std = 0.1
        initializer1 = torch.nn.init.uniform_
        gain1 = 2
        gain = 1

        # self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size), gain=gain)
        self.all_embed = initializer(torch.empty(self.n_nodes + self.n_relations, self.emb_size), gain=gain)
        # self.dis_embed = initializer(torch.empty(self.n_users, self.emb_size * self.num_like), gain=gain)
        # self.all_embed = initializer1(torch.empty(self.n_nodes + self.n_relations, self.emb_size), b=1)

        # self.W_R = initializer(torch.empty(self.n_relations, self.emb_size), gain=gain)
        # self.user_like = initializer(torch.empty(self.num_like, self.emb_size), gain=gain)
        # self.common_like = initializer(torch.empty(self.num_like, self.emb_size), gain=gain1)

    def _init_model(self):
        return GraphConv_UIFGAN(dim=self.emb_size,
                                n_hops=self.context_hops,
                                n_users=self.n_users,
                                n_items=self.n_items,
                                n_entities=self.n_entities,
                                n_relations=self.n_relations,
                                num_like=self.num_like,
                                n_epoch=self.n_epoch)

    def forward(self, batch, graph, graph_UIS, graph_r):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        # item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS, graph_r, self.all_embed,
        #                                       self.W_R,
        #                                       self.common_like, self.user_like)
        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS, self.all_embed)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        # u_dis = F.normalize(self.dis_embed, dim=1)[user]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_UIS, graph_r):
        # return self.gcn(graph, graph_UIS, graph_r, self.all_embed,
        #                 self.W_R,
        #                 self.common_like, self.user_like)
        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS, self.all_embed)
        # user_gcn_emb = user_gcn_emb - 1 * F.normalize(self.dis_embed, dim=1)
        return item_gcn_emb, user_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # pos_scores1 = torch.sum(torch.mul(u_dis, pos_items), axis=1)
        # neg_scores1 = torch.sum(torch.mul(u_dis, neg_items), axis=1)
        # mf_loss1 = -1 * torch.mean(nn.LogSigmoid()(-pos_scores1 + neg_scores1))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2

        # regularizer = (torch.norm(users) ** 2
        #                + torch.norm(pos_items) ** 2
        #                + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        # loss1 = self._cul_cor() + self._cul_cor1()
        # loss11 = -torch.mean(
        #     torch.cosine_similarity(self.common_like.unsqueeze(1),
        #                             self.all_embed[:self.n_users, :].unsqueeze(0), dim=2))
        # loss12 = -torch.mean(
        #     torch.cosine_similarity(self.user_like.unsqueeze(1),
        #                             self.W_R.unsqueeze(0), dim=2))
        # loss1 = loss1 + loss11 + loss12
        # loss2 = self.sim_decay * loss1 * 1000

        return mf_loss + emb_loss, mf_loss, emb_loss

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0)
            # return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.common_like.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.num_like):
                for j in range(i + 1, self.num_like):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.user_like[i], self.user_like[j])
                    else:
                        cor += CosineSimilarity(self.user_like[i], self.user_like[j])
        return cor

    def _cul_cor1(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0)
            # return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.common_like.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.num_like):
                for j in range(i + 1, self.num_like):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.common_like[i], self.common_like[j])
                    else:
                        cor += CosineSimilarity(self.common_like[i], self.common_like[j])
        return cor

    def _cul_cor_r(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                     torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.W_R.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_relations):
                for j in range(i + 1, self.n_relations):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.W_R[i], self.W_R[j])
                    else:
                        cor += CosineSimilarity(self.W_R[i], self.W_R[j])
            # cor = cor / (self.n_relations * (self.n_relations - 1) / 2)
        return cor


class RS_UPKGAN(nn.Module):
    def __init__(self, data_config, args_config):
        super(RS_UPKGAN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.num_like = args_config.num_like
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.entity_embed = nn.Parameter(self.entity_embed)
        self.W_R = nn.Parameter(self.W_R)
        self.common_like = nn.Parameter(self.common_like)
        self.user_like = nn.Parameter(self.user_like)

        self.gcn = self._init_model()

    def _init_weight(self):
        # initializer1 = nn.init.xavier_uniform_

        initializer = nn.init.normal_
        std = 0.1
        self.entity_embed = initializer(torch.empty(self.n_entities, self.emb_size), std=std)
        self.W_R = initializer(torch.empty(self.n_relations, self.emb_size), std=std)

        self.common_like = initializer(torch.empty(self.num_like, self.emb_size), std=std)
        self.user_like = initializer(torch.empty(self.n_users, self.emb_size), std=std)

    def _init_model(self):
        return GraphConv_UPKGAN_kmean(dim=self.emb_size,
                                      n_hops=self.context_hops,
                                      n_users=self.n_users,
                                      n_items=self.n_items,
                                      n_entities=self.n_entities,
                                      n_relations=self.n_relations,
                                      num_like=self.num_like)

    def forward(self, batch, graph, graph_UIS):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS, self.entity_embed, self.W_R,
                                              self.user_like)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_UIS):
        return self.gcn(graph, graph_UIS,
                        self.entity_embed, self.W_R,
                        self.user_like)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        #
        # loss1 = 0
        # for i in range(self.num_like - 1):
        #     for j in range(i + 1, self.num_like):
        #         loss1 = loss1 + torch.cosine_similarity(self.common_like[i], self.common_like[j], dim=0)
        # loss1 = self.sim_decay * loss1 / (self.num_like * (self.num_like - 1) / 2)

        return mf_loss + emb_loss, mf_loss, emb_loss
        # return mf_loss + emb_loss + loss1 + loss2, mf_loss, loss1 + loss2

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]  # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.all_embed(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.all_embed(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.all_embed(neg_t)  # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


class RS_UIKGAN(nn.Module):
    def __init__(self, data_config, args_config):
        super(RS_UIKGAN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.lr = args_config.lr
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.num_like = args_config.num_like
        self.coe_list = args_config.coe_list
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self._init_weight()
        self.entity_embed = nn.Parameter(self.entity_embed)
        self.W_R = nn.Parameter(self.W_R)
        # self.common_like = nn.Parameter(self.common_like)
        self.user_like = nn.Parameter(self.user_like)

        self.gcn = self._init_model()

    def _init_weight(self):
        # initializer1 = nn.init.xavier_uniform_

        initializer = nn.init.normal_
        std = 0.1
        self.entity_embed = initializer(torch.empty(self.n_entities, self.emb_size), std=std)
        self.W_R = initializer(torch.empty(self.n_relations, self.emb_size), std=std)

        # self.common_like = initializer(torch.empty(self.num_like, self.emb_size), std=std)
        self.user_like = initializer(torch.empty(self.n_users, self.emb_size), std=std)

    def _init_model(self):
        return GraphConv_UIKGAN(dim=self.emb_size,
                                n_hops=self.context_hops,
                                n_users=self.n_users,
                                n_items=self.n_items,
                                n_entities=self.n_entities,
                                n_relations=self.n_relations,
                                num_like=self.num_like)

    def forward(self, batch, graph, graph_UIS):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        item_gcn_emb, user_gcn_emb = self.gcn(graph, graph_UIS, self.entity_embed, self.W_R, self.user_like,
                                              self.coe_list)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_UIS):
        return self.gcn(graph, graph_UIS,
                        self.entity_embed, self.W_R, self.user_like, self.coe_list)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        #
        # loss1 = 0
        # for i in range(self.num_like - 1):
        #     for j in range(i + 1, self.num_like):
        #         loss1 = loss1 + torch.cosine_similarity(self.common_like[i], self.common_like[j], dim=0)
        # loss1 = self.sim_decay * loss1 / (self.num_like * (self.num_like - 1) / 2)

        return mf_loss + emb_loss, mf_loss, emb_loss
        # return mf_loss + emb_loss + loss1 + loss2, mf_loss, loss1 + loss2

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]  # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.all_embed(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.all_embed(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.all_embed(neg_t)  # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class RS_KGUI3_sub(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(RS_KGUI3_sub, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = adj_mat
        self.edge_index, self.edge_type = 1, 1
        self.edge_index_user, self.edge_type_user = 1, 1

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()
        self.user_idx = torch.arange(self.n_users)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv_KGUI3_sub(dim=self.emb_size,
                                   n_hops=self.context_hops,
                                   n_users=self.n_users,
                                   n_items=self.n_items,
                                   n_entities=self.n_entities,
                                   n_pre=self.n_pre,
                                   n_relations=self.n_relations,
                                   interact_mat=self.interact_mat,
                                   ind=self.ind,
                                   node_dropout_rate=self.node_dropout_rate,
                                   mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, graph_i2u, graph_u2u, user_dict):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb = self.gcn(user, graph, graph_i2u, graph_u2u, user_dict,
                                                user_emb, item_emb,
                                                self.latent_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.edge_index_user,
                                                self.edge_type_user,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, graph_i2u, graph_u2u, user_dict):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(self.user_idx, graph, graph_i2u, graph_u2u, user_dict,
                        user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.edge_index_user,
                        self.edge_type_user,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss, mf_loss, emb_loss


class RS_KGUI4(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(RS_KGUI4, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pre = data_config['num_pre']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = adj_mat
        self.edge_index, self.edge_type = 1, 1
        self.edge_index_user, self.edge_type_user = 1, 1

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv_KGUI4(dim=self.emb_size,
                               n_hops=self.context_hops,
                               n_users=self.n_users,
                               n_items=self.n_items,
                               n_entities=self.n_entities,
                               n_pre=self.n_pre,
                               n_relations=self.n_relations,
                               interact_mat=self.interact_mat,
                               ind=self.ind,
                               node_dropout_rate=self.node_dropout_rate,
                               mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch, graph, user_dict):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb = self.gcn(graph, user_dict,
                                                user_emb, item_emb,
                                                self.latent_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.edge_index_user,
                                                self.edge_type_user,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def generate(self, graph, user_dict):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(graph, user_dict,
                        user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.edge_index_user,
                        self.edge_type_user,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss, mf_loss, emb_loss
