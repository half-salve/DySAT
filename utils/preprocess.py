# -*- coding: UTF-8 -*-
import random
import copy

import dgl
import numpy
import scipy
import torch
import pandas

from collections import defaultdict
from sklearn.model_selection import train_test_split

def add_nodes(graphs_t,features):
    graphs=copy.deepcopy(graphs_t)
    cashed=[]
    for g in graphs:
        g_nodes=g.ndata['FID'].numpy().tolist()
        max_nodes=max(g_nodes)
        new_nodes=numpy.arange(max_nodes + 1 )###这里可能存在超越问题
        create_nodes=[]
        count=0
        for i in new_nodes:
            if i not in g_nodes:
                create_nodes.append(i)
                count=count+1
        if count>0:
            feat_g=g.ndata["feat"]
            create_feat=features[create_nodes]

            create_nodes=torch.tensor(create_nodes,dtype=g.ndata['FID'].dtype)

            g = dgl.add_nodes(g, count,{"feat":create_feat,"FID":create_nodes})
            # print(g.ndata["FID"])
            # print(g.number_of_nodes())
        cashed.append(g)
    return cashed

def graph_node_set(graphs):
    nodes=[]
    for g in graphs:
        # print(g)
        nodes.extend(g.ndata['FID'].cpu().numpy().tolist())
    nodes_set=list(set(nodes))
    return nodes_set

def random_walk(graphs,num_walks, args):
    print("Computing training pairs ...")
    random_walk_all=[]
    WINDOW_SIZE=10
    for g in graphs:
        # print(g)
        walks=[]
        pairs = defaultdict(list)
        pairs_cnt = 0

        orginal_id = g.ndata['FID'].numpy()
        new_id = g.nodes().numpy()
        # print(orginal_id.shape,new_id.shape)

        id_pd = []
        for i in range(orginal_id.shape[0]):
            id_pd.append((new_id[i], orginal_id[i]))
        id_dict = dict(id_pd)

        for i in range(0,num_walks):
            nodes=copy.deepcopy(g.nodes())
            random.shuffle(nodes)
            walk_t=dgl.sampling.random_walk(g, nodes, length=args.walk_len)[0]
            # walk_t=dgl.sampling.node2vec_random_walk(g,nodes, 1, 1, args.walk_len)
            # walk_t = dgl.sampling.node2vec_random_walk(g, nodes, 1, 1, 5)
            for i in range(walk_t.shape[0]):
                w=[id_dict[m] for m in walk_t[i].numpy()]
                walk_t[i]=torch.tensor(w)
                walks.append(w)
        # print(walks)

        for walk in walks:
            for word_index, word in enumerate(walk):
                for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE,
                                                                          len(walk)) + 1]:  # window_size倾向于一个视野。

                    if (nb_word != word):
                        pairs[word].append(nb_word)
                        pairs_cnt += 1

        random_walk_all.append(pairs)
        print("# nodes with random walk samples: {}".format(len(pairs)))
        print("# sampled pairs: {}".format(pairs_cnt))

    return random_walk_all


def pre_features(features):
    features = numpy.array(features.todense())
    rowsum = numpy.array(features.sum(1))
    r_inv = numpy.power(rowsum, -1).flatten()
    r_inv[numpy.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def to_device(batch, device):
    feed_dict = copy.deepcopy(batch)
    node_1, node_2, node_2_negative, graphs = feed_dict.values()
    # to device
    feed_dict["node_1"] = [x.to(device) for x in node_1]
    feed_dict["node_2"] = [x.to(device) for x in node_2]
    feed_dict["node_2_neg"] = [x.to(device) for x in node_2_negative]
    # feed_dict["graphs"] = [g.to(device) for g in graphs]
    return feed_dict

def evaluation_data(eval_graph,test_graph):
    """ Load train/val/test examples to evaluate link prediction performance"""
    print("Generating eval data ....")
    graph_former=copy.deepcopy(eval_graph)
    graph_later=copy.deepcopy(test_graph)

    old_later_nodes=graph_later.ndata["FID"].numpy()
    old_former_nodes=graph_former.ndata["FID"].numpy()
    new_later_nodes=graph_later.nodes().numpy()

    #将later中的边转化为numpy
    src = graph_later.edges()[0].numpy()[:, numpy.newaxis]
    dst = graph_later.edges()[1].numpy()[:, numpy.newaxis]
    edges_test = numpy.unique(numpy.hstack((src, dst)), axis=0)

    #建立原始编号到分割后编号的映射
    later_pd=[]
    for i in range(old_later_nodes.shape[0]):
        later_pd.append((new_later_nodes[i],old_later_nodes[i]))
    later_dict=dict(later_pd)

    #将graph_former中的边映射为原始编号
    edges_test[:, 0] = numpy.array([later_dict[t] for t in edges_test[:,0]])
    edges_test[:, 1] = numpy.array([later_dict[t] for t in edges_test[:,1]])
    # print("old_later_nodes:",old_later_nodes)
    # print("old_former_nodes:",old_former_nodes)
    edges_positive = []  # Constraint to restrict new links to existing nodes.
    for e in edges_test:
        if e[0] in old_former_nodes and e[1] in old_former_nodes :
            e=e.tolist()
            if [e[1],e[0]] not in edges_positive:
                edges_positive.append(e)
    edges_positive = numpy.array(edges_positive)  # [E, 2]
    numpy.save("./model_checkpoints/edges_positive.npy",edges_positive)
    # print(edges_positive.shape)
    # for e in edges_positive:
    #     print(e)
    # print(edges_positive)
    # print(edges_positive.shape)
    # 对创造的图进行负采样

    edges_negative = negative_sample(len(edges_positive), test_graph,edges_test)
    #稍有争议但是问题不大
    # 对新图和采样结果做划分，但是这是单个的图
    val_mask_fraction = 0.2
    test_mask_fraction = 0.6
    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive,edges_negative,
                                                                            test_size=val_mask_fraction + test_mask_fraction)

    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos,test_neg,
                                                                                    test_size=test_mask_fraction / (
                                                                                                test_mask_fraction + val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg

def negative_sample(pos_len,g,edges_test):
    u, v = g.edges()
    nodes_num=g.number_of_nodes()
    edges_test=edges_test.tolist()

    #将g中的边映射为原始编号

    edges_neg = []
    while len(edges_neg) < pos_len:
        idx_i = numpy.random.randint(0, nodes_num)
        idx_j = numpy.random.randint(0, nodes_num)
        if idx_i == idx_j:
            continue
        if [idx_i, idx_j] in edges_test or [idx_j, idx_i] in edges_test:
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])

    return edges_neg

def inductive_graph(graph_former_t, graph_later_t):
    graph_former=copy.deepcopy(graph_former_t)
    graph_later=copy.deepcopy(graph_later_t)
    graph_later=dgl.remove_edges(graph_later,torch.arange(graph_later.num_edges(),dtype=torch.int32))
    #去掉后者所有的边
    old_later_nodes=graph_later.ndata["FID"].numpy()
    old_former_nodes=graph_former.ndata["FID"].numpy()
    new_later_nodes=graph_later.nodes().numpy()
    new_former_nodes=graph_former.nodes().numpy()
    #将former中的边转化为numpy

    src = graph_former.edges()[0].numpy()[:, numpy.newaxis]
    dst = graph_former.edges()[1].numpy()[:, numpy.newaxis]
    edges = numpy.hstack((src, dst))
    # print(edges)
    #建立映射former的分割后编号和原始编号的映射
    former_pd=[]
    for i in range(new_former_nodes.shape[0]):
        former_pd.append((new_former_nodes[i],old_former_nodes[i]))
    former_dict=dict(former_pd)
    # print("former_dict:",former_dict)
    # print(former_dict)
    #建立原始编号到分割后编号的映射
    later_pd=[]
    for i in range(old_later_nodes.shape[0]):
        later_pd.append((old_later_nodes[i],new_later_nodes[i]))
    later_dict=dict(later_pd)

    #将graph_former中的边映射为原始编号
    edges[:, 0] = numpy.array([former_dict[t] for t in edges[:,0]])
    edges[:, 1] = numpy.array([former_dict[t] for t in edges[:,1]])

    #得到来自于前一个图的边，且边的节点在后一个图中存在
    edge_later=[]
    for e in edges:
        if e[0] in old_later_nodes and e[1] in old_later_nodes:
            edge_later.append(e)
    # print(edge_later)
    #将得到的边，节点编号映射回来
    edge_later=numpy.array(edge_later)
    edge_later_new=copy.deepcopy(edge_later)
    edge_later_new[:,0]=numpy.array([later_dict[t] for t in edge_later[:,0]])
    edge_later_new[:,1]=numpy.array([later_dict[t] for t in edge_later[:,1]])
    # print(later_dict)
    #将numpy转化为tensor
    # print(edge_later_new)
    # print("edges:\n",edges)
    # print("edge_later:\n",edge_later)
    graph_later= dgl.add_edges(graph_later, edge_later_new[:,0], edge_later_new[:,1])

    # print(graph_later)
    return graph_later








