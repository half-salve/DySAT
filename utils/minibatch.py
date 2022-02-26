# -*- coding: UTF-8 -*-
import copy

import torch
import numpy
import scipy
from torch.utils.data import Dataset
from utils.preprocess import graph_node_set


def fixed_unigram_candidate_sampler(true_clasees, num_true, num_sampled,
                                    orginal_id, unigrams):
    # TODO: implementate distortion to unigrams
    # assert true_clasees.shape[1] == num_true
    samples = []
    # print(unigrams)
    for i in range(true_clasees.shape[0]):
        dist = copy.deepcopy(unigrams)
        candidate = copy.deepcopy(orginal_id.numpy().tolist())
        #orginal_id的顺序和度的顺序是一致的
        taboo = true_clasees[i].numpy()[0]#正采样的节点，需要在节点中将正采样的节点去除
        # print(taboo)
        p = candidate.index(taboo)
        candidate.remove(taboo)
        dist.pop(p)
    #pop去除指定位置的值，获取点在度函数中的位置
    # 从图中去掉被采样的点，然后进行负采样
        sample = numpy.random.choice(candidate, size=num_sampled, replace=False, p=dist / numpy.sum(dist))
        # 输入的特定时刻的点集，对于每个点采样num_sampled次
        samples.append(sample)

    # count=0
    # dist = copy.deepcopy(unigrams)
    # candidate = copy.deepcopy(orginal_id.numpy().tolist())
    # true_clasees=numpy.unique(true_clasees.numpy())
    # # print(true_clasees)
    # for i in range(true_clasees.shape[0]):
    #     #orginal_id的顺序和度的顺序是一致的
    #     taboo = true_clasees[i]#正采样的节点，需要在节点中将正采样的节点去除
    #     # print(taboo)
    #     p = candidate.index(taboo)
    #     candidate.remove(taboo)
    #     dist.pop(p)
    # #pop去除指定位置的值，获取点在度函数中的位置
    # # 从图中去掉被采样的点，然后进行负采样
    # for idx in range(len(dist)):
    #     if dist[idx]!=0:
    #         count=count+1
    # if count>=num_sampled:
    #
    #     samples = numpy.random.choice(candidate, size=num_sampled, replace=False, p=dist / numpy.sum(dist))
    # else:
    #     samples = numpy.random.choice(candidate, size=num_sampled, replace=True, p=dist / numpy.sum(dist))
    # 输入的特定时刻的点集，对于每个点采样num_sampled次
    # samples.append(sample)
    # print("samples:\n",samples)

    return samples

class MyDataset(Dataset):
    def __init__(self, context_pairs,graphs,args):
        super(MyDataset, self).__init__()
        self.context_pairs = context_pairs
        nodes_set=graph_node_set(graphs)
        self.train_nodes = nodes_set # all nodes in the graph.
        self.weights=[self.normalize_graph_gcn(g.adj(scipy_fmt='coo'),g) for g in graphs]
        # self.weights=self.create_weight(w)
        self.degs=self.degree(graphs)
        self.args=args
        self.data_items=self.negative_sample_pairs(context_pairs,graphs)

    # def create_weight(self,w):
    #     weights = []
    #     for weight in w:
    #         row = weight.row
    #         col = weight.col
    #         edge_index = numpy.vstack((row, col)).T
    #         edge_weight = weight.data[:,numpy.newaxis]
    #         index_weight=numpy.vstack((edge_index,edge_weight))
    #         weights.append(index_weight)
    #     weights=torch.tensor(weights)
    #     return weights

    def degree(self,graphs):
        deg_t=[]
        for t in range(len(graphs)):
            G=graphs[t]
            deg_t.append(G.out_degrees().numpy().tolist())
        return deg_t

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, index):
        node = self.train_nodes[index]
        return self.data_items[node]

    def normalize_graph_gcn(self,adj,g):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        orginal_id=g.ndata["FID"].numpy()
        new_id=g.nodes().numpy()
        id_pd = []
        for i in range(orginal_id.shape[0]):
            id_pd.append((new_id[i], orginal_id[i]))
        id_dict = dict(id_pd)

        adj = scipy.sparse.coo_matrix(adj, dtype=numpy.float32)
        adj_ = adj + scipy.sparse.eye(adj.shape[0], dtype=numpy.float32)

        row = adj_.tocoo().row
        col = adj_.tocoo().col
        edge_weight = adj_.tocoo().data
        row=numpy.array([id_dict[m] for m in row])
        col = numpy.array([id_dict[m] for m in col])
        shit=scipy.sparse.coo_matrix((edge_weight,(row,col)),dtype=numpy.float32)
        numpy.save("./model_checkpoints/adj_{}.npy".format(g.num_nodes()),shit.toarray())

        rowsum = numpy.array(adj_.sum(1), dtype=numpy.float32)
        degree_mat_inv_sqrt = scipy.sparse.diags(numpy.power(rowsum, -0.5).flatten(), dtype=numpy.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

        row = adj_normalized.row
        col = adj_normalized.col
        edge_weight = adj_.data

        row=numpy.array([id_dict[m] for m in row])
        col = numpy.array([id_dict[m] for m in col])
        coo=scipy.sparse.coo_matrix((edge_weight,(row,col)),dtype=numpy.float32)
        return adj_normalized

    def negative_sample_pairs(self,context_pairs, graphs):
        # context_pairs是全局的采样结果，
        data_items = {}
        time_steps = len(graphs)
        nodes_set = graph_node_set(graphs)

        for node in nodes_set:
            feed_dict = {}
            node_1_all_time = []
            node_2_all_time = []
            for t in range(time_steps):
                node_1 = []
                node_2 = []
                if len(context_pairs[t][node]) > self.args.neg_sample_size:  # 是否存在在某一时刻某节点不存在的问题
                    node_1.extend([node] * self.args.neg_sample_size)
                    node_2.extend(numpy.random.choice(context_pairs[t][node], self.args.neg_sample_size, replace=False))
                else:
                    node_1.extend([node] * len(context_pairs[t][node]))
                    node_2.extend(context_pairs[t][node])
                assert len(node_1) == len(node_2)
                node_1_all_time.append(node_1)  # 每个时刻的某个节点的序列
                node_2_all_time.append(node_2)
            # node_1_all_time的总大小应该等于，,max_positive * time_num
            node_1_list = [torch.LongTensor(node) for node in node_1_all_time]
            node_2_list = [torch.LongTensor(node) for node in node_2_all_time]
            # 特定节点在所有时刻的随机游走的和
            node_2_negative = []
            for t in range(len(node_2_list)):
                degree = self.degs[t]
                # t时刻节点的所有度
                node_positive = node_2_list[t][:, None]
                # 对每个时刻进行采样
                # if node==0:
                    # a,sorted_index=torch.sort(graphs[t].ndata['FID'])
                    # # print(sorted_index)
                    # print("degree:\n",numpy.array(degree)[sorted_index.numpy()])
                    # print("node_2_list[t]:\n", node_2_list[t])

                node_negative = fixed_unigram_candidate_sampler(true_clasees=node_positive,
                                                                num_true=1,
                                                                num_sampled=self.args.neg_sample_size,
                                                                orginal_id=graphs[t].ndata['FID'],
                                                                unigrams=degree)
                # 他这里必须按照度的函数采样所以无法直接使用dgl的负采样
                node_2_negative.append(node_negative)
                # if node == 0:
                #     print("node_2_negative[t]:\n", torch.LongTensor(node_negative))
            node_2_neg_list = [torch.LongTensor(node) for node in node_2_negative]
            feed_dict['node_1'] = node_1_list
            feed_dict['node_2'] = node_2_list
            feed_dict['node_2_neg'] = node_2_neg_list
            feed_dict["graphs"] = self.weights

            data_items[node] = feed_dict
            #节点0：时刻0    采样结果
            #      时刻1    采样结果
            #      时刻2    采样结果
        return data_items


    @staticmethod
    def collate_fn(samples):
        batch_dict = {}
        for key in ["node_1", "node_2", "node_2_neg"]:
            data_list = []
            for sample in samples:
                data_list.append(sample[key])
            concate = []
            for t in range(len(data_list[0])):#这个t是时刻
                concate.append(torch.cat([data[t] for data in data_list]))
            # 索引为某一特定时刻的，samples中节点的序列拼接在一起 类似于 000000 1111111
            batch_dict[key] = concate
            # 若干节点的序列拼接在一起 类似于 000000 1111111
        batch_dict["graphs"] = samples[0]["graphs"]

        return batch_dict

