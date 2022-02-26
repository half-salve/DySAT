# -*- encoding: utf-8 -*-
import dgl
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.functional import edge_softmax

import copy

def scatter(feat,index,idm):
    num_node=max(index) + 1
    feat_sum=torch.zeros((num_node,feat.shape[1],feat.shape[2]))
    for i in range(num_node):
        f=torch.sum(feat[torch.nonzero(index==i)],dim=idm)
        # print(f.shape)
        feat_sum[i]=f
    return feat_sum


class StructuralAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads,
                 attn_drop,
                 ffd_drop,
                 residual):
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads #取整除
        self.n_heads = n_heads
        self.act = nn.ELU()

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)  # 创建一个全连接层
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)  # nn.Dropout将Dropout就是在不同的训练过程中随机扔掉一部分神经元。
        # 也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
        # 目的为了防止或减轻过拟合
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

    def forward(self,x):
        # print("struvtual")
        g=x[0]
        weight=x[1]
        graph = copy.deepcopy(g)
        # print(graph.device)
        feature=graph.ndata['feat']
        # print("feature.shape:",feature.size())
        row = weight.row
        col = weight.col
        edge_index=torch.tensor(numpy.vstack((row,col)).T,dtype=torch.long).to(graph.device)
        edge_weight = torch.tensor(weight.data).to(graph.device).view(-1,1)

        H, C = self.n_heads, self.out_dim
        x = self.lin(feature).view(-1, H, C)  # [N, heads, out_dim]堪称绝佳
        # attention
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze()  # [N, heads]将向量中维度为1的维度去掉[2,1]->[2]点积完再加和
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()
        #为每一个节点计算了a_l，a_r的值

        alpha_l = alpha_l[edge_index[:,0]]  # [num_edges, heads]
        alpha_r = alpha_r[edge_index[:,1]]

        #需要将[u，v]的对应节点相加计算，a_l的第一维index是节点，相加是每个head都相加
        alpha = alpha_r + alpha_l  # 对应层相加

        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)
        # print("alpha.device:",alpha.device)
        graph_softmax=dgl.graph((edge_index[:,0],edge_index[:,1])).to(graph.device)
        coefficients = edge_softmax(graph_softmax,alpha)  # [num_edges, heads]
        # print(coefficients.shape)
        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)

        #计算完WX的X
        # print("coefficients.device",coefficients.device)
        x_j = x[edge_index[:,0]]  # [num_edges, heads, out_dim]

        # output
        # print((x_j*coefficients[:, :, None]).shape)
        feat = scatter(x_j * coefficients[:, :, None], edge_index[:,1], idm=0).to(graph.device)
        # print("feat.device:",feat.device)
        out = self.act(feat)
        out = out.reshape(-1, self.n_heads * self.out_dim)  # [num_nodes, output_dim]
        # print(out.size())
        # print(out.device)
        if self.residual:
            out = out + self.lin_residual(feature)
        graph.ndata['feat'] = out

        return graph


class TemporalAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 num_time_steps,
                 attn_drop,
                 residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs]  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs  # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
