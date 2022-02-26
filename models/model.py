# -*- coding: UTF-8 -*-
import numpy
import pandas
import torch
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss

from models.layers import StructuralAttentionLayer, TemporalAttentionLayer
from utils.preprocess import graph_node_set


class DySAT(nn.Module):
    def __init__(self, args, num_features, time_length ,device):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DySAT, self).__init__()
        self.args = args
        self.device=device
        if args.window < 0:
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.
        self.num_features = num_features

        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop

        self.structural_attn, self.temporal_attn = self.build_model()

        self.bceloss = BCEWithLogitsLoss()

    def forward(self,x):

        # Structural Attention forward
        structural_out = []
        graphs=x[0]
        weight=x[1]
        # print(graphs[15].ndata['FID'])
        for t in range(0, self.num_time_steps):
            #num_time_step时间步
            structural_out.append(self.structural_attn([graphs[t],weight[t]]))
        #structural_out中是经过结构注意力的结果
        # structural_outputs = [g.ndata['feat'] for g in structural_out]  # list of [Ni, 1, F]
        structural_outputs = [g.ndata['feat'][:, None, :] for g in structural_out]

        # print("structural_learning finished。。。。。")
        # padding outputs along with Ni
        out_dim = structural_outputs[-1].shape[-1]

        #在原来得数据集上可以直接加0，但是在新数据集中需要重新排位置，这样最终出来的结果才是N,F
        orginal_id=numpy.array(graph_node_set(graphs))
        # new_id=numpy.arange(len(orginal_id))
        #
        # id_pd = []
        # for i in range(orginal_id.shape[0]):
        #     id_pd.append((orginal_id[i], new_id[i]))
        # id_dict = dict(id_pd)
        # self.id_dict = id_dict
        #
        # num_nodes=len(new_id)
        num_nodes=orginal_id.max()+1
        structural_outputs_padded = []

        for i in range(len(structural_outputs)):
            out=structural_outputs[i]
            # padding=torch.zeros((num_nodes,1,out_dim)).to(out.device)

            zero_padding = torch.zeros(num_nodes - out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
            # print("structural_outputs_padded:",padded.shape)

            orginal_idx=graphs[i].ndata['FID']

            # node_idx=orginal_idx.long()
            #建立起原节点id和现节点id的对应关系，将学习过的特征填入节点新节点id

            # structural_outputs_padded.append(padding)

            # 拼接在一起需要shape相同所以需要用0补全

        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]
        # Temporal Attention forward
        temporal_out = self.temporal_attn(structural_outputs_padded)

        return temporal_out

    def build_model(self):
        input_dim = self.num_features

        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        print("structural_head_config:", self.structural_head_config)
        print("structural_layer_config:", self.structural_layer_config)
        print("len(self.structural_layer_config):",len(self.structural_layer_config))
        print("temporal_head_config:", self.temporal_head_config)
        print("temporal_layer_config:", self.temporal_layer_config)
        print("len(self.temporal_layer_config):",len(self.temporal_layer_config))
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.args.residual)
            # 多层建立Structural Attention Layers,每层的Structural Attention Layers是多头的
            # Structural Attention Layers 造很多个 但是就一层一个并行层
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            #将structural_layer的创建的结果，加入structural_attention_layers
            input_dim = self.structural_layer_config[i]
            # 下一层structural的input是上一层的output
        # 2: Temporal Attention Layers
        #temporal的输入是structural的输出
        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.Sequential()
        # 将structural_layer的创建的结果，加入structural_attention_layers
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            #  Temporal Attention Layers造一个但是需要造很多层
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers

    def get_loss(self, feed_dict,graphs):
        node_1, node_2, node_2_negative,weight= feed_dict.values()
        # run gnn
        final_emb = self.forward([graphs,weight])  # [N, T, F]
        #graohs只用于训练
        self.graph_loss = 0

        for t in range(self.num_time_steps - 1):
            emb_t = final_emb[:, t, :].squeeze()  # [N, F]
            #本身编号为原始编号,就是将真实的编号于现在在特征矩阵的编号做一个对应，通过对应后的编号提取出对应节点的特征
            # new_node_1=torch.tensor(pandas.DataFrame(node_1[t].cpu().numpy())[0].map(self.id_dict)).to(self.device)
            # new_node_2=torch.tensor(pandas.DataFrame(node_2[t].cpu().numpy())[0].map(self.id_dict)).to(self.device)
            # new_node_2_negative=torch.tensor(pandas.DataFrame(node_2_negative[t].cpu().numpy())[0].map(self.id_dict)).to(self.device)
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]
            pos_score = torch.sum(source_node_emb * tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[:, None, :] * tart_node_neg_emb, dim=2).flatten()
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))
            graphloss = pos_loss + self.args.neg_weight * neg_loss
            self.graph_loss += graphloss
        return self.graph_loss






