# -*- coding: UTF-8 -*-
import os
import sys

import pandas
import numpy
import torch
import dgl
import networkx
import scipy
import pickle as pkl

from utils.preprocess import pre_features

#重新安装networkx
def data_process(raw_dir, processed_dir,  args):
    num_class=0
    if args.dataset=='Enron_new':
        Enron_dataset = EnronDataset(raw_dir=raw_dir, processed_dir=processed_dir,args=args)
        g , src_dst_time_path= Enron_dataset.process()

    else:
        print("Dataset does not exist!")
        sys.exit(0)

    src_dst_time = torch.IntTensor(numpy.load(os.path.join(src_dst_time_path)))
    edge_mask_by_time = []
    start_time = int(torch.min(src_dst_time[:, -1]))
    print("start_time:",start_time)
    end_time = int(torch.max(src_dst_time[:, -1]))
    # 设定开始的时间切片，结束的时间切片
    # 按边上的时间戳对图进行切片
    print("row:",src_dst_time.shape[0])
    for i in range(start_time, end_time + 1):
        # for j in range(src_dst_time.shape[0]):
        mask=src_dst_time[:, -1] == i
        edge_mask=list(numpy.where(numpy.array(mask)==True)[0])
        edge_mask_by_time.append(edge_mask)
    print("prepare slicing finished")
    return g , edge_mask_by_time ,num_class,end_time - start_time + 1

def DySAT_rawdata(Path):
    with open(Path, "rb") as f:
        graphs = pkl.load(f)
    print("Loaded {} graphs ".format(len(graphs)))

    src_dst = []
    del_index=[]
    for e in graphs[0].edges():
        e = numpy.array(e)
        src_dst.append(e)
    src_dst = numpy.array(src_dst)
    dst_src = numpy.vstack((src_dst[:, 1], src_dst[:, 0])).T
    for i in range(dst_src.shape[0]):
        if dst_src[i][0] == dst_src[i][1]:
            del_index.append(i)
    dst_src = numpy.delete(dst_src, del_index,0)

    src_dst = numpy.vstack((src_dst, dst_src))
    step = numpy.full((src_dst.shape[0], 1), 0)
    src_dst_time = numpy.hstack((src_dst, step))

    for t in range(1, len(graphs)):
        src_dst = []
        del_index=[]
        for e in graphs[t].edges():
            e = numpy.array(e)
            src_dst.append(e)
        src_dst = numpy.array(src_dst)
        dst_src = numpy.vstack((src_dst[:, 1], src_dst[:, 0])).T
        for i in range(dst_src.shape[0]):
            if dst_src[i][0]==dst_src[i][1]:
                del_index.append(i)

        dst_src=numpy.delete(dst_src,del_index,0)
        src_dst = numpy.vstack((src_dst, dst_src))
        step = numpy.full((src_dst.shape[0], 1), t)
        src_dst_t = numpy.hstack((src_dst, step))
        src_dst_time = numpy.vstack((src_dst_time, src_dst_t))

    # adjs = [networkx.adjacency_matrix(g) for g in graphs]
    # weight = [normalize_graph_gcn(a) for a in adjs]

    edge_id=numpy.arange(src_dst_time.shape[0])[:,numpy.newaxis]
    src_dst_time = numpy.hstack((edge_id,src_dst_time))

    max_nodeid=max(src_dst_time[:,1].max(),src_dst_time[:,2].max())

    feats = scipy.sparse.identity( max_nodeid + 1 ).tocsr()[range(0, max_nodeid + 1), :]
    #因为序号最大到max_nodeid从0开始编号，所以有max_nodeid+1个点
    #可能出问题。！！！！！！！！

    # id_features = position_encoding(0, graphs[0].shape[0], 64)

    features = pre_features(feats)
    # print(features.shape[0])

    id=numpy.arange(max_nodeid + 1)[:,numpy.newaxis]
    # print(id.shape[0])
    assert id.shape[0]==features.shape[0]
    id_features=numpy.hstack((id,features))

    # print("src_dst_time:",src_dst_time.shape),print("id_features:",id_features)

    return src_dst_time,id_features

def graph_create_withoutlabel(processd_dir):
    id_features = torch.Tensor(numpy.load(os.path.join(processd_dir, 'id_features.npy')))
    # id_time_features：Id , time , feature
    # edge_info: edge , info
    src_dst_time = numpy.load(os.path.join(processd_dir, 'src_dst_time.npy'))
    # src_dst_time: srcId , dstId , edge_time
    src_dst_time = torch.IntTensor(src_dst_time)
    # src_dst_time: srcId , dstId , edge_time
    src = src_dst_time[:, 1]
    dst = src_dst_time[:, 2]
    # id_label[:, 0] is used to add self loop
    print(max(src.max(),dst.max()))
    g = dgl.graph(data=(src, dst),num_nodes=id_features.shape[0])
    g.edata['timestamp'] = src_dst_time[:, 3]

    features = id_features[:, 1:]

    g.ndata['feat'] = features
    # 用ndata添加节点特征
    # used to construct time-based sub-graph.
    print("Gragh processing is complete.")

    # dgl.save_graphs('./data/reddit/processed/reddit_{}.bin'.format(self.reverse_edge), [g])
    # else:
    #     print("Data is exist directly loaded.")
    #     gs, _ = dgl.load_graphs('./data/reddit/processed/reddit_{}.bin'.format(self.reverse_edge))
    #     g = gs[0]
    return g

def position_encoding(start , max_len , emb_size):
    pe=numpy.zeros((max_len,emb_size + 1))
    position=numpy.arange(1 , max_len + 1)[:,numpy.newaxis]
    print(position)
    div_term = numpy.exp(numpy.arange(0, emb_size, 2) * -(numpy.log(10000.0) / emb_size))[numpy.newaxis,:]
    print(div_term)
    print(numpy.multiply(position,div_term))
    pe[:,0]=numpy.arange(start , max_len + start  )
    pe[:, 1::2] = numpy.sin(position * div_term)
    pe[:, 2::2] = numpy.cos(position * div_term[0:int(emb_size / 2)])
    return pe

class EnronDataset:
    def __init__(self,raw_dir,processed_dir,args):
        self.args=args
        self.raw_dir=raw_dir
        self.processd_dir=processed_dir
    def process(self):
        if not os.path.exists('./data/{}/processed/{}.bin'.format(self.args.dataset,self.args.dataset)):
            self.process_raw_data()
            g=graph_create_withoutlabel(self.processd_dir)
            dgl.save_graphs('./data/{}/processed/{}.bin'.format(self.args.dataset,self.args.dataset), [g])
        else:
            print("Data is exist directly loaded.")
            gs, _ = dgl.load_graphs('./data/{}/processed/{}.bin'.format(self.args.dataset,self.args.dataset))
            g = gs[0]
        path = os.path.join(self.processd_dir, 'src_dst_time.npy')
            # print("stop！！！")
            # g,path=graph_create(processd_dir=self.processd_dir,interval=self.interval,reverse_edge=self.reverse_edge)
        return g , path
    def process_raw_data(self):
        id_features_path = os.path.join(self.processd_dir, 'id_features.npy')
        src_dst_time_path = os.path.join(self.processd_dir, 'src_dst_time.npy')
        # edge_info_path = os.path.join(self.processd_dir, 'edge_info.npy')
        if os.path.exists(id_features_path) and os.path.exists(src_dst_time_path):
            print("The preprocessed data already exists, skip the preprocess stage!")
            return

        src_dst_time,id_features\
            =DySAT_rawdata("{}/{}".format(self.raw_dir, "graph.pkl"))

        numpy.save(id_features_path, id_features)
        numpy.save(src_dst_time_path, src_dst_time)



