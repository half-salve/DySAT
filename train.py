# -*- coding: UTF-8 -*-
import argparse
import numpy
import time
import os

import dgl
import torch
from tqdm import tqdm


from models.model import DySAT
from utils.details import plt_loss,datasave
from data_preprocess import data_process
from utils.preprocess import random_walk,evaluation_data,inductive_graph,to_device,add_nodes
from torch.utils.data import DataLoader
from utils.minibatch import MyDataset
from utils.link_preHAD import evaluate_classifier


def train(args, device):
    raw_dir='./data/{}/{}_dataset/'.format(args.dataset, args.dataset)
    processed_dir='./data/{}/processed/'.format(args.dataset)
    if not (os.path.isdir(processed_dir)):
        os.makedirs(processed_dir)

    g , edge_mask_by_time , num_classes ,interval \
        = data_process(raw_dir=raw_dir,processed_dir=processed_dir,args=args)

    #使用dgl.node_subgraph会对选取后的节点重新编号
    print("G:",g)
    print("#start random walk......")
    features=g.ndata["feat"]
    cached_subgraph = []
    cached_labeled_edge_mask = []
    #使用dgl.node_subgraph会对选取后的节点重新编号
    for i in range(len(edge_mask_by_time)):
        # we add self loop edge when we construct full graph, not here
        edge_subgraph=dgl.edge_subgraph(g,edge_mask_by_time[i])
        edge_subgraph.ndata['FID']=edge_subgraph.ndata[dgl.NID]
        # 根据切片提取子图
        cached_subgraph.append(edge_subgraph)
    print("#finished graph divide")

    context_pairs_train = random_walk(cached_subgraph, num_walks=10,args=args)#对全局进行采样
    # Load evaluation data for link prediction.
    cached_subgraph=add_nodes(cached_subgraph,features)
    # print(cached_subgraph[15])
    eval_graph=cached_subgraph[-2]
    test_graph=cached_subgraph[-1]
    # print("test_graph.ndata:",test_graph.ndata["FID"])
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
        test_edges_pos, test_edges_neg = evaluation_data(eval_graph,test_graph)

    print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
        len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
        len(test_edges_pos), len(test_edges_neg)))

    new_G = inductive_graph(cached_subgraph[args.time_step - 2], cached_subgraph[args.time_step-1])
    cached_subgraph[-1]=new_G

#从此只能用ndata['FID']
    # 在负采样时需要用到degs,且源自于未被标准化的图
    node_dataset = MyDataset(context_pairs=context_pairs_train,graphs=cached_subgraph,args=args)

    print("start training.......\n########################", )

    dataloader = DataLoader(node_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            # num_workers=10,
                            collate_fn=MyDataset.collate_fn)

    print("#start initializing the model.....")
    #模型本身已经定死了 每次加入新的图可能都需要重新训练
    model = DySAT(args, g.ndata['feat'].shape[1], args.time_step,device)#这里可以自主设定timestep的值
    # if torch.cuda.device_count() > 1:
    # print("Initialize multiple GPUs.....")
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     device_sequence = [int(w) for w in args.device_ids.split(',')]
    #     device_ids = [i for i in range(device_sequence[0], device_sequence[1] + 1)]
    #     model = torch.nn.DataParallel(model, device_ids, output_device=args.gpu)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    cached_subgraph=[g.to(device) for g in cached_subgraph]
    # in training
    best_epoch_val = 0
    patient = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = []
        #训练是只要将训练集训练完就可以
        for idx, feed_dict in enumerate(dataloader):
            feed_dict = to_device(feed_dict, device)
            opt.zero_grad()
            loss = model.get_loss(feed_dict,cached_subgraph)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())

        #到这一次训练完成了
        model.eval()
        emb = model([cached_subgraph,feed_dict["graphs"]])[:, -2, :].detach().cpu().numpy()
        val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                            train_edges_neg,
                                                            val_edges_pos,
                                                            val_edges_neg,
                                                            test_edges_pos,
                                                            test_edges_neg,
                                                            emb,
                                                            emb)
        epoch_auc_val = val_results["HAD"][1]
        epoch_auc_test = test_results["HAD"][1]
#237
        if epoch_auc_val > best_epoch_val:
            best_epoch_val = epoch_auc_val
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                break

        print("Epoch {:<3},  Loss = {:.3f}, Val AUC {:.3f} Test AUC {:.3f}".format(epoch,
                                                                numpy.mean(epoch_loss),
                                                                epoch_auc_val,
                                                                epoch_auc_test))
        print("##########################")
    # Test Best Model
    model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
    model.eval()
    emb = model([cached_subgraph,feed_dict["graphs"]])[:, -2, :].detach().cpu().numpy()
    val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                        train_edges_neg,
                                                        val_edges_pos,
                                                        val_edges_neg,
                                                        test_edges_pos,
                                                        test_edges_neg,
                                                        emb,
                                                        emb)
    # print("Best epoch val results {}\n".format(val_results))
    # print("Best epoch test results {}\n".format(test_results))
    #
    # output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (args.dataset.split("/")[0], str(today.year),
    #                                               str(today.month), str(today.day))
    # logging.info("Best epoch val results {}\n".format(val_results))
    # logging.info("Best epoch test results {}\n".format(test_results))
    #
    # write_to_csv(val_results, output_file, args.model, args.dataset, num_time_steps, mod='val')
    # write_to_csv(test_results, output_file, args.model, args.dataset, num_time_steps, mod='test')
    #
    # # Save final embeddings in the save directory.
    # emb = epochs_embeds[best_epoch]
    # np.savez(SAVE_DIR + '/{}_embs_{}_{}.npz'.format(FLAGS.model, FLAGS.dataset, FLAGS.time_steps - 2), data=emb)
    auc_val = val_results["HAD"][1]
    auc_test = test_results["HAD"][1]
    print("Best Test AUC = {:.3f}".format(auc_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DySAT")
    """NOTE: This script includes nearly all tf flag parameters as input arguments, which feed as input 
    through a generated config file."""

    # parser = argparse.ArgumentParser(description='Run script parameters')

    # Script specific parameters -> min and max time steps for executing different train files.
    # Time step range [min_time, max_time to train different models (both included).
    # Min time step is always 2 since we require at least one snapshot each for train and test.

    parser.add_argument('--time_step', type=int, nargs='?', default=16, help='time_step')

    # parser.add_argument('--max_time', type=int, nargs='?', default=15, help='max_time step')

    # NOTE: Ensure that the execution is split into different ranges so that GPU memory errors are avoided.
    # IncSAT must be executed sequentially
    print("dhajhdjahsdjashdjashdj")
    parser.add_argument('--run_parallel', type=str, nargs='?', default='False',
                        help='By default, sequential execution of different time steps (Note: IncSAT must be sequential)')

    # Necessary parameters for log creation.
    parser.add_argument('--base_model', type=str, nargs='?', default='DySAT',
                        help='Base model (DySAT/IncSAT)')

    # Additional model string to save different parameter variations.
    parser.add_argument('--model', type=str, nargs='?', default='default',
                        help='Additional model string')

    # Experimental settings.
    parser.add_argument("--dataset", type=str, nargs='?', default='Enron_new',
                        help="dataset selection wikipedia/reddit/Enron_new/email_uci/ml-10m/yelp")

    parser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training.")

    parser.add_argument('--device_ids', type=str, default='0,1',
                           help="device_ids for use Gpus")

    parser.add_argument('--num-epochs', type=int, nargs='?', default=1000,
                        help='# epochs')

    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')

    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')

    parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                        help='Batch size (# nodes)')

    # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
    parser.add_argument('--featureless', type=str, nargs='?', default='True',
                        help='True if one-hot encoding.')

    parser.add_argument('--max_gradient_norm', type=float, nargs='?', default=1.0,
                        help='Clip gradients to this norm')

    # Tunable hyper-params

    # TODO: Implementation has not been verified, performance may not be good.
    parser.add_argument('--residual', type=str, nargs='?', default='False',
                        help='Use residual')

    # Number of negative samples per positive pair.
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive')

    # Walk length for random walk sampling.
    parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                        help='Walk length for random walk sampling')

    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')

    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.001,
                        help='Initial learning rate for self-attention model.')

    parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')

    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')

    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument("--early_stop", type=int, default=10,
                        help="patient")
    # Architecture params

    parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                        help='Encoder layer config: # attention heads in each GAT layer')

    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each GAT layer')

    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')

    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each Temporal layer')

    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')

    parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')



    parser.add_argument('--interval', type=int, default='110',
                           help="The number of time slices you want to set")
    parser.add_argument('--reverse_edge', type=bool, default='True',
                           help="True for undirected graph False for directed graph")
    parser.add_argument('--scale', type=str, default='0.7,0.1',
                           help="The proportion of input train and valid")
    parser.add_argument('--patience', type=int, default=150,
                           help="Patience for early stopping.")
    args = parser.parse_args()

    # min_time = int(args.min_time)
    # max_time = int(args.max_time)



    #
    # argparser.add_argument('--num-epochs', type=int, default=1000)
    # argparser.add_argument('--n-hidden', type=int, default=256)
    # argparser.add_argument('--n-layers', type=int, default=2)
    # argparser.add_argument('--n-hist-steps', type=int, default=10,
    #                        help="If it is set to 5, it means in the first batch,"
    #                             "we use historical data of 0-4 to predict the data of time 5.")
    # argparser.add_argument('--lr', type=float, default=0.001)
    # argparser.add_argument('--loss-class-weight', type=str, default='0.25,0.75',
    #                        help='Weight for loss function. Follow the official code,'
    #                             'we need to change it to 0.25, 0.75 when use EvolveGCN-H')
    # argparser.add_argument('--eval-class-id', type=int, default=1,
    #                        help="Class type to eval. On Elliptic, type 1(illicit) is the main interest.")

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
        print("we can use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device('cpu')

    # print("use :",device)

    output_dir = "./data/"+"{}/{}_{}/".format(args.dataset,args.base_model, args.model)
    LOG_DIR = output_dir + "log"
    SAVE_DIR = output_dir + "output"
    CSV_DIR = output_dir + "csv"
    MODEL_DIR = output_dir + "model"

    if not (os.path.isdir(LOG_DIR) and os.path.isdir(SAVE_DIR)  and
        os.path.isdir(CSV_DIR) and os.path.isdir(MODEL_DIR) and os.path.isdir(output_dir)):
        os.makedirs(output_dir),os.mkdir(LOG_DIR),os.mkdir(SAVE_DIR),os.mkdir(CSV_DIR),os.mkdir(MODEL_DIR)

    with open(output_dir + '/flags_{}.txt'.format(args.dataset), 'w') as outfile:
        for k, v in vars(args).items():
            outfile.write("{}\t{}\n".format(k, v))

    # train_file = "train.py" if args.base_model == "DySAT" else "train_incremental.py"
    # commands = []
    # for t in range(args.min_time, args.max_time + 1):
    #     commands.append(' '.join(
    #         ["python", train_file, "--time_steps", str(t), "--base_model", args.base_model, "--model", args.model,
    #          "--dataset", args.dataset]))
    #将运行的命令作为字符串写进列表

    # with open(config_file, 'r') as f:
    #     config = json.load(f)
    #     for name, value in config.items():
    #         if name in FLAGS.__flags:
    #             FLAGS.__flags[name].value = value
    #该处可拓展成txt控制参数 直接运行
    start_time = time.perf_counter()
    train(args, device)
    print("train time is: {}".format(time.perf_counter() - start_time))
