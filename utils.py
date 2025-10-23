import warnings
import dgl
import torch
import torch.nn as nn
import random
import networkx as nx
import dgl
import numpy as np
import torch.nn.functional as F
from thop import profile

import json
import pickle
import os
import numpy as np
import time
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.io import read_tu_data
from torch_geometric.data.collate import collate
import random
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from thop import profile


def TreeLoader(root_path, batch_size=32, test_percent=0.2):
    """将树数据分为训练集和测试集，并返回两个可循环访问的DataLoader"""
    if not os.path.exists(root_path):
        raise ValueError(f"路径 {root_path} 不存在")

    # 按字典序读取所有 .pkl 文件
    pkl_files = sorted([f for f in os.listdir(root_path) if f.endswith('.pkl')])
    all_samples = []

    for pkl_file in pkl_files:
        with open(os.path.join(root_path, pkl_file), 'rb') as f:
            data = pickle.load(f)  # 每个文件是一个list，表示多个样本
            all_samples.extend(data)

    num_samples = len(all_samples)
    train_graphs = all_samples[:int(num_samples * (1 - test_percent))]
    test_graphs = all_samples[int(num_samples * (1 - test_percent)):]

    train_batched_graphs = [dgl.batch(train_graphs[i:i + batch_size]).to("cuda:0") 
                            for i in range(0, len(train_graphs), batch_size)]
    test_batched_graphs = [dgl.batch(test_graphs[i:i + batch_size]).to("cuda:0") 
                           for i in range(0, len(test_graphs), batch_size)]

    def train_loader():
        for batch in train_batched_graphs:
            yield batch

    def test_loader():
        for batch in test_batched_graphs:
            yield batch

    return train_loader(), test_loader()

def save_data_pkl(nodes, edges, edge_attr, y, name="custom_dataset"):
    """将数据保存为.pkl文件"""
    data_path = '/data/data/ws/MaliVD/test/test_GNN/create_GNN_data/custom_dataset/data/{}/pkl/'.format(name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    with open(data_path + name + '.pkl', 'wb') as f:
        pickle.dump({'nodes': nodes, 'edges': edges, 'edge_attr': edge_attr, 'y': y}, f)


def save_data_txt(nodes, edges, edge_attr, y, name="custom_dataset", 
                  path="/data/data/ws/MaliVD/test/test_GNN/create_GNN_data/custom_dataset/data/"):
    """保存成5个txt文件
    _A.txt：(m,2)；表示m条边 每行表示 (row, col) -> (node_id, node_id)
    _graph_indicator.txt：(n,1)，第 i 行表示第 i 个结点属于哪个图graph_id
    _graph_labels.txt：(N,1)，第 i 行表示第 i 个图的标签
    _node_labels.txt：(n,1) 行， 第 i 行表示节点标签
    _node_attributes.txt：(n, num_nodefeatures)，第 i 行表示节点 i 的特征
    :param nodes: (num_graph, num_node, num_nodefeature)
    :param edges: (num_graph, num_edge, 2)
    :param edge_attr: (num_graph, num_edge, num_edgefeature)
    :param y: (num_graph, 1)
    :return:
    """
    data_path = path + name + '/raw/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    with open(data_path + name + '_A.txt', 'w') as f:
        for i in range(len(edges)):
            for j in range(len(edges[i][0])):
                f.write("{}, {}\n".format(edges[i][0][j], edges[i][1][j]))

    with open(data_path + name + '_graph_indicator.txt', 'w') as f:
        for i in range(len(nodes)):
            for j in range(len(nodes[i])):
                f.write("{}\n".format(i + 1))

    with open(data_path + name + '_graph_labels.txt', 'w') as f:
        for i in range(len(y)):
            f.write("{}\n".format(y[i]))
    
    # with open(data_path + name + '_node_labels.txt', 'w') as f:
    #     for i in range(len(node_labels)):
    #         for j in range(len(node_labels[i])):
    #             f.write("{}\n".format(", ".join([str(_) for _ in node_labels[i][j]])))

    with open(data_path + name + '_node_attributes.txt', 'w') as f:
        for i in range(len(nodes)):
            for j in range(len(nodes[i])):
                f.write(", ".join([str(x) for x in nodes[i][j]]) + '\n')

    with open(data_path + name + '_edge_attributes.txt', 'w') as f:
        for i in range(len(edge_attr)):
            for j in range(len(edge_attr[i])):
                f.write(", ".join([str(x) for x in edge_attr[i][j]]) + '\n')


class CPGDataset(InMemoryDataset):
    """自定义图数据集类，用于加载和处理图数据"""
    def __init__(self, root='./data/custom_dataset', file_path='./data/custom_dataset/raw/',
                 name="custom_dataset", use_edge_attr=True, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.root = root
        self.file_path = file_path
        self.filenames = os.listdir(file_path)
        self.use_edge_attr = use_edge_attr
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super(CPGDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.sizes = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.file_path
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')
    
    @property
    def raw_file_names(self):
        return self.filenames

    @property
    def processed_file_names(self):
        return self.name + '.pt'
    
    def download(self):
        pass

    def process(self):
        """主程序：如果保存路径没有文件，则对原数据处理并保存，如果有文件，则直接加载
        """
        self.data, self.slices, self.sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data = data_list

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data = data_list

        torch.save((self.data, self.slices, self.sizes), self.processed_paths[0])


def GraphLoader(root_path, batch_size=32, test_percent=0.2):
    """加载图数据并划分为训练集和测试集，返回DataLoader"""
    dataset = CPGDataset(root=root_path, file_path='{}raw/'.format(root_path), name="custom_dataset")
    num_samples = len(dataset)

    train_dataset = dataset[:int(num_samples * (1 - test_percent))]
    test_dataset = dataset[int(num_samples * (1 - test_percent)):]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def SeqLoader(root_path, batch_size=32, test_percent=0.2):
    """
    遍历root_path下所有的pkl文件（按字典序排序后读取），
    每个pkl文件是一个list，表示多个样本，每个样本是一个多维特征向量list。
    拼接所有样本后，划分为train和test，返回DataLoader
    """
    if not os.path.exists(root_path):
        raise ValueError(f"路径 {root_path} 不存在")

    # 按字典序读取所有 .pkl 文件
    pkl_files = sorted([f for f in os.listdir(root_path) if f.endswith('.pkl')])
    all_samples = []

    for pkl_file in pkl_files:
        with open(os.path.join(root_path, pkl_file), 'rb') as f:
            data = pickle.load(f)  # 每个文件是一个list，表示多个样本
            all_samples.extend(data)  # 拼接所有样本

    all_samples = torch.tensor(all_samples, dtype=torch.float32)
    num_samples = len(all_samples)

    train_samples = all_samples[:int(num_samples * (1 - test_percent))]
    test_samples = all_samples[int(num_samples * (1 - test_percent)):]

    train_loader = torch.utils.data.DataLoader(train_samples, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_samples, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



def eval(model, loader, device, batch=None):
    """评估模型性能，返回真实值、预测值和Top-k值"""
    model.eval()
    y, y_hat, topk = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred, loc = model(data, data.batch)
            _y, _y_hat, _topk = output_to_metrics(data, pred, loc, data.batch)
            y.extend(_y)
            y_hat.extend(_y_hat)
            topk.extend(_topk)
    return y, y_hat, topk


def output_to_metrics(data, pred, loc, batch=None):
    """将模型输出转换为评估指标"""
    y, y_hat, topk = data.y.tolist(), pred.argmax(dim=1).tolist(), []
    loc_y = data.x[:, -1].int().tolist()
    batch = data.batch.tolist()
    data_size_count = Counter(batch)    # key: 第几个图，value: 该图的节点数量

    pre_idx = 0
    for key in sorted(data_size_count.keys()):
        post_idx = pre_idx + data_size_count[key]

        loc_pred = loc[pre_idx:post_idx].tolist()
        loc_gt = loc_y[pre_idx:post_idx]
        rank = (data_size_count[key] - np.argsort(np.argsort(loc_pred))).tolist()

        tmp_topk = []
        for node_idx in range(len(loc_gt)):
            if loc_gt[node_idx] == 1:
                tmp_topk.append(rank[node_idx])
        topk.append(max(tmp_topk) if len(tmp_topk) > 0 else None)
        pre_idx = post_idx
    return y, y_hat, topk


def save_model(model, path, **kwargs):
    """保存模型及其相关信息到指定路径"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    kwargs['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    kwargs['path'] = path
    kwargs['model_paras'] = str(model)
    with open(path + '.json', 'w') as f:
        json.dump(kwargs, f)


def get_metrics(y, pred):
    """计算并返回分类评估指标"""
    return {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred),
        'recall': recall_score(y, pred),
        'f1': f1_score(y, pred, average='macro')
    }


def localization_report(topk_list: list, interval: int = 5, max_k: int = 100):
    """生成定位性能报告"""
    source_len = len(topk_list)
    topk_list = [_ for _ in topk_list if _ is not None]
    ks = [i for i in range(1, interval)] + [i for i in range(interval, max_k+1, interval)]
    topk_res = {k: 0 for k in ks}
    IFA_res = {k: 0 for k in ks}
    for k in ks:
        IFA_res[k] = sum([1 for _ in topk_list if _ == k]) / len(topk_list)
        topk_res[k] = sum([1 for _ in topk_list if _ <= k]) / len(topk_list)

    # 处理成report的形式，输出到output，每个输出占
    output = "{:>5}{:>10}{:>10}{:>5}{:>6}{:>10}{:>10}\n".format("k", "Top-k", "IFA", '|', "k", "Top-k", "IFA")
    for k in range(len(ks) // 2):
        output += "{:>6}{:>10.4f}{:>10.4f}{:>5}{:>6}{:>10.4f}{:>10.4f}\n".format(
            ks[k], topk_res[ks[k]], IFA_res[ks[k]], '|', 
            ks[k + len(ks) // 2], topk_res[ks[k + len(ks) // 2]], IFA_res[ks[k + len(ks) // 2]])
    output += " * samples: origin: {}, none: {}, actual: {}".format(source_len, source_len - len(topk_list), len(topk_list))
    return output

