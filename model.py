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

# This warning also appears in official DGL Tree-LSTM docs, so ignore it.
warnings.filterwarnings("ignore", message="The input graph for the user-defined edge")


class ChildSumTreeLSTMCell(nn.Module):
    """Copied from official implementation."""

    def __init__(self, x_size, h_size):
        """Init."""
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        """Message UDF."""
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        """Reduce UDF."""
        h_tild = torch.sum(nodes.mailbox["h"], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox["h"]))
        c = torch.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_tild), "c": c}

    def apply_node_func(self, nodes):
        """Apply UDF."""
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * torch.tanh(c)
        return {"h": h, "c": c}

class TreeLSTM(nn.Module):
    """Customised N-ary TreeLSTM.

    Example:
    a = BigVulGraphDataset(sample=10)
    asts = a.item(180189)["asts"]
    batched_g = dgl.batch([i for i in asts if i])
    model = TreeLSTM(200, 200)
    model(batched_g)
    """

    def __init__(
        self,
        x_size=128,
        h_size=128,
        out_dim=100,
        dropout=0,
    ):
        """Init.

        Args:
            x_size (int): Input size.
            h_size (int): Hidden size.
            dropout (int): Dropout (final layer)
        """
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.cell = ChildSumTreeLSTMCell(x_size, h_size)
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.hc1 = torch.randn(self.h_size, 1).to(self.dev)
        self.hc2 = torch.randn(self.h_size, 1).to(self.dev)

        self.fc = nn.Linear(self.h_size, self.out_dim)

    def forward(self, g):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        g : dgl.DGLGraph
            Tree for computation.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        n = g.number_of_nodes()

        # 无掩码
        g.ndata["iou"] = self.cell.W_iou(self.dropout(g.ndata["x"]))
        g.ndata["h"] = torch.zeros((n, self.h_size)).to(self.dev)
        g.ndata["c"] = torch.zeros((n, self.h_size)).to(self.dev)

        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        h1 = self.dropout(g.ndata.pop("h"))  # num_nodes x h_size
        h1 = F.relu(h1)
        node_att = torch.matmul(h1, self.hc1).squeeze(1)  # num_nodes x 1
        
        # 有掩码
        mask_path = g.ndata["mask_path"].long().repeat(self.x_size, 1).T
        mask_poi = g.ndata["mask_poi"].long()
        g.ndata["iou"] = self.cell.W_iou(self.dropout(g.ndata["x"] * mask_path))
        g.ndata["h"] = torch.zeros((n, self.h_size)).to(self.dev)
        g.ndata["c"] = torch.zeros((n, self.h_size)).to(self.dev)
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        h2 = self.dropout(g.ndata.pop("h"))  # num_nodes x h_size
        h2 = F.relu(h2)
        node_att_mask = torch.matmul(h2, self.hc2).squeeze(1)  # num_nodes x 1

        node_att = (node_att + node_att_mask) * mask_poi.float()

        g.ndata["h"] = h1 + h2  # unbatch and get root node (ASSUME ROOT NODE AT IDX=0)
        unbatched = dgl.unbatch(g)
        # 对每个graph提取i.ndata["h"][0]，组成batch_size * (2 * h_size)
        h = torch.cat([i.ndata["h"][0].unsqueeze(0) for i in unbatched], dim=0)
        out = self.fc(h)
        return out, node_att

class myGAT(nn.Module):
    def __init__(self, node_dim=128, edge_dim=128, out_dim=100, hidden_dim=128, device='cpu'):
        super(myGAT, self).__init__()
        self.gat1 = GATConv(
                in_channels=node_dim,
                out_channels=hidden_dim,
                edge_dim=edge_dim
            )
        self.gat2 = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=edge_dim
            )
        self.hc1 = torch.randn(hidden_dim, 1).to(device)
        self.hc2 = torch.randn(hidden_dim, 1).to(device)

        self.fc1 = nn.Linear(hidden_dim, out_dim)
        self.fc2 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, data, batch=None):
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
    
        # 无掩码
        x = x[:, :-3]
        att = self.gat1(x, edge_index, edge_attr)
        att = F.relu(att)
        att = self.gat2(att, edge_index, edge_attr)
        att = F.relu(att)
        node_att = torch.matmul(att, self.hc1).squeeze(1)
        global_att = global_mean_pool(att, batch)
        res = self.fc1(global_att)

        # 有掩码，x[:, -1]是漏洞触发位置标签，此处不需要
        mask_poi = x[:, -2].long()
        mask_path = x[:, -3].long().repeat(x.size(1), 1).T
        x_mask = x * mask_path.float()
        att_mask = self.gat1(x_mask, edge_index, edge_attr)
        att_mask = F.relu(att_mask)
        att_mask = self.gat2(att_mask, edge_index, edge_attr)
        att_mask = F.relu(att_mask)
        node_att_mask = torch.matmul(att_mask, self.hc2).squeeze(1)
        global_att_mask = global_mean_pool(att_mask, batch)
        res_mask = self.fc1(global_att_mask)
    
        # 拼接
        res = torch.cat([res, res_mask], dim=1)
        res = self.fc2(res)

        # 计算注意力
        node_att = (node_att + node_att_mask) * mask_poi.float()

        return res, node_att
        # return torch.softmax(res, dim=0), torch.sigmoid(node_att)
    
def get_cosine_similarity_matrix(B, L):
    # input: Tensor 张量
    # 返回输入张量给定维dim上每行的p范数
    G = torch.matmul(L, B.transpose(1, 2))  # b * k * l
    B = torch.norm(B, p=2, dim=2)  # B: b * l * e -> b * l, dim = 2 is for e's axis
    L = torch.norm(L, p=2, dim=1)  # L: k * e -> (k,), dim = 1 is also for e's axis

    B = B.unsqueeze(1)
    L = L.unsqueeze(1)
    G_norm = torch.matmul(L, B)  # b * k * l    # tensor的乘法

    # prevent divide 0 then, we replace 0 with 1
    ones = torch.ones(G_norm.size())  # 返回一个全为1的张量，形状为G_norm的形状
    device = G.device
    ones = ones.to(G.device)
    G_norm = torch.where(G_norm != 0, G_norm, ones)
    G_hat = G / G_norm
    return G_hat

class WLJAN(nn.Module):
    def __init__(self, position_embed_matrix, overall_label_idx, embedding_dim=128, num_words=10000,
                 num_label=2, kernel_size=8, head_size=20, dropout=0.5, padding_idx=0, out_dim=100):
        super(WLJAN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_words = num_words
        self.num_label = num_label
        self.kernel_size = kernel_size
        self.head_size = head_size
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.out_dim = out_dim

        self.position_embed_matrix = position_embed_matrix.float()
        self.overall_label_idx = overall_label_idx

        # embedding layers（random initialize）   B', L
        self.b_embed = nn.Embedding(self.num_words, self.embedding_dim, padding_idx=self.padding_idx)
        self.b_embed.requires_grad = True
        self.l_embed = nn.Embedding(self.num_label, self.embedding_dim)
        self.l_embed.requires_grad = True

        self.att_b = ATT_BYTE(self.num_label, self.ngram_h, self.kernel_h)
        # self.att_l = ATT_LABEL(self.size_h, self.head_size)

        self.fc = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.out_dim),
            # nn.Linear(self.embedding_dim, self.num_label),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        # x: batch_size * len_byte
        # mask: batch_size * len_byte
        # 对应位置相乘得到x_mask
        len_byte = x.size(1) // 2
        mask = mask[:, len_byte:]
        x = x[:, :len_byte]
        x_mask = x * mask.float()
        if x.device != self.overall_label_idx.device:
            self.overall_label_idx = self.overall_label_idx.to(x.device)
            self.position_embed_matrix = self.position_embed_matrix.to(x.device)

        x_embed = self.b_embed(x)  # b * l * e (batch_size, len_byte, embedding)
        x_mask_embed = self.b_embed(x_mask)  # b * l * e (batch_size, len_byte, embedding)
        label_embed = self.l_embed(self.overall_label_idx)  # k * e (k is num_label)

        G = get_cosine_similarity_matrix(x_embed, label_embed)  # b * k * l
        G_mask = get_cosine_similarity_matrix(x_mask_embed, label_embed)  # b * k * l

        att_b, att_b_avg = self.att_b(G, x_embed)  # b * e
        att_b_mask, att_b_avg_mask = self.att_b(G_mask, x_mask_embed)

        res = self.fc(torch.cat((att_b_avg, att_b_avg_mask), dim=1))  # b * k, for loss
        att = (att_b + att_b_mask) * mask.float()  # b * len

        return res, att 

class ATT_BYTE(nn.Module):
    def __init__(self, num_label, ngram=50, out_channels=8):
        super(ATT_BYTE, self).__init__()
        self.num_label = num_label
        self.hidden_size = self.num_label
        self.out_channels = out_channels
        self.ngram = ngram
        self.num_padding = int(self.ngram / 2)
        self.conv = nn.Sequential(  # 卷积函数
            nn.Conv2d(
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=(self.ngram, self.hidden_size),
                stride=(1, 1),
                padding=(self.num_padding, 0)
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, similarity_matrix, B):
        batch_size = similarity_matrix.size(0)
        similarity_matrix = similarity_matrix.transpose(-1, -2)
        att_byte = self.conv(similarity_matrix.unsqueeze(1)).view(batch_size, self.out_channels, -1)    # [bs, oc, len]
        att_byte = F.max_pool2d(att_byte, (self.out_channels, 1))
        att_byte = F.softmax(att_byte.view(batch_size, -1), dim=1) # [bs, len]
        B = F.avg_pool2d(B, kernel_size=(self.ngram, 1), stride=(1, 1))
        att_byte_avg = torch.matmul(att_byte.unsqueeze(1), B).view(batch_size, -1)

        return att_byte, att_byte_avg  # [bs, e], [bs, len]

class ATT_LABEL(nn.Module):
    def __init__(self, pkt_len, num_head):
        super(ATT_LABEL, self).__init__()
        self.pkt_len = pkt_len
        self.num_head = num_head
        self.head_size = self.pkt_len // self.num_head
        if self.pkt_len % self.num_head != 0:  # 整除
            raise ValueError("The head size (%d/%d) is not a multiple of the packet length heads (%d)"
                             % (pkt_len, num_head, self.head_size))
        self.fc1 = nn.Sequential(
            nn.Linear(self.head_size, 1),
            nn.LeakyReLU(0.01)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.num_head, 1),
            nn.LeakyReLU(0.01)
        )

    def forward(self, similarity_matrix, L):
        batch_size, num_label, _ = similarity_matrix.size()
        att_label = self.fc1(similarity_matrix.view(batch_size, num_label, self.num_head, self.head_size)).squeeze(-1)
        att_label = self.fc2(att_label)
        att_label = F.softmax(att_label.view(batch_size, -1), dim=1).unsqueeze(1)
        att_label = torch.matmul(att_label, L).view(batch_size, -1)
        return att_label


class MaliVD(nn.Module):
    def __init__(self, overall_label_idx, position_embed_matrix, out_dim=100, num_labels=2):
        super(MaliVD, self).__init__()
        self.wljan = WLJAN(position_embed_matrix, overall_label_idx, out_dim=out_dim)
        self.tree_lstm = TreeLSTM(out_dim=out_dim)
        self.gat = myGAT(out_dim=out_dim)
        self.fc = nn.Linear(3 * out_dim, num_labels)  # 3个模型的输出拼接后，最后分类

    def forward(self, x):
        # x: [sequence, tree, graph]
        sequence, tree, graph = x
        # 1. sequence
        seq_out, seq_att = self.wljan(sequence)
        # 2. tree
        tree_out, tree_att = self.tree_lstm(tree)
        # 3. graph
        graph_out, graph_att = self.gat(graph)

        out = torch.cat([seq_out, tree_out, graph_out], dim=1)
        out = self.fc(out)
        
        return out, seq_att, tree_att, graph_att

if __name__ == "__main__":
    pass