import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from generic import GenericModel


class BasicGraphEncoder(GenericModel):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super().__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = torch.nn.ReLU()
        self.ln = torch.nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = torch.nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = torch.nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x


class BigGraphEncoder(GenericModel):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super().__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = torch.nn.ReLU()
        self.ln = torch.nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv4 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv5 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = torch.nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = torch.nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x1 = x.relu()
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        x3 = self.conv3(x2, edge_index)
        x3 = x3 + x1  # residual connection
        x3 = x3.relu()
        x4 = self.conv4(x3, edge_index)
        x4 = x4.relu()
        x5 = self.conv5(x4, edge_index)
        x5 = x5 + x3  # residual connection
        x = global_mean_pool(x5, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
