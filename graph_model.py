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


class FatGraphEncoder(GenericModel):
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
        self.conv6 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv7 = GCNConv(graph_hidden_channels, graph_hidden_channels)
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
        x6 = self.conv6(x5, edge_index)
        x6 = x6.relu()
        x7 = self.conv7(x6, edge_index)
        x7 = x7 + x5  # residual connection
        x = global_mean_pool(x7, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x


class UltraFatGraphEncoder(GenericModel):
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
        self.conv6 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv7 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv8 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv9 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv10 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv11 = GCNConv(graph_hidden_channels, graph_hidden_channels)
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
        x6 = self.conv6(x5, edge_index)
        x6 = x6.relu()
        x7 = self.conv7(x6, edge_index)
        x7 = x7 + x5  # residual connection

        x8 = self.conv8(x7, edge_index)
        x8 = x8.relu()
        x9 = self.conv9(x8, edge_index)
        x9 = x9 + x7  # residual connection

        x10 = self.conv10(x9, edge_index)
        x10 = x10.relu()
        x11 = self.conv11(x10, edge_index)
        x11 = x11 + x9  # residual connection

        x = global_mean_pool(x11, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
