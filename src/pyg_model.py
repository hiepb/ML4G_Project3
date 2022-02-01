import torch
import torch.nn as nn
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.nn import Sequential
from torch_geometric.nn import BatchNorm


class RGGConvModel(nn.Module):
    def __init__(self, dimIn, dimEmb, dimHid, dimOut, numLayers):
        super(RGGConvModel, self).__init__()

        self.numLayers = numLayers

        self.encoder = nn.Embedding(dimIn, dimEmb)
        # self.ggcIn = ResGatedGraphConv(in_channels=dimIn, out_channels=dimHid)

        if self.numLayers > 0:
            hidLayers = []
            hidLayers.append((ResGatedGraphConv(dimEmb, dimHid), 'x, edge_index -> x'))
            hidLayers.append(BatchNorm(dimHid, track_running_stats=False))
            hidLayers.append(nn.ReLU(inplace=True))
            for _ in range(self.numLayers-1):
                hidLayers.append((ResGatedGraphConv(dimHid, dimHid), 'x, edge_index -> x'))
                hidLayers.append(BatchNorm(dimHid, track_running_stats=False))
                hidLayers.append(nn.ReLU(inplace=True))
            self.hid = Sequential('x, edge_index', hidLayers)

        self.fc = nn.Linear(dimHid, dimOut)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        # x = self.ggcIn(x, edge_index)
        # x = nn.functional.relu(x)
        if self.numLayers > 0:
            x = self.hid(x, edge_index)
        x = self.fc(x)
        return x
