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
            for _ in range(self.numLayers):
                hidLayers.append((ResGatedGraphConv(dimHid, dimHid), 'x, edge_index -> x'))
                hidLayers.append(BatchNorm(dimHid))
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

    # def loss(self, y, y_target, weight):
    #     loss = nn.CrossEntropyLoss(weight=weight.type(torch.float))(y ,y_target)

    #     return loss


    # def update(self, lr):

    #     update = torch.optim.Adam(self.parameters(), lr=lr )

    #     return update


    # def update_learning_rate(self, optimizer, lr):

    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

    #         return optimizer


    # def nb_param(self):

    #     # return self.nb_param
    #     return self.dimOut
