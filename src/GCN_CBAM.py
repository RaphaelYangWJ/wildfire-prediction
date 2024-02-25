import torch
from torch_geometric.nn import GCNConv
import numpy as np
from torch_geometric.data import Data


class Convolutioner(torch.nn.Module):
    def __init__(self, features, out_dim, kernel_size = 7):
        super(Convolutioner, self).__init__()
        self.gcnconv = GCNConv(features, out_dim)
        self.conv = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(5,5) ,stride=(1,1),bias=False)
        self.pool = torch.nn.AvgPool2d((3,3))
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self, data):
        # === computations for gcn inputs
        edge_matrix = torch.tensor([[0,1,2,3,4,5,6,7,8], [5,5,5,5,5,5,5,5,5]], dtype=torch.long)
        attr_matrix = torch.tensor(np.expand_dims(data.flatten(),axis=1), dtype=torch.float32)
        x = torch.tensor([[0],[1],[2],[3],[4],[5],[6],[7],[8]], dtype=torch.float32)
        input_img = Data(x=x, edge_index=edge_matrix, edge_attr=attr_matrix)

        # === feed forward
        x, edge_index, edge_attr = input_img.x, input_img.edge_index, input_img.edge_attr
        out = self.gcnconv(x, edge_index, edge_attr)
        out=self.sigmoid(self.conv(torch.unsqueeze(out, dim=0)))
        out = self.pool(out)
        return out.detach().numpy()[0][0][0]