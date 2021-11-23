"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_cls.py
@Time: 2021/01/21 3:10 PM
"""

import torch.nn as nn
import torch.nn.functional as F
from .curvenet_util import *

curve_config = {
        'default': [[50, 10], [50, 10], None, None],
        'long':  [[10, 30], None,  None,  None]
    }

class CurveNet(nn.Module):
    def __init__(self, num_classes=40, k=20, num_input_to_curvenet=1024, setting='default', device=torch.device('cuda')):
        super(CurveNet, self).__init__()
        self.num_input_to_curvenet = num_input_to_curvenet

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True, device=device)

        # encoder
        self.cic11 = CIC(npoint=self.num_input_to_curvenet, radius=0.2, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=self.num_input_to_curvenet, radius=0.2, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])
        
        self.cic21 = CIC(npoint=1024, radius=0.3, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=1024, radius=0.3, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.5, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.5, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=1, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=1, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][3])

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))
        self.conv01 = nn.Linear(1024 + 512, 512 * 2, bias=True)
        self.conv1 = nn.Linear(512 * 2, 512 * 2, bias=False)
        self.conv2 = nn.Linear(512 * 2, num_classes)
        self.bn01 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dp01 = nn.Dropout(p=0.5)
        self.dp1 = nn.Dropout(p=0.5)
        
    def shuffle(self, xyz):
        x = torch.rand(xyz.shape[0], xyz.shape[1])
        indices = torch.argsort(torch.rand(*x.shape), dim=-1)
        return xyz[torch.arange(xyz.shape[0]).unsqueeze(-1), indices]

    def forward(self, xyz, seqvec):
        # shuffled = torch.swapaxes(self.shuffle(torch.swapaxes(xyz, 1, 2)), 1, 2)
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
 
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)
        x = torch.cat((torch.cat((x_max, x_avg), dim=1).squeeze(-1), seqvec), dim=1)
        
        x = F.relu(self.bn01(self.conv01(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp01(x)
        
        latent_feat = self.bn1(F.leaky_relu(self.conv1(x).unsqueeze(-1), negative_slope=0.2, inplace=True)).squeeze(-1)
        x = self.dp1(latent_feat)
        x = self.conv2(x)
        return x, latent_feat
    
class LSTMWithMetadata(nn.Module):
    def __init__(self, num_classes=40, k=20, num_input_to_curvenet=1024):
        # super(CurveNetWithLSTMHead, self).__init__(num_classes, k, num_input_to_curvenet, setting)
        super(LSTMWithMetadata, self).__init__()
        
        factory_kwargs = {'dtype': torch.float}
        self.num_shapes = 9
        self.num_aminos = 21
        self.input_size = 3 + self.num_shapes + self.num_aminos
        self.num_lstm_layers = 3
        self.lstm_hidden = 10
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_hidden, num_layers=self.num_lstm_layers, bidirectional=True, batch_first=True)
        
        self.projection_size = 3
        self.project = nn.Parameter(torch.empty((self.lstm_hidden*2, self.projection_size), **factory_kwargs), requires_grad=True)
        nn.init.xavier_uniform_(self.project)
#         self.project = nn.Parameter(torch.empty((self.lstm_hidden*2, self.projection_size), **factory_kwargs), requires_grad=True) 
        
        self.fcn1 = nn.Sequential(
            nn.Linear(num_input_to_curvenet * self.projection_size, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),)
        self.fcn2 = nn.Sequential(
            nn.Linear(256 + 1024, 1024, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),)
        self.fcn3 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),)
        self.fcn4 = nn.Linear(512, num_classes)
        
#     def choose_n_points(self, xyz):
#         # (batch, 3, n_points) -> (batch, n_points, 3)
#         xyz = torch.swapaxes(xyz, 1, 2)
#         lstm_out, lstm_grad = self.lstm_head(xyz)
#         # flatten bidirectional output to two features per point
#         # input one hidden layer to extract single value per point
#         out = self.linear1(torch.flatten(lstm_out, start_dim=1))
        
#         # gather top values to move on to curvenet
#         topk_val, topk_ind = torch.topk(out, self.num_input_to_curvenet, dim=1)
#         topk_points = torch.gather(xyz, 1, topk_ind.unsqueeze(-1).repeat(1, 1, 3))
#         shuffled = self.shuffle(topk_points)
#         return torch.swapaxes(shuffled, 1, 2)
    
    def forward(self, xyz, shapes, amino_acids, seqvec):
        # (batch, 3, n_points) -> (batch, n_points, 3)
        x = torch.cat((xyz, shapes, amino_acids), dim=2) # (Batch, SeqLen, self.input_size)
        x, _ = self.lstm(x) # (Batch, SeqLen, self.lstm_hidden*2)
        x = torch.matmul(x, self.project) # (Batch, SeqLen, proj_size)
        x = torch.flatten(x, start_dim=1) # (Batch, SeqLen * proj_size)
        x = self.fcn1(x)
        x = torch.cat((x, seqvec), dim=1).squeeze(-1)
        latent_feat = self.fcn2(x)
        x = self.fcn3(latent_feat)
        x = self.fcn4(x)
        return x, None