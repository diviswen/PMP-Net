import torch
import torch.nn as nn
from models.utils import Conv1d, PointNet_FP_Module, PointNet_SA_Module

class Unit(nn.Module):
    def __init__(self, step=1, in_channel=256):
        super(Unit, self).__init__()
        self.step = step
        if step == 1:
            return

        self.conv_z = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_r = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_h = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.relu)

    def forward(self, cur_x, prev_s):
        """
        Args:
            cur_x: Tensor, (B, in_channel, N)
            prev_s: Tensor, (B, in_channel, N)

        Returns:
            h: Tensor, (B, in_channel, N)
            h: Tensor, (B, in_channel, N)
        """
        if self.step == 1:
            return cur_x, cur_x

        z = self.conv_z(torch.cat([cur_x, prev_s], 1))
        r = self.conv_r(torch.cat([cur_x, prev_s], 1))
        h_hat = self.conv_h(torch.cat([cur_x, r * prev_s], 1))
        h = (1 - z) * cur_x + z * h_hat
        return h, h

class StepModel(nn.Module):
    def __init__(self, step=1):
        super(StepModel, self).__init__()
        self.step = step
        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3, [64, 64, 128], group_all=False)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 1024], group_all=True)

        self.fp_module_3 = PointNet_FP_Module(1024, [256, 256], use_points1=True, in_channel_points1=256)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 128], use_points1=True, in_channel_points1=128)
        self.fp_module_1 = PointNet_FP_Module(128, [128, 128, 128], use_points1=True, in_channel_points1=6)

        self.unit_3 = Unit(step=step, in_channel=256)
        self.unit_2 = Unit(step=step, in_channel=128)
        self.unit_1 = Unit(step=step, in_channel=128)

        mlp = [128, 64, 3]
        last_channel = 128 + 32
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)

    def forward(self, point_cloud, prev_s):
        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        # print('l1_xyz, l1_points', l1_xyz.shape, l1_points.shape)
        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        # print('l2_xyz, l2_points', l2_xyz.shape, l2_points.shape)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, 1024, 1)
        # print('l3_xyz, l3_points', l3_xyz.shape, l3_points.shape)

        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points, prev_s['l2'] = self.unit_3(l2_points, prev_s['l2'])
        # print('l2_points, prev_s[l2]', l2_points.shape, prev_s['l2'].shape)

        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points, prev_s['l1'] = self.unit_2(l1_points, prev_s['l1'])
        # print('l1_points, prev_s[l1]', l1_points.shape, prev_s['l1'].shape)

        l0_points = self.fp_module_1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        l0_points, prev_s['l0'] = self.unit_1(l0_points, prev_s['l0'])  # (B, 128, 2048)
        # print('l0_points, prev_s[l0]', l0_points.shape, prev_s['l0'].shape)

        b, _, n = l0_points.shape
        noise = torch.normal(mean=0, std=torch.ones((b, 32, n), device=device))
        delta_xyz = torch.tanh(self.mlp_conv(torch.cat([l0_points, noise], 1))) * 1.0 / 10 ** (self.step - 1)
        point_cloud = point_cloud + delta_xyz
        return point_cloud, delta_xyz




'''
# -------------test StepModel -------------
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
b = 10
npoint = 2048
device = torch.device('cuda:0')
S = StepModel(step=0).to(device)
prev_s = {
    'l0': torch.normal(mean=0, std=torch.ones((b, 128, npoint), dtype=torch.float, device=device)*0.01),
    'l1': torch.normal(mean=0, std=torch.ones((b, 128, 512), dtype=torch.float, device=device)*0.01),
    'l2': torch.normal(mean=0, std=torch.ones((b, 256, 128), dtype=torch.float, device=device)*0.01),
}
point_cloud = torch.randn((b, 3, npoint), device=device)
p, d = S(point_cloud, prev_s)

out = torch.sum(p)

out.backward()
print(p.shape, d.shape)

# ------------------------------------------
'''


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.step_1 = StepModel(step=1)
        self.step_2 = StepModel(step=2)
        self.step_3 = StepModel(step=3)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: Tensor, (B, 2048, 3)
        """
        b, npoint, _ = point_cloud.shape
        device = point_cloud.device
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        prev_s = {
            'l0': torch.normal(mean=0, std=torch.ones((b, 128, npoint), dtype=torch.float, device=device) * 0.01),
            'l1': torch.normal(mean=0, std=torch.ones((b, 128, 512), dtype=torch.float, device=device) * 0.01),
            'l2': torch.normal(mean=0, std=torch.ones((b, 256, 128), dtype=torch.float, device=device) * 0.01)
        }

        pcd_out_1, delta1 = self.step_1(point_cloud, prev_s)
        pcd_out_2, delta2 = self.step_2(pcd_out_1, prev_s)
        pcd_out_3, delta3 = self.step_3(pcd_out_2, prev_s)

        return [pcd_out_1.permute(0, 2, 1).contiguous(), pcd_out_2.permute(0, 2, 1).contiguous(),
                pcd_out_3.permute(0, 2, 1).contiguous()], [delta1, delta2, delta3]

class StepModelNoise(nn.Module):
    def __init__(self, step=1, if_noise=False, noise_dim=3, noise_stdv=1e-2):
        super(StepModelNoise, self).__init__()
        self.step = step
        self.if_noise = if_noise
        self.noise_dim = noise_dim
        self.noise_stdv = noise_stdv
        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3 + (self.noise_dim if self.if_noise else 0), [64, 64, 128],
                                              group_all=False)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 1024], group_all=True)

        self.fp_module_3 = PointNet_FP_Module(1024, [256, 256], use_points1=True, in_channel_points1=256)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 128], use_points1=True, in_channel_points1=128)
        self.fp_module_1 = PointNet_FP_Module(128, [128, 128, 128], use_points1=True,
                                              in_channel_points1=6 + (self.noise_dim if self.if_noise else 0))

        self.unit_3 = Unit(step=step, in_channel=256)
        self.unit_2 = Unit(step=step, in_channel=128)
        self.unit_1 = Unit(step=step, in_channel=128)

        mlp = [128, 64, 3]
        last_channel = 128 + 32
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)

    def forward(self, point_cloud, prev_s):
        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud

        b, _, n = l0_points.shape

        noise_points = torch.normal(mean=0, std=torch.ones((b, (self.noise_dim if self.if_noise else 0), n),
                                                           device=device) * self.noise_stdv)
        l0_points = torch.cat([l0_points, noise_points], 1)

        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)

        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)

        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, 1024, 1)

        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points, prev_s['l2'] = self.unit_3(l2_points, prev_s['l2'])

        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points, prev_s['l1'] = self.unit_2(l1_points, prev_s['l1'])

        l0_points = self.fp_module_1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        l0_points, prev_s['l0'] = self.unit_1(l0_points, prev_s['l0'])  # (B, 128, 2048)

        noise = torch.normal(mean=0, std=torch.ones((b, 32, n), device=device))
        delta_xyz = torch.tanh(self.mlp_conv(torch.cat([l0_points, noise], 1))) * 1.0 / 10 ** (self.step - 1)
        point_cloud = point_cloud + delta_xyz
        return point_cloud, delta_xyz

class ModelNoise(nn.Module):
    def __init__(self, noise_dim=3, noise_stdv=1e-2):
        super(ModelNoise, self).__init__()
        self.step_1 = StepModelNoise(step=1, if_noise=True, noise_dim=noise_dim, noise_stdv=noise_stdv)
        self.step_2 = StepModelNoise(step=2)
        self.step_3 = StepModelNoise(step=3)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: Tensor, (B, 2048, 3)
        """
        b, npoint, _ = point_cloud.shape
        device = point_cloud.device
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        prev_s = {
            'l0': torch.normal(mean=0, std=torch.ones((b, 128, npoint), dtype=torch.float, device=device) * 0.01),
            'l1': torch.normal(mean=0, std=torch.ones((b, 128, 512), dtype=torch.float, device=device) * 0.01),
            'l2': torch.normal(mean=0, std=torch.ones((b, 256, 128), dtype=torch.float, device=device) * 0.01)
        }

        pcd_out_1, delta1 = self.step_1(point_cloud, prev_s)
        pcd_out_2, delta2 = self.step_2(pcd_out_1, prev_s)
        pcd_out_3, delta3 = self.step_3(pcd_out_2, prev_s)

        return [pcd_out_1.permute(0, 2, 1).contiguous(), pcd_out_2.permute(0, 2, 1).contiguous(),
                pcd_out_3.permute(0, 2, 1).contiguous()], [delta1, delta2, delta3]


