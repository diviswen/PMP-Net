import jittor
import jittor.nn as nn
from jittor import init
from jittor.contrib import concat
from models.misc.ops import PointNetFeaturePropagation
from models.transformers import Transformer
from models.pointnet2_partseg import PointnetModule

class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=nn.Relu()):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        # self.conv = nn.Linear(in_channel, out_channel)
        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def execute(self, input):
        """
        Args:
            input: (b, c, n)
        """
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class Unit(nn.Module):
    def __init__(self, step=1, in_channel=256):
        super(Unit, self).__init__()
        self.step = step
        if step == 1:
            return

        self.conv_z = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=nn.Sigmoid())
        self.conv_r = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=nn.Sigmoid())
        self.conv_h = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=nn.Relu())

    def execute(self, cur_x, prev_s):
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

        z = self.conv_z(concat([cur_x, prev_s], dim=1))
        r = self.conv_r(concat([cur_x, prev_s], dim=1))
        h_hat = self.conv_h(concat([cur_x, r * prev_s], dim=1))
        h = (1 - z) * cur_x + z * h_hat
        return h, h


class StepModel(nn.Module):
    def __init__(self, step=1, if_noise=False, noise_dim=3, noise_stdv=1e-2, dim_tail=32):
        super(StepModel, self).__init__()
        self.step = step
        self.noise_dim = noise_dim
        self.noise_stdv = noise_stdv
        self.if_noise = if_noise
        self.dim_tail = dim_tail

        self.sa_module_1 = PointnetModule([3 + (self.noise_dim if self.if_noise else 0), 64, 64, 128], n_points=512, radius=0.2, n_samples=32)
        self.sa_module_2 = PointnetModule([128, 128, 128, 256], n_points=128, radius=0.4, n_samples=32)
        self.sa_module_3 = PointnetModule([256, 256, 512, 1024], n_points=None, radius=0.2, n_samples=32)

        self.fp_module_3 = PointNetFeaturePropagation(1024+256, [256, 256])
        self.fp_module_2 = PointNetFeaturePropagation(256+128, [256, 128])
        self.fp_module_1 = PointNetFeaturePropagation(128+6, [128, 128, 128])

        self.unit_3 = Unit(step=step, in_channel=256)
        self.unit_2 = Unit(step=step, in_channel=128)
        self.unit_1 = Unit(step=step, in_channel=128)

        mlp = [128, 64, 3]
        last_channel = 128 + self.dim_tail
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True, activation_fn=nn.Relu()))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)

        self.tanh = nn.Tanh()

    def execute(self, pcd, prev_s):
        """
        Args:
            pcd: b, n, 3
            prev_s: b, c, n
        """
        b, n, _ = pcd.shape
        pcd_bcn = pcd.transpose(0, 2, 1)
        l0_xyz = pcd
        l0_points = pcd_bcn
        if self.if_noise:
            noise_points = init.gauss([b, 3, n], 'float', mean=0.0, std=self.noise_stdv)
            l0_points = jittor.concat([l0_points, noise_points], 1)
        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_points)  # b, 512, 128 (bnc)
        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)

        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points, prev_s['l2'] = self.unit_3(l2_points, prev_s['l2'])

        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points, prev_s['l1'] = self.unit_2(l1_points, prev_s['l1'])

        l0_points = self.fp_module_1(l0_xyz, l1_xyz, concat([pcd_bcn, pcd_bcn], dim=1), l1_points)
        l0_points, prev_s['l0'] = self.unit_1(l0_points, prev_s['l0'])  # (B, 128, 2048)

        noise = init.gauss([b, 32, n], 'float', mean=0.0, std=1.0)
        feat = concat([l0_points, noise], dim=1)
        delta_xyz = self.tanh(self.mlp_conv(feat)) * 1.0 / 10 ** (self.step - 1)
        point_cloud = (pcd_bcn + delta_xyz).transpose(0, 2, 1)
        return point_cloud, delta_xyz

class PMPNet(nn.Module):
    def __init__(self, dataset='ShapeNet'):
        super(PMPNet, self).__init__()
        self.step_1 = StepModel(step=1, if_noise=dataset=='ShapeNet')
        self.step_2 = StepModel(step=2)
        self.step_3 = StepModel(step=3)

    def execute(self, point_cloud):
        """
        Args:
            point_cloud: Tensor, (B, 2048, 3)
        """
        b, npoint, _ = point_cloud.shape
        #pcd_bcn = point_cloud.permute(0, 2, 1)
        prev_s = {
            'l0': init.gauss((b, 128, npoint), 'float', mean=0.0, std=1.0),
            'l1': init.gauss((b, 128, 512), 'float', mean=0.0, std=1.0),
            'l2': init.gauss((b, 256, 128), 'float', mean=0.0, std=1.0)
        }
        pcd_out_1, delta1 = self.step_1(point_cloud, prev_s)
        pcd_out_2, delta2 = self.step_2(pcd_out_1, prev_s)
        pcd_out_3, delta3 = self.step_3(pcd_out_2, prev_s)

        return [pcd_out_1, pcd_out_2, pcd_out_3], [delta1, delta2, delta3]

class StepModelTransformer(nn.Module):
    def __init__(self, step=1, if_noise=False, noise_dim=3, noise_stdv=1e-2, dim_tail=32):
        super(StepModelTransformer, self).__init__()
        self.step = step
        self.noise_dim = noise_dim
        self.noise_stdv = noise_stdv
        self.if_noise = if_noise
        self.dim_tail = dim_tail

        self.sa_module_1 = PointnetModule([3 + (self.noise_dim if self.if_noise else 0), 64, 64, 128], n_points=512, radius=0.2, n_samples=32)
        self.transformer_start_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointnetModule([128, 128, 128, 256], n_points=128, radius=0.4, n_samples=32)
        self.transformer_start_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointnetModule([256, 256, 512, 1024], n_points=None, radius=0.2, n_samples=32)

        self.fp_module_3 = PointNetFeaturePropagation(1024+256, [256, 256])
        self.fp_module_2 = PointNetFeaturePropagation(256+128, [256, 128])
        self.fp_module_1 = PointNetFeaturePropagation(128+6, [128, 128, 128])

        self.unit_3 = Unit(step=step, in_channel=256)
        self.unit_2 = Unit(step=step, in_channel=128)
        self.unit_1 = Unit(step=step, in_channel=128)

        mlp = [128, 64, 3]
        last_channel = 128 + self.dim_tail
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True, activation_fn=nn.Relu()))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)

        self.tanh = nn.Tanh()

    def execute(self, pcd, prev_s):
        """
        Args:
            pcd: b, n, 3
            prev_s: b, c, n
        """
        b, n, _ = pcd.shape
        pcd_bcn = pcd.transpose(0, 2, 1)
        l0_xyz = pcd
        l0_points = pcd_bcn
        if self.if_noise:
            noise_points = init.gauss([b, 3, n], 'float', mean=0.0, std=self.noise_stdv)
            l0_points = jittor.concat([l0_points, noise_points], 1)
        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_points)  # b, 512, 128 (bnc)

        l1_points = self.transformer_start_1(l1_points, l1_xyz)
        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)

        l2_points = self.transformer_start_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)

        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points, prev_s['l2'] = self.unit_3(l2_points, prev_s['l2'])

        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points, prev_s['l1'] = self.unit_2(l1_points, prev_s['l1'])

        l0_points = self.fp_module_1(l0_xyz, l1_xyz, concat([pcd_bcn, pcd_bcn], dim=1), l1_points)
        l0_points, prev_s['l0'] = self.unit_1(l0_points, prev_s['l0'])  # (B, 128, 2048)

        noise = init.gauss([b, 32, n], 'float', mean=0.0, std=1.0)
        feat = concat([l0_points, noise], dim=1)
        delta_xyz = self.tanh(self.mlp_conv(feat)) * 1.0 / 10 ** (self.step - 1)
        point_cloud = (pcd_bcn + delta_xyz).transpose(0, 2, 1)
        return point_cloud, delta_xyz

class PMPNetPlus(nn.Module):
    def __init__(self, dataset='ShapeNet'):
        super(PMPNetPlus, self).__init__()
        self.step_1 = StepModelTransformer(step=1, if_noise=dataset=='ShapeNet', dim_tail=32)
        self.step_2 = StepModelTransformer(step=2, if_noise=dataset=='ShapeNet', dim_tail=32)
        self.step_3 = StepModelTransformer(step=3, if_noise=dataset=='ShapeNet', dim_tail=32)

    def execute(self, point_cloud):
        """
        Args:
            point_cloud: Tensor, (B, 2048, 3)
        """
        b, npoint, _ = point_cloud.shape
        #pcd_bcn = point_cloud.permute(0, 2, 1)
        prev_s = {
            'l0': init.gauss((b, 128, npoint), 'float', mean=0.0, std=1.0),
            'l1': init.gauss((b, 128, 512), 'float', mean=0.0, std=1.0),
            'l2': init.gauss((b, 256, 128), 'float', mean=0.0, std=1.0)
        }
        pcd_out_1, delta1 = self.step_1(point_cloud, prev_s)
        pcd_out_2, delta2 = self.step_2(pcd_out_1, prev_s)
        pcd_out_3, delta3 = self.step_3(pcd_out_2, prev_s)

        return [pcd_out_1, pcd_out_2, pcd_out_3], [delta1, delta2, delta3]

