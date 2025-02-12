import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from .hourglass import HourglassNetWoSpp, HourglassNetGuid, HourglassNetWoSpp4qt, HourglassNetGuid4mttmask, HourglassNetWoSpp415
from einops import rearrange
from spynet.Spy_net import flow_warp


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B * N, C)

    x[idx1.reshape(-1)] = x1.reshape(B * N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B * N2, C)

    x = x.reshape(B, N, C)
    return x

@torch.no_grad()
def dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=4, label_batch_list=None, out_type='res'):
    """return：
    input_batch: current frame texture YUV, the residual of the previous and current frame after warping, the residual of the next and current frame after warping, or the p_frame_aligned after 64x64 alignment
    flow: optical flow and MDF for the previous frame, optical flow and MDF for the next frame
    """
    # YYY spynet downsampling
    I_frame_YYY = F.interpolate(i_frame[:, 0:1], scale_factor=1 / ds).repeat(1, 3, 1, 1) / 255.0
    P0_frame_YYY = F.interpolate(p0_frame[:, 0:1], scale_factor=1 / ds).repeat(1, 3, 1, 1) / 255.0
    P1_frame_YYY = F.interpolate(p1_frame[:, 0:1], scale_factor=1 / ds).repeat(1, 3, 1, 1) / 255.0
    P0_flow_list = flow_net(I_frame_YYY, P0_frame_YYY)
    P1_flow_list = flow_net(I_frame_YYY, P1_frame_YYY)
    return F.interpolate(P0_flow_list[-1] * ds, scale_factor=ds), F.interpolate(P1_flow_list[-1] * ds, scale_factor=ds)  # flow


@torch.no_grad()
def flow_norm(tensorFlow):
    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorFlow.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorFlow.size(2) - 1.0) / 2.0)], 1)
    return tensorFlow


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding, stride=1, norm=False):
        super(ResidualBlock, self).__init__()
        if norm:
            self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),  
                                      torch.nn.InstanceNorm2d(outchannel), nn.ReLU(inplace=True), nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),  # nn.BatchNorm2d(outchannel)
                                      torch.nn.InstanceNorm2d(outchannel))
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),  
                                              torch.nn.InstanceNorm2d(outchannel))
        else:
            self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),  
                                      nn.ReLU(inplace=True), nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),  
                                      )
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),  
                                              )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UpResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding, stride=1, norm=False):
        super(UpResidualBlock, self).__init__()

        self.norm = norm
        upsample_outchannel = outchannel * 2

        if norm:
            self.left = nn.Sequential(nn.Conv2d(inchannel, upsample_outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), torch.nn.InstanceNorm2d(upsample_outchannel), nn.ReLU(inplace=True), nn.ConvTranspose2d(upsample_outchannel, outchannel, kernel_size=2, stride=2, bias=False),
                                      torch.nn.InstanceNorm2d(outchannel))
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False), torch.nn.InstanceNorm2d(outchannel))
        else:
            self.left = nn.Sequential(nn.Conv2d(inchannel, upsample_outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), nn.ReLU(inplace=True), nn.ConvTranspose2d(upsample_outchannel, outchannel, kernel_size=2, stride=2, bias=False), )
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = self.left(x)
        shortcut = self.shortcut(x)
        shortcut = F.interpolate(shortcut, scale_factor=2, mode='nearest')
        out += shortcut
        out = F.relu(out)
        return out


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b = 0
        uniform_distribution = Uniform(-0.5 * torch.ones(x.size()) * (2 ** b), 0.5 * torch.ones(x.size()) * (2 ** b)).sample().cuda()
        return torch.round(x + uniform_distribution) - uniform_distribution

    @staticmethod
    def backward(ctx, g):
        return g


class ACConv2d(nn.Module):
    """Asysmetirc kernels conv layers, refer to https://zhuanlan.zhihu.com/p/338800630

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ACConv2d, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.ac1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), bias=bias)
        self.ac2 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0), bias=bias)
        self.fusedconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        if self.training:
            ac1 = self.ac1(x)
            ac2 = self.ac2(x)
            x = self.conv(x)
            return (ac1 + ac2 + x) / 3
        else:
            x = self.fusedconv(x)
            return x

    def train(self, mode=True):
        super().train(mode=mode)
        if mode is False:
            weight = self.conv.weight.cpu().detach().numpy()
            weight[:, :, 1:2, :] = weight[:, :, 1:2, :] + self.ac1.weight.cpu().detach().numpy()
            weight[:, :, :, 1:2] = weight[:, :, :, 1:2] + self.ac2.weight.cpu().detach().numpy()
            self.fusedconv.weight = torch.nn.Parameter(torch.FloatTensor(weight / 3))
            if self.bias:
                bias = self.conv.bias.cpu().detach().numpy() + self.conv.ac1.cpu().detach().numpy() + self.conv.ac2.cpu().detach().numpy()
                self.fusedconv.bias = torch.nn.Parameter(torch.FloatTensor(bias / 3))
            if torch.cuda.is_available():
                self.fusedconv = self.fusedconv.cuda()


class AccResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding, stride=1, norm=False):
        super(AccResidualBlock, self).__init__()
        if norm:
            self.left = nn.Sequential(ACConv2d(inchannel, outchannel, kernel_size, stride, padding, bias=False), 
                                      nn.InstanceNorm2d(outchannel), nn.ReLU(inplace=True), ACConv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),  
                                      nn.InstanceNorm2d(outchannel))
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential( 
                    ACConv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, bias=False), 
                    nn.InstanceNorm2d(outchannel), )
        else:
            self.left = nn.Sequential( 
                ACConv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),  
                nn.ReLU(inplace=True), ACConv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),  
            )
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(ACConv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, bias=False),  
                                              )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, text_fe=False) -> None:
        super().__init__()
        if text_fe:
            self.texture_extractor = nn.Sequential(ACConv2d(1, 4, 3, 1, 1), ACConv2d(4, 8, 3, 2, 1), ResidualBlock(8, 16, 3, 1, 1), ResidualBlock(16, 32, 3, 1, 1))
            self.res_extractor = nn.Sequential(ACConv2d(2, 4, 3, 1, 1), ResidualBlock(4, 8, 3, 1, 2), ResidualBlock(8, 8, 3, 1, 1))
            self.mv_extractor = nn.Sequential(ACConv2d(4, 8, 3, 1, 1), ResidualBlock(8, 8, 3, 1, 2), ResidualBlock(8, 8, 3, 1, 1))
        else:
            self.texture_extractor = nn.Sequential(ACConv2d(1, 4, 5, 1, 2), ResidualBlock(4, 8, 3, 1, 2), ResidualBlock(8, 16, 3, 1, 1))
            self.res_extractor = nn.Sequential(ACConv2d(2, 8, 5, 1, 2), ResidualBlock(8, 16, 3, 1, 2), ResidualBlock(16, 16, 3, 1, 1))
            self.mv_extractor = nn.Sequential(ACConv2d(4, 8, 5, 1, 2), ResidualBlock(8, 16, 3, 1, 2), ResidualBlock(16, 16, 3, 1, 1))

    def forward(self, text, res, mv):
        return torch.cat([self.texture_extractor(text), self.res_extractor(res), self.mv_extractor(mv)], dim=1)


class FeatureExtractor_light2mtt(nn.Module):
    def __init__(self, dlm=True) -> None:
        super().__init__()
        self.dlm = dlm
        self.texture_extractor2 = nn.Sequential(ResidualBlock(1, 4, 3, 1, 2), ResidualBlock(4, 8, 3, 1, 2), ResidualBlock(8, 16, 3, 1, 2))
        if self.dlm:
            self.res_extractor2 = nn.Sequential(ResidualBlock(6, 12, 3, 1, 2), ResidualBlock(12, 24, 3, 1, 2, norm=True), ResidualBlock(24, 36, 3, 1, 2))
        else:
            self.res_extractor2 = nn.Sequential(ResidualBlock(4, 8, 3, 1, 2), ResidualBlock(8, 16, 3, 1, 2, norm=True), ResidualBlock(16, 36, 3, 1, 2))

        self.qt_extractor = nn.Sequential(ResidualBlock(1, 4, 3, 1, 1), ResidualBlock(4, 8, 3, 1, 1, norm=True), ResidualBlock(8, 8, 3, 1, 1))
        self.mv_extractor = nn.Sequential(ResidualBlock(4, 4, 3, 1, 2), ResidualBlock(4, 8, 3, 1, 2, norm=True), ResidualBlock(8, 16, 3, 1, 2))

    def forward(self, luma, qt_feat, mv, res):
        return torch.cat([self.texture_extractor2(luma), self.res_extractor2(res), self.mv_extractor(mv), self.qt_extractor(qt_feat), ], dim=1)


class FeatureExtractor_light3mtt(nn.Module):
    def __init__(self, res_dim=4) -> None:
        super().__init__()
        self.texture_extractor2 = nn.Sequential(ResidualBlock(1, 4, 3, 1, 2), ResidualBlock(4, 8, 3, 1, 2), ResidualBlock(8, 16, 3, 1, 1), ResidualBlock(16, 32, 3, 1, 1))
        if res_dim == 6:
            self.res_extractor2 = nn.Sequential(ResidualBlock(6, 12, 3, 1, 2), ResidualBlock(12, 24, 3, 1, 2, norm=True), ResidualBlock(24, 32, 3, 1, 1))
        elif res_dim == 4:
            self.res_extractor2 = nn.Sequential(ResidualBlock(4, 8, 3, 1, 2), ResidualBlock(8, 16, 3, 1, 2, norm=True), ResidualBlock(16, 32, 3, 1, 1))
        else:
            raise Exception("invalid residual dimension.")
        self.mv_extractor = nn.Sequential(ResidualBlock(4, 4, 3, 1, 2), ResidualBlock(4, 8, 3, 1, 2, norm=True), ResidualBlock(8, 16, 3, 1, 1), ResidualBlock(16, 32, 3, 1, 1))
        self.fusion = nn.Sequential(ResidualBlock(96, 128, 3, 1, 1), ResidualBlock(128, 128, 3, 1, 1))

    def forward(self, luma, mv, res):
        return self.fusion(torch.cat([self.texture_extractor2(luma), self.res_extractor2(res), self.mv_extractor(mv)], dim=1))


class QT_Net_HLG(nn.Module):
    def __init__(self, qml=False, tml=False, guide=False, text_fe=False) -> None:
        """_summary_

        Args:
            spp (bool, optional): spatial pyramid pooling. Defaults to False.
            qml (bool, optional): qp modualtion layer. Defaults to False.
            tml (bool, optional): TID Modulation Layer. Defaults to False.
        """
        super().__init__()
        self.guide = False
        self.feature_extractor = FeatureExtractor(text_fe=text_fe)
        if guide:
            self.guide = True
            self.guidance_extraction = nn.Sequential(ResidualBlock(1, 2, 3, 1, 2), nn.ReLU(), ResidualBlock(2, 4, 3, 1, 1), nn.ReLU(), ResidualBlock(4, 4, 3, 1, 1))
            self.qt_net = HourglassNetGuid(nStacks=3, nModules=2, nFeat=64, nClasses=1, inplanes=48, qml=qml, tml=tml, guid_in_chans=4, guide_stride=2)
        else:
            self.qt_net = HourglassNetWoSpp4qt(nStacks=3, nModules=2, nFeat=64, nClasses=1, inplanes=48, qml=qml, tml=tml) 

    def forward(self, luma, flow, qp=None, trans_flow_DAM=False, p0_frame=None, p1_frame=None, out_medium_feat=False, upsample=0, p0_flow=None, p1_flow=None):
        if trans_flow_DAM:
            with torch.no_grad():
                if flow is not None:
                    p0_flow, p1_flow = flow[:, 0], flow[:, 1]
                p0_flow_norm, p1_flow_norm = flow_norm(p0_flow), flow_norm(p1_flow)
                p0_flow_angle, p1_flow_angle = torch.atan2(p0_flow_norm[:, 1], p0_flow_norm[:, 0]), torch.atan2(p1_flow_norm[:, 1], p1_flow_norm[:, 0])
                p0_flow_mag, p1_flow_mag = torch.norm(p0_flow_norm, dim=1), torch.norm(p1_flow_norm, dim=1)
                flow = torch.stack([p0_flow_mag, p1_flow_mag, p0_flow_angle, p1_flow_angle], dim=1)

        with torch.no_grad():
            res = torch.cat([p0_frame - luma, p1_frame - luma], dim=1)
        if self.training:
            res.requires_grad_(True)
            flow.requires_grad_(True)
        input_tensor = self.feature_extractor(luma, res, flow)
        if self.guide:
            guidance = torch.ones_like(luma[:, 0:1])
            grid_size = 32
            for i in range(0, guidance.shape[-2], grid_size):
                guidance[:, :, i, :] = -1
            for j in range(0, guidance.shape[-1], grid_size):
                guidance[:, :, :, j] = -1
            guidance = self.guidance_extraction(guidance) 
            # guidance = None
            if out_medium_feat:
                pred_qt, _, qt_feat = self.qt_net(input_tensor, qp=qp, guidance=guidance, out_mid=out_medium_feat)
            else:
                pred_qt, _ = self.qt_net(input_tensor, qp=qp, guidance=guidance)  
        else:
            if out_medium_feat:
                pred_qt, _, qt_feat = self.qt_net(input_tensor, qp=qp, out_mid=out_medium_feat)
            else:
                pred_qt, _ = self.qt_net(input_tensor, qp=qp)
        if out_medium_feat:
            return pred_qt, input_tensor
        return pred_qt


class FeatureExtractor_mtt(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.texture_extractor = nn.Sequential(ACConv2d(1, 2, 5, 1, 2), ResidualBlock(2, 4, 5, 2, 2), )
        self.res_extractor = nn.Sequential(ACConv2d(2, 8, 5, 1, 2), ResidualBlock(8, 16, 5, 2, 2), )
        self.mv_extractor = nn.Sequential(ACConv2d(4, 8, 5, 1, 2), ResidualBlock(8, 16, 5, 2, 2), )

    def forward(self, text, res, mv):
        return torch.cat([self.texture_extractor(text), self.res_extractor(res), self.mv_extractor(mv)], dim=1)


class MTT_mask_net(nn.Module):
    def __init__(self, qml, dlm=True) -> None:
        super().__init__()
        self.dlm = dlm
        self.feature_extractor2 = FeatureExtractor_light2mtt(dlm=dlm)
        self.mtt_mask_feat = HourglassNetGuid4mttmask(nStacks=2, nModules=2, nFeat=128, nClasses=1, inplanes=76, qml=qml, tml=False, guid_in_chans=4, no_pooling=True, guide_stride=1)
        self.classification = nn.Sequential(nn.Linear(64 * 128, 128), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, luma, p0_flow=None, p1_flow=None, qt_pred=None, qp=None, trans_flow_DAM=True, p0_frame=None, p1_frame=None):
        if trans_flow_DAM:
            p0_flow_norm, p1_flow_norm = p0_flow, p1_flow
            p0_flow_angle, p1_flow_angle = torch.atan2(p0_flow_norm[:, 1], p0_flow_norm[:, 0]) / 3.14, torch.atan2(p1_flow_norm[:, 1], p1_flow_norm[:, 0]) / 3.14
            p0_flow_mag, p1_flow_mag = torch.norm(p0_flow_norm, dim=1) / 128, torch.norm(p1_flow_norm, dim=1) / 128
            flow = torch.stack([p0_flow_mag, p1_flow_mag, p0_flow_angle, - p1_flow_angle], dim=1)
        else:
            flow = torch.cat([p0_flow, -p1_flow], dim=1) / 128

        if self.dlm:
            p0_flow_b, p1_flow_b = 0, 0

            for internal in [0, 1, 2]:
                down_stride, up_stride = 2 ** (6 - internal), 2 ** (5 - internal)
                mask = (qt_pred.round() >= internal) * (qt_pred.round() < (internal + 1))
                p0_flow_b += F.interpolate(F.avg_pool2d(p0_flow, kernel_size=down_stride, stride=down_stride), scale_factor=down_stride, mode='nearest') * F.interpolate((internal + 1 - qt_pred) * mask, scale_factor=8, mode='nearest') + F.interpolate(F.avg_pool2d(p0_flow, kernel_size=up_stride, stride=up_stride),
                                                                                                                                                                                                                                                          scale_factor=up_stride, mode='nearest') * F.interpolate(
                    (qt_pred - internal) * mask, scale_factor=8, mode='nearest')

                p1_flow_b += F.interpolate(F.avg_pool2d(p1_flow, kernel_size=down_stride, stride=down_stride), scale_factor=down_stride, mode='nearest') * F.interpolate((internal + 1 - qt_pred) * mask, scale_factor=8, mode='nearest') + F.interpolate(F.avg_pool2d(p1_flow, kernel_size=up_stride, stride=up_stride),
                                                                                                                                                                                                                                                          scale_factor=up_stride, mode='nearest') * F.interpolate(
                    (qt_pred - internal) * mask, scale_factor=8, mode='nearest')

            p0_frame_aligned = flow_warp(im=p0_frame, flow=p0_flow_b)
            p1_frame_aligned = flow_warp(im=p1_frame, flow=p1_flow_b)

            # from spynet.flow_viz import flow_to_image
            # flow_rgb = flow_to_image(p0_flow[0].permute(1,2,0).detach().cpu().numpy())
            # plt.imsave('/code/flow_vis.png', flow_rgb[:, :, [0,1,2]]/255.0)

            res_p0 = torch.cat([luma - p0_frame, luma - flow_warp(im=p0_frame, flow=p0_flow), luma - p0_frame_aligned], dim=1)
            res_p1 = torch.cat([luma - p1_frame, luma - flow_warp(im=p1_frame, flow=p1_flow), luma - p1_frame_aligned], dim=1)
            res = torch.cat([res_p0, res_p1], dim=1)
        else:
            res_p0 = torch.cat([luma - p0_frame, luma - flow_warp(im=p0_frame, flow=p0_flow)], dim=1)
            res_p1 = torch.cat([luma - p1_frame, luma - flow_warp(im=p1_frame, flow=p1_flow)], dim=1)
            res = torch.cat([res_p0, res_p1], dim=1)
        input_tensor = self.feature_extractor2(luma, qt_pred, flow, res)
        ctu_decision_list = self.mtt_mask_feat(input_tensor, qp=qp) 
        last_list = []
        for ctu_decision in ctu_decision_list:
            tmp_decision = rearrange(rearrange(ctu_decision, 'b c (hi h) (wi w) -> b c hi h wi w', h=8, w=8), 'b c hi h wi w -> b hi wi c h w')
            last_list.append(self.classification(tmp_decision.reshape(-1, 64 * 128)))
        return last_list


class MTT_Net_HLG(nn.Module):
    def __init__(self, qml=False, tml=False, max_depth=3, residual_type='dyloc') -> None:
        """_summary_
        Args:
            spp (bool, optional): spatial pyramid pooling. Defaults to False.
            qml (bool, optional): qp modualtion layer. Defaults to False.
            tml (bool, optional): TID Modulation Layer. Defaults to False.
            max_depth: 1 means mtt-0, 2 means mtt-0 and mtt-1, 3 means mtt-0, mtt-1, and mtt-2
        """
        super().__init__()

        self.residual_type = residual_type
        if self.residual_type == 'dyloc':
            res_dim = 4
        elif self.residual_type == 'all':
            res_dim = 6

        self.feature_extractor2 = FeatureExtractor_light3mtt(res_dim=res_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        self.guidance_extraction_light = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1), nn.ReLU())

        self.trunk_M1 = HourglassNetWoSpp415(nStacks=3, nModules=2, nFeat=128, nClasses=3, inplanes=128, qml=qml, tml=tml, hg2res=True, no_pooling=True, nopadding=False)
        self.max_depth = max_depth

        if max_depth >= 2:
            self.trunk_M2 = HourglassNetWoSpp415(nStacks=1, nModules=2, nFeat=128, nClasses=3, inplanes=128, qml=qml, tml=tml, hg2res=True, no_pooling=True)  
        if max_depth >= 3:
            self.trunk_M3 = HourglassNetWoSpp415(nStacks=1, nModules=2, nFeat=128, nClasses=3, inplanes=128, qml=qml, tml=tml, hg2res=True, no_pooling=True)

    def forward(self, luma, p0_flow=None, p1_flow=None, qt_pred=None, p0_frame=None, p1_frame=None, qp=None, trans_flow_DAM=True, ctu_decision=None, mask_ratio=None):
        with torch.no_grad():
            if trans_flow_DAM:
                p0_flow_norm, p1_flow_norm = p0_flow, p1_flow
                p0_flow_angle, p1_flow_angle = torch.atan2(p0_flow_norm[:, 1], p0_flow_norm[:, 0]) / 3.14, torch.atan2(p1_flow_norm[:, 1], p1_flow_norm[:, 0]) / 3.14
                p0_flow_mag, p1_flow_mag = torch.norm(p0_flow_norm, dim=1) / 128, torch.norm(p1_flow_norm, dim=1) / 128
                flow = torch.stack([p0_flow_mag, p1_flow_mag, p0_flow_angle, - p1_flow_angle], dim=1)
            else:
                flow = torch.cat([p0_flow, -p1_flow], dim=1) / 128
        if self.residual_type == 'dyloc':
            p0_flow_b, p1_flow_b = 0, 0
            for internal in [0, 1, 2]:
                down_stride, up_stride = 2 ** (7 - internal), 2 ** (6 - internal)
                mask = (qt_pred.round() >= internal) * (qt_pred.round() < (internal + 1))
                p0_flow_b += F.interpolate(F.avg_pool2d(p0_flow, kernel_size=down_stride, stride=down_stride), scale_factor=down_stride, mode='nearest') * F.interpolate((internal + 1 - qt_pred) * mask, scale_factor=16, mode='nearest') + F.interpolate(F.avg_pool2d(p0_flow, kernel_size=up_stride, stride=up_stride),
                                                                                                                                                                                                                                                           scale_factor=up_stride, mode='nearest') * F.interpolate(
                    (qt_pred - internal) * mask, scale_factor=16, mode='nearest')

                p1_flow_b += F.interpolate(F.avg_pool2d(p1_flow, kernel_size=down_stride, stride=down_stride), scale_factor=down_stride, mode='nearest') * F.interpolate((internal + 1 - qt_pred) * mask, scale_factor=16, mode='nearest') + F.interpolate(F.avg_pool2d(p1_flow, kernel_size=up_stride, stride=up_stride),
                                                                                                                                                                                                                                                           scale_factor=up_stride, mode='nearest') * F.interpolate(
                    (qt_pred - internal) * mask, scale_factor=16, mode='nearest')

            p0_frame_aligned = flow_warp(im=p0_frame, flow=p0_flow_b)
            p1_frame_aligned = flow_warp(im=p1_frame, flow=p1_flow_b)

            # from spynet.flow_viz import flow_to_image
            # flow_rgb = flow_to_image(p0_flow[0].permute(1,2,0).detach().cpu().numpy())
            # plt.imsave('/code/flow_vis.png', flow_rgb[:, :, [0,1,2]]/255.0)

            res_p0 = torch.cat([luma - p0_frame_aligned, luma - flow_warp(im=p0_frame, flow=p0_flow)], dim=1)
            res_p1 = torch.cat([luma - p1_frame_aligned, luma - flow_warp(im=p1_frame, flow=p1_flow)], dim=1)
            res = torch.cat([res_p0, res_p1], dim=1)

        elif self.residual_type == 'all':
            res_p0 = torch.cat([luma - p0_frame, luma - flow_warp(im=p0_frame, flow=p0_flow)], dim=1)
            res_p1 = torch.cat([luma - p1_frame, luma - flow_warp(im=p1_frame, flow=p1_flow)], dim=1)
            res = torch.cat([res_p0, res_p1], dim=1)
        else:
            res_p0 = torch.cat([luma - p0_frame, luma - flow_warp(im=p0_frame, flow=p0_flow)], dim=1)
            res_p1 = torch.cat([luma - p1_frame, luma - flow_warp(im=p1_frame, flow=p1_flow)], dim=1)
            res = torch.cat([res_p0, res_p1], dim=1)

        input_tensor = self.feature_extractor2(luma, res, flow)
        overlapping_tensor = rearrange(rearrange(input_tensor, 'b c (hi h) (wi w) -> b c hi h wi w', h=32, w=32), 'b c hi h wi w -> b (hi wi) c h w')
        block_h_num, block_w_num = input_tensor.shape[-2] // 32, input_tensor.shape[-1] // 32
        if self.training:
            ctu_decision = torch.argsort(ctu_decision.view(-1, overlapping_tensor.size(1)), dim=-1, descending=True)
            N_new = int(ctu_decision.shape[-1] * mask_ratio)
            ctu_decision = ctu_decision[:, :N_new]
        else:
            N_new = torch.sum(ctu_decision >= mask_ratio)
            ctu_decision = torch.argsort(ctu_decision.view(-1, overlapping_tensor.size(1)), dim=-1, descending=True)
            drop_decision = ctu_decision[:, N_new:]
            ctu_decision = ctu_decision[:, :N_new]
        c_tmp, h_tmp, w_tmp = overlapping_tensor.shape[-3], overlapping_tensor.shape[-2], overlapping_tensor.shape[-1]
        overlapping_tensor = rearrange(overlapping_tensor, 'b n c h w -> b n (c h w)')

        overlapping_tensor = batch_index_select(x=overlapping_tensor, idx=ctu_decision)
        overlapping_tensor = rearrange(overlapping_tensor, 'b n (c h w) -> b n c h w', c=c_tmp, h=h_tmp, w=w_tmp)
        qp_mtt = qp.reshape(-1, 1) * torch.ones(overlapping_tensor.shape[0], overlapping_tensor.shape[1]).to(overlapping_tensor.device)
        overlapping_tensor = rearrange(overlapping_tensor, 'b n c h w -> (b n) c h w')
        qp_mtt = qp_mtt.reshape(-1, 1)
        out0, _, mid_feat_list = self.trunk_M1(overlapping_tensor, qp=qp_mtt, mtt_0=True)
        out0 = out0[-1]
        out0 = rearrange(out0, 'b c h w -> b h w c')
        if self.max_depth == 1:
            if self.training:
                return out0, ctu_decision
            else:
                return out0, ctu_decision, drop_decision
        out1, _ = self.trunk_M2(mid_feat_list[0], qp=qp_mtt)
        out1 = out1[-1]
        out1 = rearrange(out1, 'b c h w -> b h w c')
        if self.max_depth == 2:
            if self.training:
                return [out0, out1], ctu_decision
            else:
                return [out0, out1], ctu_decision, drop_decision
        out2, _ = self.trunk_M3(mid_feat_list[1], qp=qp_mtt)
        out2 = out2[-1]
        out2 = rearrange(out2, 'b c h w -> b h w c')
        if self.training:
            return [out0, out1, out2], ctu_decision  
        else:
            return [out0, out1, out2], ctu_decision, drop_decision


class MTT_Dire_HLG_base(nn.Module):
    def __init__(self, spp=False, qml=False, tml=False, guide=False, max_depth=2) -> None:
        """_summary_

        Args:
            spp (bool, optional): spatial pyramid pooling. Defaults to False.
            qml (bool, optional): qp modualtion layer. Defaults to False.
            tml (bool, optional): TID Modulation Layer. Defaults to False.
            max_depth: 1 means mtt-0, 2 means mtt-0 and mtt-1, 3 means mtt-0, mtt-1, and mtt-2
        """
        super().__init__()
        self.feature_extractor2 = FeatureExtractor_light2mtt()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        if guide:
            self.guide = True
            self.guidance_extraction = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1), nn.ReLU())
            self.trunk_M1 = HourglassNetGuid(nStacks=3, nModules=2, nFeat=36, nClasses=1, inplanes=48, qml=qml, tml=tml, guid_in_chans=4, no_pooling=True, guide_stride=1)
        else:
            self.trunk_M1 = HourglassNetWoSpp(nStacks=3, nModules=2, nFeat=36, nClasses=1, inplanes=48, qml=qml, tml=tml)
        self.max_depth = max_depth
        if max_depth >= 2:
            if guide:
                self.trunk_M2 = HourglassNetGuid(nStacks=1, nModules=2, nFeat=36, nClasses=1, inplanes=48, qml=qml, tml=tml, guid_in_chans=8, no_pooling=True, guide_stride=1)
            else:
                self.trunk_M2 = HourglassNetWoSpp(nStacks=1, nModules=2, nFeat=36, nClasses=1, inplanes=48, qml=qml, tml=tml)  # max_depth==3, 划分到MTT-2
        if max_depth >= 3:
            if guide:
                self.trunk_M3 = HourglassNetGuid(nStacks=1, nModules=2, nFeat=36, nClasses=1, inplanes=48, qml=qml, tml=tml, guid_in_chans=8, no_pooling=True, guide_stride=1)
            else:
                self.trunk_M3 = HourglassNetWoSpp(nStacks=1, nModules=2, nFeat=36, nClasses=1, inplanes=48, qml=qml, tml=tml)

    def forward(self, texture_feat, luma, qt_feature, p0_flow=None, p1_flow=None, qt_pred=None, qp=None, dyloc=False, trans_flow_DAM=True):
        if dyloc:
            pass
        if trans_flow_DAM:
            with torch.no_grad():
                p0_flow_norm, p1_flow_norm = flow_norm(p0_flow), flow_norm(p1_flow)
                p0_flow_angle, p1_flow_angle = torch.atan2(p0_flow_norm[:, 1], p0_flow_norm[:, 0]), torch.atan2(p1_flow_norm[:, 1], p1_flow_norm[:, 0])
                p0_flow_mag, p1_flow_mag = torch.norm(p0_flow_norm, dim=1), torch.norm(p1_flow_norm, dim=1)
                flow = torch.stack([p0_flow_mag, p1_flow_mag, p0_flow_angle, p1_flow_angle], dim=1)
        input_tensor = self.feature_extractor2(texture_feat, luma, qt_feature, flow)
        if self.guide:
            guidance = F.interpolate(torch.ones_like(texture_feat[:, 0:1]), scale_factor=8)
            grid_size = 128
            for i in range(0, guidance.shape[-2], grid_size):
                guidance[:, :, i, :] = -1
            for j in range(0, guidance.shape[-1], grid_size):
                guidance[:, :, :, j] = -1
            guidance = self.guidance_extraction(guidance)
        else:
            guidance = None
        out0, _, mid_feat_list = self.trunk_M1(input_tensor, qp=qp, guidance=guidance, mtt_0=True)
        if self.max_depth == 1:
            return out0, [torch.zeros_like(out0[0]) for i in range(3)], [torch.zeros_like(out0[0]) for i in range(3)]
        out1, _ = self.trunk_M2(mid_feat_list[0], qp=qp, guidance=guidance)
        if self.max_depth == 2:
            return out0, out1, [torch.zeros_like(out0[0]) for i in range(3)]
        out2, _ = self.trunk_M3(mid_feat_list[1], qp=qp, guidance=guidance)

        return out0, out1, out2
