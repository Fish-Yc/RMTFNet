import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import (
    ResidualBlocksWithInputConv,
    PixelShufflePack,
    flow_warp
)
from basicsr.archs.spynet_arch import SpyNet

from mmcv.ops.modulated_deform_conv import (
    ModulatedDeformConv2d,
    modulated_deform_conv2d
)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class MotionReliabilityGate(nn.Module):
    """Estimate whether the propagated feature is trustworthy at each pixel."""

    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or max(channels // 4, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3 + 1, hidden_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 1, 1, 1, 0)
        )
        constant_init(self.gate[-1], 0, bias=2)

    def forward(self, current_feat, prop_feat, flow):
        flow_mag = torch.norm(flow, p=2, dim=1, keepdim=True)
        residual = torch.abs(current_feat - prop_feat)
        cond = torch.cat([current_feat, prop_feat, residual, flow_mag], dim=1)
        return torch.sigmoid(self.gate(cond))


class TemporalSelectiveFusion(nn.Module):
    """Lightweight stream-wise weighting before recurrent feature refinement."""

    def __init__(self, num_streams, channels, hidden_channels=16):
        super().__init__()
        self.num_streams = num_streams
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.weight_net = nn.Sequential(
            nn.Conv2d(num_streams * channels, hidden_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, num_streams, 1, 1, 0)
        )
        constant_init(self.weight_net[-1], 0)

    def forward(self, feats):
        descriptor = torch.cat([self.pool(feat) for feat in feats], dim=1)
        weights = self.weight_net(descriptor).softmax(dim=1)
        weights = weights * self.num_streams
        return [feat * weights[:, i:i + 1] for i, feat in enumerate(feats)]


class FrequencyDetailRefinement(nn.Module):
    """A small high-frequency residual branch with fixed Laplacian guidance."""

    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or max(channels // 4, 16)
        kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', kernel)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0),
            nn.Sigmoid()
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, feat):
        kernel = self.laplacian_kernel.repeat(feat.size(1), 1, 1, 1)
        high_freq = F.conv2d(feat, kernel, padding=1, groups=feat.size(1))
        detail = self.refine(high_freq) * self.gate(feat)
        return feat + self.res_scale * detail


# =========================================================
# Reliability-Aware Deformable Alignment
# =========================================================
class DeformAlignment(ModulatedDeformConv2d):
    """DCNv2 alignment controlled by a lightweight motion reliability gate."""

    def __init__(self,
                 *args,
                 max_residue_magnitude=10,
                 use_motion_reliability=True,
                 **kwargs):
        self.max_residue_magnitude = max_residue_magnitude
        self.use_motion_reliability = use_motion_reliability
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        constant_init(self.conv_offset[-1], 0)
        if use_motion_reliability:
            self.reliability_gate = MotionReliabilityGate(self.out_channels)

    def forward(self, x, feat, flow):
        """
        x    : warped previous propagated feature
        feat : current frame feature
        flow : optical flow
        """
        if self.use_motion_reliability:
            reliability = self.reliability_gate(feat, x, flow)
        else:
            reliability = 1

        cond = torch.cat([feat, x, flow], dim=1)
        out = self.conv_offset(cond)

        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset_residue = self.max_residue_magnitude * torch.tanh(
            torch.cat([o1, o2], dim=1)
        ) * reliability
        # The input feature has already been warped by optical flow, so DCNv2
        # only predicts a small residual offset around the warped position.
        offset = offset_residue
        mask = torch.sigmoid(mask) * reliability

        return modulated_deform_conv2d(
            x, offset, mask,
            self.weight, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups, self.deform_groups
        )


# =========================================================
# RMTFNet with reliability-guided motion-temporal-frequency restoration
# =========================================================
@ARCH_REGISTRY.register()
class RMTFNet(nn.Module):
    """
    Reliability-Guided Motion-Temporal-Frequency Network.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 propagation_rounds=2,
                 spynet_pretrained=None,
                 cpu_cache_length=100,
                 use_motion_reliability=True,
                 use_temporal_selective_fusion=True,
                 use_frequency_detail=True,
                 ):
        super().__init__()

        self.mid_channels = mid_channels
        self.cpu_cache_length = cpu_cache_length
        self.use_temporal_selective_fusion = use_temporal_selective_fusion
        self.use_frequency_detail = use_frequency_detail
        self.propagation_rounds = int(propagation_rounds)
        if self.propagation_rounds < 1:
            raise ValueError('propagation_rounds must be at least 1.')

        # -----------------------------------------------------
        # Optical Flow
        # -----------------------------------------------------
        self.spynet = SpyNet(load_path=spynet_pretrained)

        # -----------------------------------------------------
        # Feature extraction
        # -----------------------------------------------------
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 5)
        )

        # -----------------------------------------------------
        # Deformable alignment + backbone
        # -----------------------------------------------------
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.temporal_select = nn.ModuleDict()

        self.propagation_modules = [
            f'{direction}_{iter_}'
            for iter_ in range(1, self.propagation_rounds + 1)
            for direction in ['backward', 'forward']
        ]
        for i, module in enumerate(self.propagation_modules):
            self.deform_align[module] = DeformAlignment(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=8,
                max_residue_magnitude=max_residue_magnitude,
                use_motion_reliability=use_motion_reliability
            )
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels,
                mid_channels,
                num_blocks
            )
            if use_temporal_selective_fusion:
                self.temporal_select[module] = TemporalSelectiveFusion(
                    2 + i, mid_channels
                )

        # -----------------------------------------------------
        # Reconstruction
        # -----------------------------------------------------
        self.reconstruction = ResidualBlocksWithInputConv(
            (1 + len(self.propagation_modules)) * mid_channels, mid_channels, 5
        )
        if use_frequency_detail:
            self.frequency_detail = FrequencyDetailRefinement(mid_channels)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3
        )
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3
        )
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    # ---------------------------------------------------------
    # Flow
    # ---------------------------------------------------------
    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(
            n, t - 1, 2, h, w
        )
        flows_forward = self.spynet(lqs_2, lqs_1).view(
            n, t - 1, 2, h, w
        )

        return flows_forward, flows_backward

    def match_flow_size(self, flow, size):
        n, t, _, h, w = flow.size()
        out_h, out_w = size
        if (h, w) == (out_h, out_w):
            return flow

        flow = flow.reshape(-1, 2, h, w)
        scale = flow.new_tensor([float(out_w) / float(w), float(out_h) / float(h)])
        flow = flow * scale.view(1, 2, 1, 1)
        flow = F.interpolate(flow, size=size, mode='bilinear', align_corners=False)
        return flow.reshape(n, t, 2, out_h, out_w)

    # ---------------------------------------------------------
    # Upsample
    # ---------------------------------------------------------
    def upsample(self, lqs, feats):
        outputs = []

        for i in range(lqs.size(1)):
            hr = [feats['spatial'][i]]
            hr.extend([feats[k][i] for k in self.propagation_modules])
            hr = torch.cat(hr, dim=1)

            hr = self.reconstruction(hr)
            if self.use_frequency_detail:
                hr = self.frequency_detail(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)

            hr = hr + lqs[:, i]

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, lqs):
        n, t, c, h, w = lqs.size()

        # Downsample for flow
        lqs_down = F.interpolate(
            lqs.view(-1, c, h, w),
            scale_factor=0.25,
            mode='bicubic'
        )
        h_down, w_down = lqs_down.shape[2:]
        lqs_down = lqs_down.view(n, t, c, h_down, w_down)

        # Spatial features
        feats = {}
        feats_ = self.feat_extract(lqs.view(-1, c, h, w))
        hh, ww = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, hh, ww)
        feats['spatial'] = [feats_[:, i] for i in range(t)]

        # Flow
        flows_forward, flows_backward = self.compute_flow(lqs_down)
        flows_forward = self.match_flow_size(flows_forward, (hh, ww))
        flows_backward = self.match_flow_size(flows_backward, (hh, ww))

        # Propagation
        for iter_ in range(1, self.propagation_rounds + 1):
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []

                flows = flows_backward if direction == 'backward' else flows_forward
                frame_idx = list(range(t))
                flow_idx = list(range(t - 1))

                if direction == 'backward':
                    frame_idx = frame_idx[::-1]
                    flow_idx = flow_idx[::-1]

                feat_prop = flows.new_zeros(
                    n, self.mid_channels, hh, ww
                )

                for i, idx in enumerate(frame_idx):
                    x_i = feats['spatial'][idx]

                    if i > 0:
                        flow = flows[:, flow_idx[i - 1]]
                        feat_prop = flow_warp(
                            feat_prop, flow.permute(0, 2, 3, 1)
                        )
                        feat_prop = self.deform_align[module](
                            feat_prop, x_i, flow
                        )

                    feat = [x_i] + [
                        feats[k][idx]
                        for k in feats if k not in ['spatial', module]
                    ] + [feat_prop]
                    if self.use_temporal_selective_fusion:
                        feat = self.temporal_select[module](feat)

                    feat_prop = feat_prop + self.backbone[module](
                        torch.cat(feat, dim=1)
                    )
                    feats[module].append(feat_prop)

                if direction == 'backward':
                    feats[module] = feats[module][::-1]

        return self.upsample(lqs, feats)
