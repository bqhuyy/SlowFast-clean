'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from slowfast.models.tsm_helper import TemporalShift
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm
from .build import MODEL_REGISTRY
from . import head_helper, resnet_helper, stem_helper
from slowfast.models.video_model_builder import FuseFastToSlow


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, n_segment=None, n_div=None):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        
        if n_segment is None and n_div is None:
            self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv1 = TemporalShift(
                nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False),
                n_segment,
                n_div,
            )
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

    
class Stage(nn.Sequential):
    def __init__(self, dim_in, beta_inv, stage_idx, n_segment, n_div):
        if stage_idx == 2:
            super(Stage, self).__init__(
                Block(3, dim_in//beta_inv, 64//beta_inv, 24//beta_inv, nn.ReLU(inplace=True), None, 2), #16
                Block(3, 24//beta_inv, 72//beta_inv, 24//beta_inv, nn.ReLU(inplace=True), None, 1, n_segment, n_div),
            )
        elif stage_idx == 3:
            super(Stage, self).__init__(
                Block(5, dim_in//beta_inv, 72//beta_inv, 40//beta_inv, nn.ReLU(inplace=True), SeModule(40//beta_inv), 2), #24
                Block(5, 40//beta_inv, 120//beta_inv, 40//beta_inv, nn.ReLU(inplace=True), SeModule(40//beta_inv), 1, n_segment, n_div),
                Block(5, 40//beta_inv, 120//beta_inv, 40//beta_inv, nn.ReLU(inplace=True), SeModule(40//beta_inv), 1, n_segment, n_div),
            )
        elif stage_idx == 4:
            super(Stage, self).__init__(
                Block(3, dim_in//beta_inv, 240//beta_inv, 80//beta_inv, hswish(), None, 2), #40
                Block(3, 80//beta_inv, 200//beta_inv, 80//beta_inv, hswish(), None, 1, n_segment, n_div),
                Block(3, 80//beta_inv, 184//beta_inv, 80//beta_inv, hswish(), None, 1, n_segment, n_div),
                Block(3, 80//beta_inv, 184//beta_inv, 80//beta_inv, hswish(), None, 1, n_segment, n_div),
                Block(3, 80//beta_inv, 480//beta_inv, 112//beta_inv, hswish(), SeModule(112//beta_inv), 1),
                Block(3, 112//beta_inv, 672//beta_inv, 112//beta_inv, hswish(), SeModule(112//beta_inv), 1, n_segment, n_div),
            )
        elif stage_idx == 5:
            super(Stage, self).__init__(
                Block(5, dim_in//beta_inv, 672//beta_inv, 160//beta_inv, hswish(), SeModule(160//beta_inv), 1), #112
                Block(5, 160//beta_inv, 672//beta_inv, 160//beta_inv, hswish(), SeModule(160//beta_inv), 2, n_segment, n_div),
                Block(5, 160//beta_inv, 960//beta_inv, 160//beta_inv, hswish(), SeModule(160//beta_inv), 1, n_segment, n_div),
                nn.Conv2d(160//beta_inv, 960//beta_inv, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(960//beta_inv),
                hswish()
            )
            
class MobilenetV3Stage(nn.Module):
    def __init__(
        self,
        dim_in,
        beta_inv,
        stage_idx,
        n_segment, 
        n_div,
    ):
        super(MobilenetV3Stage, self).__init__()
        self.num_pathways = len(dim_in)
        self.stage_idx = stage_idx
        self.n_segment = n_segment
        self.n_div = n_div
        self._construct(
            dim_in,
            beta_inv,
        )

    def _construct(
        self,
        dim_in,
        beta_inv,
    ):
        for pathway in range(self.num_pathways):
            res_stage = Stage(dim_in[pathway], beta_inv[pathway], self.stage_idx, self.n_segment[pathway], self.n_div[pathway])
            self.add_module("pathway{}_stage{}".format(pathway, self.stage_idx), res_stage)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            m = getattr(self, "pathway{}_stage{}".format(pathway, self.stage_idx))
            x = m(x)
            output.append(x)
        return output


class MobileNetV3Stem(nn.Module):
    def __init__(
        self,
        dim_in,
        beta_inv,
    ):
        super(MobileNetV3Stem, self).__init__()
        self.num_pathways = len(dim_in)
        self.beta_inv = beta_inv
        
        # Construct the stem layer.
        self._construct_stem(dim_in)

    def _construct_stem(self, dim_in):
        for pathway in range(len(dim_in)):
            stem = Stem(
                dim_in[pathway],
                self.beta_inv[pathway]
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])
        return x


class Stem(nn.Module):
    def __init__(
        self,
        dim_in,
        beta_inv,
    ):
        super(Stem, self).__init__()
        self.beta_inv = beta_inv
        # Construct the stem layer.
        self._construct_stem(dim_in)

    def _construct_stem(self, dim_in):
        self.conv1 = nn.Conv2d(dim_in, 16//self.beta_inv, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16//self.beta_inv)
        self.hs1 = hswish()
        self.block1 = Block(3, 16//self.beta_inv, 16//self.beta_inv, 16//self.beta_inv, nn.ReLU(inplace=True), None, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hs1(x)
        x = self.block1(x)
        return x

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 18: (2, 2, 2, 2)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}

_RES_BLOCK_DIM_OUT = {
    "bottleneck_transform": (4, 8, 16, 32), 
    "basic_transform": (1, 2, 4, 8),
}

class FuseFastToSlow2D(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        n_segment,
        n_div,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow2D, self).__init__()
        self.alpha = alpha
        self.conv_f2s = TemporalShift(
            nn.Conv2d(
                dim_in,
                dim_in * fusion_conv_channel_ratio // alpha,
                kernel_size=[1, 1],
                stride=[1, 1],
                padding=[0, 0],
                bias=False,
            ), 
            n_segment=n_segment, 
            n_div=n_div
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio // alpha,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        nt, c, h, w = fuse.size()
        fuse = fuse.view(nt // self.alpha, c * self.alpha, h, w)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]
    
@MODEL_REGISTRY.register()
class SlowFast_MobileNetV3(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast_MobileNetV3, self).__init__()
        self.norm_module = nn.BatchNorm2d
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )
        
        self.n_segment = [
            cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA, 
            cfg.DATA.NUM_FRAMES
        ]
        
        self.s1 = MobileNetV3Stem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            beta_inv=[1, cfg.SLOWFAST.BETA_INV],
        )
        
        self.s1_fuse = FuseFastToSlow2D(
            16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
            n_segment=self.n_segment[1],
            n_div=cfg.TSM.FUSION_N_DIV[0],
        )
        
        self.s2 = MobilenetV3Stage(
            dim_in=[
                16 + 16 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA, 
                16
            ],
            beta_inv=[1, cfg.SLOWFAST.BETA_INV],
            stage_idx=2,
            n_segment=self.n_segment,
            n_div=cfg.TSM.N_DIV[0],
        )
        
        self.s2_fuse = FuseFastToSlow2D(
            24 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
            n_segment=self.n_segment[1],
            n_div=cfg.TSM.FUSION_N_DIV[1],
        )
        
        self.s3 = MobilenetV3Stage(
            dim_in=[
                24 + 24 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA, 
                24
            ],
            beta_inv=[1, cfg.SLOWFAST.BETA_INV],
            stage_idx=3,
            n_segment=self.n_segment,
            n_div=cfg.TSM.N_DIV[1],
        )
        
        self.s3_fuse = FuseFastToSlow2D(
            40 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
            n_segment=self.n_segment[1],
            n_div=cfg.TSM.FUSION_N_DIV[2],
        )
        
        self.s4 = MobilenetV3Stage(
            dim_in=[
                40 + 40 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA, 
                40
            ],
            beta_inv=[1, cfg.SLOWFAST.BETA_INV],
            stage_idx=4,
            n_segment=self.n_segment,
            n_div=cfg.TSM.N_DIV[2],
        )
        
        self.s4_fuse = FuseFastToSlow2D(
            112 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
            n_segment=self.n_segment[1],
            n_div=cfg.TSM.FUSION_N_DIV[3],
        )
        
        self.s5 = MobilenetV3Stage(
            dim_in=[
                112 + 112 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA, 
                112
            ],
            beta_inv=[1, cfg.SLOWFAST.BETA_INV],
            stage_idx=5,
            n_segment=self.n_segment,
            n_div=cfg.TSM.N_DIV[3],
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    960,
                    960 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    960,
                    960 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None, reshape=True):
        if reshape:
            # (N, C, T, H, W) -> (N*T, C, H, W)
            for pathway in range(self.num_pathways):
                n, c, t, h, w = x[pathway].size()
                x[pathway] = x[pathway].transpose(1, 2).contiguous().view(n*t, c, h, w)
        else:
            # (N, T*C, H, W) -> (N*T, C, H, W)
            x = [i.view((-1, 3) + i.size()[-2:]) for i in x]
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        # (N*T, C, H, W) -> (N, C, T, H, W)
        for pathway in range(self.num_pathways):
            nt, c, h, w = x[pathway].size()
            x[pathway] = x[pathway].view(nt // self.n_segment[pathway], self.n_segment[pathway], c, h, w).transpose(1, 2)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x

class KDStage(nn.Module):
    def __init__(
        self, 
        dim_in,
        dim_out,
        n_segment,
    ):
        super(KDStage, self).__init__()
        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                }
            )
            == 1
        )
        self.num_pathways = len(dim_in)
        self.n_segment = n_segment
        for pathway in range(self.num_pathways):
            kd = PWConv(dim_in[pathway], dim_out[pathway])
            self.add_module("pathway{}_kd".format(pathway), kd)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            m = getattr(self, "pathway{}_kd".format(pathway))
            x = m(x)
            nt, c, h, w = x.size()
            x = x.view(nt // self.n_segment[pathway], self.n_segment[pathway], c, h, w).transpose(1, 2)
            output.append(x)
        return output
    
class PWConv(nn.Module):
    def __init__(
        self, 
        dim_in,
        dim_out,
    ):
        super(PWConv, self).__init__()
        self.pwconv = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
@MODEL_REGISTRY.register()
class Teacher_SlowFast_MobileNetV3(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(Teacher_SlowFast_MobileNetV3, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self.cfg = cfg
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )
        
        (dim_out2, dim_out3, dim_out4, dim_out5) = _RES_BLOCK_DIM_OUT[cfg.RESNET.TRANS_FUNC]

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * dim_out2,
                width_per_group * dim_out2 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * dim_out2 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * dim_out2 + width_per_group * dim_out2 // out_dim_ratio,
                width_per_group * dim_out2 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * dim_out3,
                width_per_group * dim_out3 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * dim_out3 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * dim_out3 + width_per_group * dim_out3 // out_dim_ratio,
                width_per_group * dim_out3 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * dim_out4,
                width_per_group * dim_out4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * dim_out4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * dim_out4 + width_per_group * dim_out4 // out_dim_ratio,
                width_per_group * dim_out4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * dim_out5,
                width_per_group * dim_out5 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * dim_out5,
                    width_per_group * dim_out5 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * dim_out5,
                    width_per_group * dim_out5 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        feature2 = [F.normalize(x[0], dim=0), F.normalize(x[1], dim=0)]
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        feature3 = [F.normalize(x[0], dim=0), F.normalize(x[1], dim=0)]
        x = self.s4(x)
        x = self.s4_fuse(x)
        feature4 = [F.normalize(x[0], dim=0), F.normalize(x[1], dim=0)]
        x = self.s5(x)
        feature5 = [F.normalize(x[0], dim=0), F.normalize(x[1], dim=0)]
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x, [feature2, feature3, feature4, feature5]
    
@MODEL_REGISTRY.register()
class Student_SlowFast_MobileNetV3(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(Student_SlowFast_MobileNetV3, self).__init__()
        self.norm_module = nn.BatchNorm2d
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self.cfg = cfg
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )
        
        self.n_segment = [
            cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA, 
            cfg.DATA.NUM_FRAMES
        ]
        
        self.s1 = MobileNetV3Stem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            beta_inv=[1, cfg.SLOWFAST.BETA_INV],
        )
        self.s1_fuse = FuseFastToSlow2D(
            16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
            n_segment=self.n_segment[1],
            n_div=cfg.TSM.FUSION_N_DIV[0],
        )
        
        self.s2 = MobilenetV3Stage(
            dim_in=[
                16 + 16 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA, 
                16
            ],
            beta_inv=[1, cfg.SLOWFAST.BETA_INV],
            stage_idx=2,
            n_segment=self.n_segment,
            n_div=cfg.TSM.N_DIV[0],
        )
        self.s2_fuse = FuseFastToSlow2D(
            24 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
            n_segment=self.n_segment[1],
            n_div=cfg.TSM.FUSION_N_DIV[1],
        )
        self.s2_kd = KDStage(
            dim_in=[
                24 + 24 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA,
                24 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                320,
                32,
            ],
            n_segment=self.n_segment
        )
        
        self.s3 = MobilenetV3Stage(
            dim_in=[
                24 + 24 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA, 
                24
            ],
            beta_inv=[1, cfg.SLOWFAST.BETA_INV],
            stage_idx=3,
            n_segment=self.n_segment,
            n_div=cfg.TSM.N_DIV[1],
        )
        self.s3_fuse = FuseFastToSlow2D(
            40 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
            n_segment=self.n_segment[1],
            n_div=cfg.TSM.FUSION_N_DIV[2],
        )
        self.s3_kd = KDStage(
            dim_in=[
                40 + 40 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA,
                40 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                640,
                64,
            ],
            n_segment=self.n_segment
        )
        
        self.s4 = MobilenetV3Stage(
            dim_in=[
                40 + 40 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA, 
                40
            ],
            beta_inv=[1, cfg.SLOWFAST.BETA_INV],
            stage_idx=4,
            n_segment=self.n_segment,
            n_div=cfg.TSM.N_DIV[2],
        )
        self.s4_fuse = FuseFastToSlow2D(
            112 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
            n_segment=self.n_segment[1],
            n_div=cfg.TSM.FUSION_N_DIV[3],
        )
        self.s4_kd = KDStage(
            dim_in=[
                112 + 112 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA,
                112 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                1280,
                128,
            ],
            n_segment=self.n_segment
        )
        
        self.s5 = MobilenetV3Stage(
            dim_in=[
                112 + 112 // out_dim_ratio // cfg.SLOWFAST.ALPHA * cfg.SLOWFAST.ALPHA, 
                112
            ],
            beta_inv=[1, cfg.SLOWFAST.BETA_INV],
            stage_idx=5,
            n_segment=self.n_segment,
            n_div=cfg.TSM.N_DIV[3],
        )
        self.s5_kd = KDStage(
            dim_in=[
                960,
                960 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                2048,
                256,
            ],
            n_segment=self.n_segment
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    960,
                    960 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    960,
                    960 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None, reshape=True):
        if reshape:
            # (N, C, T, H, W) -> (N*T, C, H, W)
            for pathway in range(self.num_pathways):
                n, c, t, h, w = x[pathway].size()
                x[pathway] = x[pathway].transpose(1, 2).contiguous().view(n*t, c, h, w)
        else:
            # (N, T*C, H, W) -> (N*T, C, H, W)
            x = [i.view((-1, 3) + i.size()[-2:]) for i in x]
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        if self.cfg.TRAIN.ENABLE and self.cfg.KD.ENABLE:
            kd2 = self.s2_kd(x)
            feature2 = [F.normalize(kd2[0], dim=0), F.normalize(kd2[1], dim=0)]
        x = self.s3(x)
        x = self.s3_fuse(x)
        if self.cfg.TRAIN.ENABLE and self.cfg.KD.ENABLE:
            kd3 = self.s3_kd(x)
            feature3 = [F.normalize(kd3[0], dim=0), F.normalize(kd3[1], dim=0)]
        x = self.s4(x)
        x = self.s4_fuse(x)
        if self.cfg.TRAIN.ENABLE and self.cfg.KD.ENABLE:
            kd4 = self.s4_kd(x)
            feature4 = [F.normalize(kd4[0], dim=0), F.normalize(kd4[1], dim=0)]
        x = self.s5(x)
        if self.cfg.TRAIN.ENABLE and self.cfg.KD.ENABLE:
            kd5 = self.s5_kd(x)
            feature5 = [F.normalize(kd5[0], dim=0), F.normalize(kd5[1], dim=0)]
        # (N*T, C, H, W) -> (N, C, T, H, W)
        for pathway in range(self.num_pathways):
            nt, c, h, w = x[pathway].size()
            x[pathway] = x[pathway].view(nt // self.n_segment[pathway], self.n_segment[pathway], c, h, w).transpose(1, 2)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        if self.cfg.TRAIN.ENABLE and self.cfg.KD.ENABLE:
            return x, [feature2, feature3, feature4, feature5]
        else:
            return x
