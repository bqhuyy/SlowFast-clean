'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from slowfast.models.tsm_helper import TemporalShift


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
            stage = Stage(dim_in[pathway], beta_inv[pathway], self.stage_idx, self.n_segment[pathway], self.n_div[pathway])
            self.add_module("pathway{}_stage{}".format(pathway, self.stage_idx), stage)

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