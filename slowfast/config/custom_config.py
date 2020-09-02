#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""

from fvcore.common.config import CfgNode

def add_custom_config(_C):
    # Knowledge distillation
    _C.KD = CfgNode()

    # If True enable KD, else skip KD.
    _C.KD.ENABLE = False

    # Teacher's config
    _C.KD.CONFIG = ""

    # Alpha
    _C.KD.ALPHA = 0.95

    # Temperature
    _C.KD.TEMPERATURE = 6

    # Teacher's config
    _C.KD.CONFIG = "configs/Kinetics/SLOWFAST_8x8_R50.yaml"
    
    # Path to the checkpoint to load the initial weight.
    _C.KD.CHECKPOINT_FILE_PATH = ""

    # Checkpoint types include `caffe2` or `pytorch`.
    _C.KD.CHECKPOINT_TYPE = "pytorch"
    
    _C.KD.TEACHER_TRANS_FUNC = 'bottleneck_transform'
    
    # TSM
    _C.TSM = CfgNode()

    # n_div for TSM
    _C.TSM.N_DIV = [[8, 8], [8, 8], [8, 8], [8, 8]]

    # fusion n_div
    _C.TSM.FUSION_N_DIV = [8, 8, 8, 8]
    
    _C.TEST.CLASS_LIST = 'filenames/kinetics-40'
    