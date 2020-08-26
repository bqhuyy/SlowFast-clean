import numpy as np
import torch
import os
import pandas as pd
import torch.nn as nn
import torchfunc
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.models import build_model 
import slowfast.models.video_model_builder as v
import slowfast.models.resnet_helper as resnet_helper
import slowfast.models.resnet2d_helper as resnet2d_helper
import slowfast.datasets.cv2_transform as cv2_transform
from slowfast.config.defaults import get_cfg
from slowfast.utils.env import setup_environment
from slowfast.datasets import transform as transform
import torch
from fvcore.common.registry import Registry
from slowfast.config.defaults import get_cfg

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""

logger = logging.get_logger(__name__)

class ActivityRecognition(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.setup_env()
        self.load_model()
#         self.setup_data()

    def setup_env(self):
        #setup_environment()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        #logger.info("Model Config")
        #logger.info(self.cfg)
        self.model = build_model(self.cfg)
        self.model.eval()
        #if du.is_master_proc():
        misc.log_model_info(self.model, self.cfg, is_train = False)

        model_path = cfg.TRAIN.CHECKPOINT_FILE_PATH
        assert os.path.exists(model_path), "%s. Model Path Not Found" % model_path

        cu.load_checkpoint(
            model_path,
            self.model,
            self.cfg.NUM_GPUS > 1)#
#            None,
#            inflation = False,
#            convert_from_caffe2 = self.cfg.TRAIN.CHECKPOINT_TYPE == "caffe2"
#        )

    def setup_data(self):
        self.data_mean = torch.tensor(self.cfg.DATA.MEAN)
        self.data_std = torch.tensor(self.cfg.DATA.STD)
        self.image_size = self.cfg.DATA.TEST_CROP_SIZE
        self.num_frames = self.cfg.DATA.NUM_FRAMES
        self.alpha = self.cfg.SLOWFAST.ALPHA
        self.num_gpus = self.cfg.NUM_GPUS
        self.min_scale, self.max_scale, self.crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
        #print(self.min_scale, self.max_scale, self.crop_size)
        #labels_df = pd.read_csv(cfg.DATA.LABEL_FILE_PATH)
        #self.labels = labels_df['name'].values
        #self.labels = np.array(["fighting", "lifting", "picking", "shopping", "stealing"])
        #self.labels = np.array(["normal", "stealing"])
        self.labels = np.array(["normal", "fighting"])

    def preprocess_data(self, images):
        # T x H x W x C
        images = [image[:, :, ::-1] for image in images]
        images = np.concatenate([image[np.newaxis] for image in images])
        images = torch.from_numpy(images).float()
        images = images / 255.
        images -= self.data_mean
        images /= self.data_std
        images = images.permute(3, 0, 1, 2) # -> C x T x H x W
        images, _ = transform.random_short_side_scale_jitter(images, self.min_scale, self.max_scale)
        images, _ = transform.uniform_crop(images, self.crop_size, 0)
        images = images.unsqueeze(0)
        # Fast Path Way
        index = torch.linspace(0, images.shape[2] - 1, self.num_frames).long()
        fast_pathway = torch.index_select(images, 2, index)
        # Slow Path Way
        index = torch.linspace(0, fast_pathway.shape[2] - 1, fast_pathway.shape[2]//self.alpha).long()
        slow_pathway = torch.index_select(fast_pathway, 2, index)
        inputs = [slow_pathway, fast_pathway]

        for i in range(2):
            inputs[i] = inputs[i].to(self.device)
        return inputs


    def forward(self, images):
        inputs = self.preprocess_data(images)
        outputs = self.model(inputs)

        if self.num_gpus > 1:
            outputs = du.all_gather(outputs)[0]
        outputs = outputs.squeeze().cpu().detach().numpy()
        index = np.argsort(outputs)[::-1]

       
        return outputs[index], self.labels[index]


if __name__ == "__main__":
    import time
    import sys
    import cv2

    config_path = sys.argv[1] #"configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(config_path)

    name = cfg.MODEL.MODEL_NAME
    
    print('Create model')
    if cfg.MODEL.MODEL_NAME=='SlowFast_MobileNetV3':
        model = v.SlowFast_MobileNetV3(cfg)
    elif cfg.MODEL.MODEL_NAME=='SlowFast':
        model = v.SlowFast(cfg)
    elif cfg.MODEL.MODEL_NAME=='SlowFast_TSM':
        model = v.SlowFast_TSM(cfg)
    elif cfg.MODEL.MODEL_NAME=='ResNet2D':
        model = v.ResNet2D(cfg)
    elif cfg.MODEL.MODEL_NAME=='ResNet':
        model = v.ResNet(cfg)
    elif cfg.MODEL.MODEL_NAME=='ResNetTSM':
        model = v.ResNetTSM(cfg)
     
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    
    model.eval()
    print(model)
    total = 0
    if cfg.MODEL.MODEL_NAME=='ResNet2D':
        shift_buffer = [
            [
                torch.zeros([1, 8, 56, 56]).cuda(),
                torch.zeros([1, 32, 56, 56]).cuda(),
                torch.zeros([1, 32, 56, 56]).cuda(),
            ],
            [
                torch.zeros([1, 32, 56, 56]).cuda(),
                torch.zeros([1, 64, 28, 28]).cuda(),
                torch.zeros([1, 64, 28, 28]).cuda(),
                torch.zeros([1, 64, 28, 28]).cuda(),
            ],
            [
                torch.zeros([1, 64, 28, 28]).cuda(),
                torch.zeros([1, 128, 14, 14]).cuda(),
                torch.zeros([1, 128, 14, 14]).cuda(),
                torch.zeros([1, 128, 14, 14]).cuda(),
                torch.zeros([1, 128, 14, 14]).cuda(),
                torch.zeros([1, 128, 14, 14]).cuda(),
            ],
            [
                torch.zeros([1, 128, 14, 14]).cuda(),
                torch.zeros([1, 256, 7, 7]).cuda(),
                torch.zeros([1, 256, 7, 7]).cuda(),
                torch.zeros([1, 256, 7, 7]).cuda(),
            ]
        ]
        with torchfunc.Timer() as timer:
            for i in range(100):
                _, shift_buffer = model([torch.rand(1,3,1,224,224).cuda()], shift_buffer=shift_buffer)
    #             model([torch.rand(1,3,8,224,224), torch.rand(1,3,32,224,224)])
                total += timer.checkpoint()
        print(total, total/100.0)
        
    elif cfg.MODEL.MODEL_NAME=='ResNet' or cfg.MODEL.MODEL_NAME=='ResNetTSM':
        with torchfunc.Timer() as timer:
            for i in range(100):
                model([torch.rand(1,3,8,224,224).cuda()])
                total += timer.checkpoint()
        print(total, total/100.0)
    
    
    
#     misc.log_model_info(model, cfg, is_train = False)

#     num_groups = cfg.RESNET.NUM_GROUPS
#     width_per_group = cfg.RESNET.WIDTH_PER_GROUP
#     dim_inner = num_groups * width_per_group
#     if '2D' in config_path or '2d' in config_path:
#         s2 = resnet2d_helper.ResStage(
#             dim_in=[64],
#             dim_out=[64 * 4],
#             dim_inner=[dim_inner],
#             temp_kernel_sizes=[[1]],
#             stride=cfg.RESNET.SPATIAL_STRIDES[0],
#             num_blocks=[3],
#             num_groups=[num_groups],
#             num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
#             nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
#             nonlocal_group=cfg.NONLOCAL.GROUP[0],
#             nonlocal_pool=cfg.NONLOCAL.POOL[0],
#             instantiation=cfg.NONLOCAL.INSTANTIATION,
#             trans_func_name=cfg.RESNET.TRANS_FUNC,
#             stride_1x1=cfg.RESNET.STRIDE_1X1,
#             inplace_relu=cfg.RESNET.INPLACE_RELU,
#             dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
#             norm_module=nn.BatchNorm2d,
#         )
#     elif '3D' in config_path or '3d' in config_path::
#         s2 = resnet_helper.ResStage(
#             dim_in=[64],
#             dim_out=[64 * 4],
#             dim_inner=[dim_inner],
#             temp_kernel_sizes=[[1]],
#             stride=cfg.RESNET.SPATIAL_STRIDES[0],
#             num_blocks=[3],
#             num_groups=[num_groups],
#             num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
#             nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
#             nonlocal_group=cfg.NONLOCAL.GROUP[0],
#             nonlocal_pool=cfg.NONLOCAL.POOL[0],
#             instantiation=cfg.NONLOCAL.INSTANTIATION,
#             trans_func_name=cfg.RESNET.TRANS_FUNC,
#             stride_1x1=cfg.RESNET.STRIDE_1X1,
#             inplace_relu=cfg.RESNET.INPLACE_RELU,
#             dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
#             norm_module=nn.BatchNorm3d,
#         )
#     s2 = s2.cuda(device=cur_device)
#     s2.eval()
#     import torchfunc
#     print(s2)
#     for _ in range(10):
#         if '2D' in config_path or '2d' in config_path:
#             with torchfunc.Timer() as timer:
#                 s2([torch.rand(32,64,56,56).cuda()])
#                 print(timer.checkpoint())
#         else:
#             with torchfunc.Timer() as timer:
#                 s2([torch.rand(1,64,32,56,56).cuda()])
#                 print(timer.checkpoint())
