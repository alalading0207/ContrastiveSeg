##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Microsoft Research
## Author: RainbowSecret, LangHuang, JingyiXie, JianyuanGuo
## Copyright (c) 2019
## yuyua@microsoft.com
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Our approaches including FCN baseline, HRNet, OCNet, ISA, OCR
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FCN baseline 
from lib.models.nets.fcnet import FcnNet

# OCR
from lib.models.nets.ocrnet import SpatialOCRNet, ASPOCRNet
from lib.models.nets.ideal_ocrnet import IdealSpatialOCRNet, IdealSpatialOCRNetB, IdealSpatialOCRNetC, IdealGatherOCRNet, IdealDistributeOCRNet

# HRNet
from lib.models.nets.hrnet import HRNet_W48, HRNet_W48_CONTRAST
from lib.models.nets.hrnet import HRNet_W48_OCR, HRNet_W48_OCR_B, HRNet_W48_OCR_B_HA, HRNet_W48_OCR_CONTRAST, HRNet_W48_MEM

# OCNet
from lib.models.nets.ocnet import BaseOCNet, AspOCNet

# ISA Net
from lib.models.nets.isanet import ISANet

# CE2P
from lib.models.nets.ce2pnet import CE2P_OCRNet, CE2P_IdealOCRNet, CE2P_ASPOCR

# SegFix
from lib.models.nets.segfix import SegFix_HRNet

from lib.utils.tools.logger import Logger as Log

from lib.models.nets.deeplab import DeepLabV3, DeepLabV3Contrast

from lib.models.nets.ms_ocrnet import MscaleOCR

SEG_MODEL_DICT = {
    # SegFix
    'segfix_hrnet': SegFix_HRNet,
    # OCNet series
    'base_ocnet': BaseOCNet,
    'asp_ocnet': AspOCNet,
    # ISA Net
    'isanet': ISANet,
    # OCR series
    'spatial_ocrnet': SpatialOCRNet,
    'spatial_asp_ocrnet': ASPOCRNet,
    # OCR series with ground-truth   
    'ideal_spatial_ocrnet': IdealSpatialOCRNet,
    'ideal_spatial_ocrnet_b': IdealSpatialOCRNetB,
    'ideal_spatial_ocrnet_c': IdealSpatialOCRNetC, 
    'ideal_gather_ocrnet': IdealGatherOCRNet,
    'ideal_distribute_ocrnet': IdealDistributeOCRNet,
    # HRNet series
    'hrnet_w48': HRNet_W48,
    'hrnet_w48_ocr': HRNet_W48_OCR,
    'hrnet_w48_ocr_b': HRNet_W48_OCR_B,
    # CE2P series
    'ce2p_asp_ocrnet': CE2P_ASPOCR,
    'ce2p_ocrnet': CE2P_OCRNet,
    'ce2p_ideal_ocrnet': CE2P_IdealOCRNet, 
    # baseline series
    'fcnet': FcnNet,
    'hrnet_w48_contrast': HRNet_W48_CONTRAST,
    'hrnet_w48_ocr_contrast': HRNet_W48_OCR_CONTRAST,
    'hrnet_w48_mem': HRNet_W48_MEM,
    'deeplab_v3': DeepLabV3,
    'deeplab_v3_contrast': DeepLabV3Contrast,
    'ms_ocr': MscaleOCR,
    'hrnet_w48_ocr_b_ha': HRNet_W48_OCR_B_HA,
}


class ModelManager(object):
    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get('network', 'model_name')  # 获取model name

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)  # SEG_MODEL_DICT[model_name]= HRNet_W48_OCR
        # model = HRNet_W48_OCR(self.configer)

        return model
