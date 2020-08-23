"""
This script is used to generate OCR texts from image using ABCNet
This script use GPU Accelerator
"""

import numpy as np
import pandas as pd
import cv2
import glob
import os, sys

!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!gcc --version

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html

!git clone https://github.com/aim-uofa/AdelaiDet.git
os.chdir('AdelaiDet')

DATA_DIR = '/kaggle/input'
ROOT_DIR = '/kaggle/working'

sys.path.append(os.path.join(ROOT_DIR, 'AdelaiDet')) 

!python setup.py build develop

!wget -O tt_attn_R_50.pth https://cloudstor.aarnet.edu.au/plus/s/t2EFYGxNpKPUqhc/download
!ls -lh tt_attn_R_50.pth

os.chdir("demo")

import argparse
import multiprocessing as mp
import time
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

from tqdm.auto import tqdm, trange
from tqdm import tqdm_notebook

logger = setup_logger()

cfg = get_cfg()
cfg.merge_from_file("../configs/BAText/TotalText/attn_R_50.yaml")
cfg.merge_from_list(["MODEL.WEIGHTS", "../tt_attn_R_50.pth"])

confidence = 0.5
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence
cfg.freeze()

demo = VisualizationDemo(cfg)

# The variable inmages represents the files we want to perform OCR on
## First line for train data, second line for test data
inmages = glob.glob("/kaggle/input/shopee-product-detection-open/train/train/train/**/*.jpg")
# inmages = glob.glob("/kaggle/input/shopee-product-detection-open/test/test/test/*.jpg")

def decode_recognition(rec):
    CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

    s = ''
    for c in rec:
        c = int(c)
        if c < 95:
            s += CTLABELS[c]
        elif c == 95:
            s += u'口'
    return s

p = []
for path in tqdm(inmages):
    # use PIL, to be consistent with evaluation
    img = read_image(path, format="BGR")
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    tqdm.write(
        "{}: detected {} instances in {:.2f}s".format(
            path, len(predictions["instances"]), time.time() - start_time
        )
    )
    p.append([decode_recognition(p) for p in predictions["instances"].recs])
    
anott = pd.DataFrame({'path': inmages, 'annot': p})

## First line for train data, second line for test data
anott.to_csv("../../train_ocr.csv", index=False)
# anott.to_csv("../../test_ocr.csv", index=False)