import time
import os

import numpy as np
import torch
import cv2 as cv

from mmdet.apis import init_detector, inference_detector

config_file = r'E:\BaiduNetdiskDownload\7楼\防震锤锈蚀_thr0.9\faster_rcnn_r50_150e\faster_rcnn_r50.py'
checkpoint_file = r'E:\BaiduNetdiskDownload\7楼\防震锤锈蚀_thr0.9\faster_rcnn_r50_150e\epoch_30.pth'

if __name__ == '__main__':
    model = init_detector(config_file, checkpoint_file, device='cuda')
    model.eval()

    img_dir = r'L:\CustomDataset\Paper-OD-sh_damaged\JPEGImages/'
    imgs = os.listdir(img_dir)

    for i, img in enumerate(imgs):
        result = inference_detector(model, img_dir + img)  # result from model.simple_test
        model.show_result(img_dir + img, result, score_thr=0.90,
                                                out_file=r'L:\ret_dirs\rusted_sh\result_{}.jpg'.format(img.split('.')[0]))
        # try:
        #     result = inference_detector(model, img_dir + img) # result from model.simple_test
        #     model.show_result(img_dir + img, result, score_thr=0.90,
        #                       out_file=r'L:\ret_dirs\rusted_sh\result_{}.jpg'.format(img.split('.')[0]))
        # except AttributeError:
        #     continue

    print('done')
