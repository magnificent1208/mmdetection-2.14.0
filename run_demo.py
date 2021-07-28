import time
import os
import shutil

import numpy as np
import torch
import cv2 as cv

from mmdet.apis import init_detector, inference_detector

config_file = r'L:\7楼\防震锤损坏_thr0.99\faster_rcnn_r50\faster_rcnn_r50_bak.py'
checkpoint_file = r'L:\7楼\防震锤损坏_thr0.99\faster_rcnn_r50\epoch_90.pth'
# img_orig = r'L:\7楼\防震锤损坏_thr0.99\img_orig'
count_limit = 50
score_thr = 0.5

if __name__ == '__main__':
    model = init_detector(config_file, checkpoint_file, device='cuda')
    model.eval()

    img_dir = r'L:\7楼\防震锤损坏_thr0.99\img_orig'
    imgs = os.listdir(img_dir)

    count = 0
    for i, img in enumerate(imgs):
        try:
            result = inference_detector(model, os.path.join(img_dir,img))  # result from model.simple_test
            bboxes = np.vstack(result)
            for item in bboxes[:, -1]:
                if item > float(score_thr) and count <= count_limit:
                    model.show_result(os.path.join(img_dir,img), result, score_thr=score_thr,
                                      out_file=r'L:\7楼\防震锤损坏_thr0.99/img_ret/result_{}.jpg'.format(
                                          img.split('.')[0]))
                    # shutil.copyfile(os.path.join(img_dir, img), os.path.join(img_orig, img))

        except AttributeError:
            continue

        count += 1
        print(count)
        if count > count_limit:
            break
    print('done')
