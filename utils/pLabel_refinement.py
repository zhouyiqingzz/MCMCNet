import numpy as np
import cv2
from skimage import filters
import torch
import torch.nn as nn

def assign_labels(pred, pred1, round=0):
    th_ske = [0.1, 0.5]
    th_rd = [0.7, 0.9]
    th_ske1 = [0.05, 0.1]
    th_rd1 = [0.6, 0.9]

    # if round == 0:
    #     th_ske = [0.08, 0.12]
    #     th_rd = [0.7, 0.9]

    #     th_ske1 = [0.05, 0.1]
    #     th_rd1 = [0.6, 0.9]
    # else:
    #     # R2 new values
    #     th_ske = [0.3, 0.5]
    #     th_rd = [0.2, 0.7]

    #     th_ske1 = [0.1, 0.5]
    #     th_rd1 = [0.4, 0.7]

    pred = pred.squeeze(0).detach().cpu().numpy()
    pred1 = pred1.squeeze(0).detach().cpu().numpy()

    pred1_ = pred1.copy()
    row, col = np.where((pred1 > th_ske[0]) & (pred1 < th_ske[1]))
    pred1[pred1 >= th_ske[1]] = 255
    pred1[pred1 < th_ske[1]] = 0
    pred1[row, col] = 50

    # print("done")
    pred_ = pred.copy()
    rows, cols = np.where((pred > th_rd[0]) & (pred < th_rd[1]))
    pred[pred >= th_rd[1]] = 255
    pred[pred < th_rd[0]] = 0
    pred[rows, cols] = 50

    rows, cols = np.where((pred_ > th_rd1[0]) & (pred_ < th_rd1[1]))
    pred_[pred_ >= th_rd1[1]] = 255
    pred_[pred_ < th_rd1[0]] = 0
    pred_[rows, cols] = 50

    row, col = np.where((pred1_ > th_ske1[0]) & (pred1_ < th_ske1[1]))
    pred1_[pred1_ >= th_ske1[1]] = 255
    pred1_[pred1_ < th_ske1[1]] = 0
    pred1_[row, col] = 50

    kernel_r = np.ones((3, 3), np.uint8)
    kernel_s = np.ones((3, 3), np.uint8)

    # pred_ = cv2.dilate(pred_, kernel_r, iterations=3)
    # pred_ = cv2.erode(pred_, kernel_r, iterations=3)
    # pred_[pred_ != pred] = 50

    if round == 0:
        hystt = filters.apply_hysteresis_threshold(pred_, 1, 254)
        hystt1 = filters.apply_hysteresis_threshold(pred1_, 1, 254)

        pred[hystt] = 255
        pred1[hystt1] = 255
        pred_ = cv2.dilate(pred.astype(np.uint8), kernel_r, iterations=2)
    # else:

    #     pred_ = cv2.dilate(pred_, kernel_r, iterations=2)

    pred_[pred_ != pred] = 0 #此处值有待考察
    pred = pred_

    pred = pred / 255
    pred1 = pred1 /255
    return pred, pred1


if __name__ == '__main__':
    pred = np.random.randn(512, 512).astype(np.uint8)
    pred1 = np.random.randn(512, 512).astype(np.uint8)
    pred, pred1 = assign_labels(pred, pred1, round=0)
    print(pred, pred1)