import numpy as np
from skimage import morphology
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2


def generate_save_skeleton(label_root, skeleton_save_root):
    label_names = os.listdir(label_root)
    for label_name in label_names:
        label_path = label_root + label_name
        label = Image.open(label_path).convert('L')  # 替换为您的图像路径
        label = np.array(label).astype(np.float32)  # 将图像转换为 0-255 的灰度图像
        # 使用skeletonize提取骨干
        tmp_label = np.copy(label)
        tmp_label /= 255
        tmp_label[tmp_label > 0.5] = 1
        tmp_label[tmp_label < 0.5] = 0
        print(tmp_label)
        skeleton_label = morphology.skeletonize(tmp_label).astype(np.int64)*255
        print(np.max(skeleton_label),np.min(skeleton_label))
        skeleton_label = Image.fromarray(skeleton_label.astype(np.uint8))
        skeleton_label.save(skeleton_save_root + label_name)

def generate_save_edge(label_root, edge_save_root):
    # 读取黑白图像
    label_names = os.listdir(label_root)
    for label_name in label_names:
        label_path = label_root + label_name
        image = cv2.imread(label_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 对图像进行阈值处理
        _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # 查找轮廓
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
        #保存轮廓图
        cv2.imwrite(edge_save_root + label_name, contour_image)


if __name__ == '__main__':
    train_dir_root = '/home/arsc/tmp/pycharm_project_639/CSA/data/Massachusetts/crops/'
    train_label_root = train_dir_root + 'train_labels_crops/'
    train_edge_root = train_dir_root + 'train_edges_crops/'
    val_dir_root = '/home/arsc/tmp/pycharm_project_639/CSA/data/Massachusetts/crops/'
    val_label_root = val_dir_root + 'val_labels_crops/'
    val_edge_root = val_dir_root + 'val_edges_crops/'

    generate_save_edge(train_label_root, train_edge_root)
    generate_save_edge(val_label_root, val_edge_root)
