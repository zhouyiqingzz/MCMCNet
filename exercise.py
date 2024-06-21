import numpy as np
from skimage import morphology
from PIL import Image
import matplotlib.pyplot as plt

# 读取单通道灰度图像
image = Image.open('/home/arsc/tmp/pycharm_project_503/CDCL/data/Massachusetts/crops/test_labels_crops/10378780_15_0_0.jpg').convert('L')  # 替换为您的图像路径
image = np.array(image).astype(np.float32)   # 将图像转换为 0-255 的灰度图像
# 使用skeletonize提取骨干
tmp_image = np.copy(image)
tmp_image/=255
tmp_image[tmp_image>0.5]=255
tmp_image[tmp_image<0.5]=0
skeleton = morphology.skeletonize(tmp_image).astype(np.int64)#*255
print(type(skeleton))
print(np.sum(skeleton))
print(skeleton)
print(np.max(skeleton),np.min(skeleton))

# 显示原始图像和提取的骨干
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')

axes[1].imshow(skeleton, cmap='gray')
axes[1].set_title('Skeleton')

plt.tight_layout()
plt.show()
