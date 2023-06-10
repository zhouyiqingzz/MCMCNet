import cv2
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# label_path='Massachusetts Roads Dataset/tiff/test_labels/10378780_15.tif'
# a=torch.tensor([[1,2,3],[4,5,6],[7,8,9]],dtype=torch.float)
# a=torch.unsqueeze(a,0)
# b=torch.tensor([[1.0,1,1],[1,1,0],[1,0,1]])
# b=torch.unsqueeze(b,0)
# print(a.shape,b.shape)
# criterion=nn.BCELoss()
# loss=criterion(F.softmax(a,dim=1),b)
# print(loss)
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()
# for i in range(10):
#     writer.add_scalar('quadratic', i**2, global_step=i)
#     writer.add_scalar('exponential', 2**i, global_step=i)

def statistic_information(root_dir='data/Massachusetts Roads Dataset'):
    with open(root_dir+'/'+'metadata.csv') as f:
        lines=f.readlines()[1:]
    imgs_0,imgs_1,imgs_2=[],[],[]
    for line in lines:
        image_id, spl, tiff_image_path, tif_label_path=line.split(',')[0],line.split(',')[1],line.split(',')[-4],line.split(',')[-3]

        img=cv2.imread(root_dir+'/'+tiff_image_path,1)
        img_n=img/255.0
        imgs_0.extend(list(img_n[:, :, 0].reshape(1500*1500)))
        imgs_1.extend(list(img_n[:, :, 1].reshape(1500*1500)))
        imgs_2.extend(list(img_n[:, :, 2].reshape(1500*1500)))
        print(len(imgs_0),len(imgs_1),len(imgs_2))

    imgs_0_mean,imgs_1_mean,imgs_2_mean=np.mean(np.array(imgs_0)),np.mean(np.array(imgs_1)),np.mean(np.array(imgs_2))
    imgs_0_std,imgs_1_std,imgs_2_std=np.std(np.array(imgs_0)),np.std(np.array(imgs_1)),np.std(np.array(imgs_2))
    print(imgs_0_mean,imgs_1_mean,imgs_2_mean)
    print(imgs_0_std,imgs_1_std,imgs_2_std)

# statistic_information()
print(max([1,2,4]))