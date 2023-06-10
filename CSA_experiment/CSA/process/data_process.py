import os
import torch
import numpy as np
import cv2

def slicing_image_label():
    root_dir='../data/Massachusetts Roads Dataset'
    with open(root_dir+'/'+'metadata.csv','r') as f:
        lines=f.readlines()[1:]
    count_train, count_test, count_val = 0, 0, 0
    for line in lines:
        image_id, spl, tiff_image_path, tif_label_path=line.split(',')[0],line.split(',')[1],line.split(',')[-4],line.split(',')[-3]

        img=cv2.imread(root_dir+'/'+tiff_image_path,1)
        label=cv2.imread(root_dir+'/'+tif_label_path,0)
        # img_p=cv2.copyMakeBorder(img,3,3,3,3,cv2.BORDER_CONSTANT,value=(0,0,0))
        # label_p=cv2.copyMakeBorder(label,3,3,3,3,cv2.BORDER_CONSTANT,value=0)
        # print(img_p.shape,label_p.shape)

        for i in range(3):
            for j in range(3):
                crop_name=image_id+'_'+str(i)+'_'+str(j)+'.jpg'
                crop_img=img[i*(512-18):i*(512-18)+512,j*(512-18):j*(512-18)+512,:]
                crop_label=label[i*(512-18):i*(512-18)+512,j*(512-18):j*(512-18)+512]
                if spl=='train':
                    print(spl)
                    if not os.path.exists(root_dir+'/crops/train_crops'):
                        os.mkdir(root_dir+'/crops/train_crops')
                    if not os.path.exists(root_dir + '/crops/train_labels_crops'):
                        os.mkdir(root_dir + '/crops/train_labels_crops')
                    cv2.imwrite(root_dir+'/crops/train_crops/'+crop_name,crop_img)
                    cv2.imwrite(root_dir+'/crops/train_labels_crops/'+crop_name,crop_label)
                    count_train+=1
                elif spl=='test':
                    print(spl)
                    if not os.path.exists(root_dir+'/crops/test_crops'):
                        os.mkdir(root_dir+'/crops/test_crops')
                    if not os.path.exists(root_dir + '/crops/test_labels_crops'):
                        os.mkdir(root_dir + '/crops/test_labels_crops')
                    cv2.imwrite(root_dir + '/crops/test_crops/' + crop_name, crop_img)
                    cv2.imwrite(root_dir + '/crops/test_labels_crops/' + crop_name, crop_label)
                    count_test+=1
                elif spl=='val':
                    print(spl)
                    if not os.path.exists(root_dir+'/crops/val_crops'):
                        os.mkdir(root_dir+'/crops/val_crops')
                    if not os.path.exists(root_dir + '/crops/val_labels_crops'):
                        os.mkdir(root_dir + '/crops/val_labels_crops')
                    cv2.imwrite(root_dir + '/crops/val_crops/' + crop_name, crop_img)
                    cv2.imwrite(root_dir + '/crops/val_labels_crops/' + crop_name, crop_label)
                    count_val+=1
    return (count_train,count_test,count_val)

count_train,count_test,count_val=slicing_image_label()
print(count_train,count_test,count_val)
