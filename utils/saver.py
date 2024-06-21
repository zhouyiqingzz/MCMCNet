import os
import shutil
import torch
import glob
from datetime import datetime
from collections import OrderedDict

class Saver(object):
    def __init__(self,args):
        self.args=args
        if not os.path.exists(os.path.join('run_model_experiments',args.model_name)):
            try:
                os.mkdir(os.path.join('run_model_experiments',args.model_name))
            except FileExistsError:
                pass
        self.directory=os.path.join('run_model_experiments', args.model_name, args.dataset)
        if not os.path.exists(self.directory):
            try:
                os.mkdir(self.directory)
            except FileExistsError:
                pass

        self.run_id=datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir=os.path.join(self.directory,'experiment_{}'.format(str(self.run_id)))#run_DeepGlobe/experiment_年月日_时分秒
        if os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)
        if not os.path.exists(self.experiment_dir):
            try:
                os.mkdir(self.experiment_dir)
            except FileExistsError:
                pass

    def save_checkpoint(self, state_dict, epoch, metric_str):#实际保存的有state_dict,optimizer,epoch,best_pred,后续需要state_dict['state_dict']来取得真正的模型权重
        model_filename = os.path.join(self.experiment_dir, str(epoch)+'_checkpoint.pth')#run_DeepGlobe/experiment_年月日_时分秒/checkpoint.pth.tar
        pred_filename = os.path.join(self.experiment_dir, str(epoch)+'_metric.txt')
        torch.save(state_dict, model_filename)#可以看一下加上该保存模型命令后多花费时间，仍要控制变量
        with open(pred_filename, 'a') as f:
            f.write(metric_str + '\n')
        # if is_best:
        #     pred_best_iou=state_dict['best_pred']
        #     path = os.path.join(self.directory, 'pred_best_iou.txt')  # run_DeepGlobe/experiment_年月日_时分秒/best_pred.txt
        #     if self.run_id:
        #         if os.path.exists(path):
        #             with open(path, 'r') as f1:
        #                 try:
        #                     max_iou = float(f1.readline())
        #                 except ValueError:
        #                     max_iou=0.0
        #         else:
        #             max_iou = 0.0
        #     if pred_best_iou >= max_iou:
        #         with open(path,'w') as f2:
        #             f2.write(str(pred_best_iou))
        #         f2.close()
        #         shutil.copyfile(model_filename, os.path.join(self.experiment_dir, 'model_best_now.pth'))#model_filename,pred_filename不存在也不会报错，只不过不执行复制操作
        #         shutil.copyfile(pred_filename, os.path.join(self.experiment_dir, 'pred_best_now.txt'))


    # def save_experiment_config(self):
    #     logfile=os.path.join(self.experiment_dir,'parameters.txt')
    #     f=open(logfile,'w')
    #     p=OrderedDict()
    #     p['dataset']=self.args.dataset
    #     p['backbone']=self.args.backbone
    #     p['out_stride']=self.args.out_stride
    #     p['lr']=self.args.lr
    #     p['lr_scheduler']=self.args.lr_scheduler
    #     p['loss_type']=self.args.loss_type
    #     p['epochs']=self.args.epochs
    #     p['image_size']=self.args.image_size
    #     p['crop_size']=self.args.crop_size
    #     for key,val in p.items():
    #         f.write(key+':'+str(val)+'\n')
    #     f.close()

