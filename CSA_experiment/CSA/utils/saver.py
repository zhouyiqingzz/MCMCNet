import os
import shutil
import torch
import glob
from datetime import datetime
from collections import OrderedDict

class Saver(object):
    def __init__(self,args):
        self.args=args
        self.directory='run_'+self.args.dataset #run_DeepGlobe
        self.runs=sorted(glob.glob(os.path.join(self.directory,"experiment_*")))#run_DeepGlobe/experiment_***
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir=os.path.join(self.directory,'experiment_{}'.format(str(run_id)))#run_DeepGlobe/experiment_年月日_时分秒
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self,state,is_best,filename='checkpoint.pth.tar'):
        filename=os.path.join(self.experiment_dir,filename)#run_DeepGlobe/experiment_年月日_时分秒/checkpoint.pth.tar
        torch.save(state,filename)
        if is_best:
            best_pred=state['best_pred']
            with open(os.path.join(self.experiment_dir,'best_pred.txt'),'w') as f:#run_DeepGlobe/experiment_年月日_时分秒/best_pred.txt
                f.write(str(best_pred))
            if self.runs:
                previous_miou=[0.0]
                for run in self.runs:
                    run_id=run.split('t_')[-1]
                    path=os.path.join(self.directory,'experiment_{}'.format(str(run_id)),'best_pred.txt')#run_DeepGlobe/experiment_年月日_时分秒/best_pred.txt
                    if os.path.exists(path):
                        with open(path,'r') as f:
                            miou=float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou=max(previous_miou)
                if best_pred>max_miou:
                    shutil.copyfile(filename,os.path.join(self.directory,'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile=os.path.join(self.experiment_dir,'parameters.txt')
        f=open(logfile,'w')
        p=OrderedDict()
        p['dataset']=self.args.dataset
        p['backbone']=self.args.backbone
        p['out_stride']=self.args.out_stride
        p['lr']=self.args.lr
        p['lr_scheduler']=self.args.lr_scheduler
        p['loss_type']=self.args.loss_type
        p['epoch']=self.args.epochs
        p['base_size']=self.args.base_size
        p['crop_size']=self.args.crop_size
        for key,val in p.items():
            f.write(key+':'+str(val)+'\n')
        f.close()

