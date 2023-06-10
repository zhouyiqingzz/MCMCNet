import numpy as np
import torch

class EarlyStopping(object):
    def __init__(self,patience=7,verbose=False,delta=0,save_path=""):
        self.patience=patience
        self.verbose=verbose
        self.counter=0
        self.best_score=None
        self.early_stop=False
        self.val_loss_min=np.Inf#无穷大
        self.delta=delta
        self.save_path=save_path

    def __call__(self,val_loss,model):
        score=-val_loss
        if self.best_score is None:#首次训练无加载模型
            self.best_score=score
            self.save_checkpoint(val_loss,model)
        elif score < self.best_score + self.delta:#没有改进时
            self.counter+=1
            if self.counter >= self.patience:
                print("EarlyStopping counter :{} out of patience :{}".format(self.counter,self.patience))
                self.early_stop=True
        else:#有较大进步时
            self.best_score=score
            self.save_checkpoint(val_loss,model)
            self.counter=0

    def save_checkpoint(self,val_loss,model):
        if self.verbose:
            print("Validation loss decreased from {} to {}".format(self.val_loss_min,val_loss))
        self.val_loss_min=val_loss

