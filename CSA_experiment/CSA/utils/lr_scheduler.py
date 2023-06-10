import math

class LR_Scheduler(object):
    def __init__(self,mode,base_lr,num_epochs,iters_per_epoch=9900//8,lr_step=0,warmup_epochs=1):#此时iters_per_epoch为1246
        self.mode=mode
        self.lr=base_lr
        if mode=='step':
            assert lr_step
        self.lr_step=lr_step
        self.iters_per_epoch=iters_per_epoch
        self.N=num_epochs*iters_per_epoch
        self.epoch=-1
        self.warmup_iters=warmup_epochs*iters_per_epoch

    def __call__(self,optimizer,i,epoch):
        T=epoch * self.iters_per_epoch + i
        if self.mode=='cos':
            lr=0.5 * self.lr * (1.0 + math.cos(1.0 * T /self.N * math.pi))
        elif self.mode=='poly':
            lr=self.lr * pow((1.0 - 1.0 * T / self.N),3)#指数为1或2
        elif self.mode=='step':
            lr=self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        #预热
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr=lr*1.0*T / self.warmup_iters
        # if epoch > self.epoch:
        #     print("\n=>Epochs %i,learning_rate=%.4f"%(epoch,lr))
        assert lr >=0#若lr<0则报错
        self._adjust_learning_rate(optimizer,lr)

    def _adjust_learning_rate(self,optimizer,lr):
        if len(optimizer.param_groups)==1:
            optimizer.param_groups[0]['lr']=lr
        else:
            optimizer.param_groups[0]['lr']=lr
            for i in range(1,len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr']=lr * 10