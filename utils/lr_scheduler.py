import math

class LR_Scheduler(object):
    def __init__(self,mode, base_lr, num_epochs, iters_per_epoch=1174,lr_step=0,warmup_epochs=0):#此时iters_per_epoch为1246
        self.mode=mode
        self.lr=base_lr
        # if mode=='step':
        #     assert lr_step
        self.lr_step=lr_step
        self.iters_per_epoch=iters_per_epoch
        self.N=num_epochs*iters_per_epoch
        self.epoch=-1
        self.warmup_iters=warmup_epochs*iters_per_epoch

    def __call__(self,optimizer,i,epoch):
        T=epoch * self.iters_per_epoch + i
        if self.mode=='cos':
            lr=0.5 * self.lr * (1.0 + math.cos(1.0 * T /self.N * math.pi))-0.0002
            if lr<=0.0001:
                lr=0.0001
        elif self.mode=='poly':
            lr = self.lr * pow((1.0 - 1.0 * T / self.N),3) - 0.0003 #指数为1或2,指数越大变得越小#pow只是调节的系数，可大致看成目前的epoch与总的epochs比值
            if lr > 0.005 and lr <= 0.01:
                lr = lr-0.0005
            elif lr > 0.003 and lr <= 0.004:
                lr = lr-0.0002
            elif lr > 0.002 and lr <= 0.003:
                lr = lr-0.0001
            elif lr > 0.001 and lr <= 0.002 :
                lr = lr-0.00005
            # elif lr <= 0.001 and lr > 0.0001:
            #     lr = lr-0.00001
            # elif lr <= 0.00001:  #无论哪个if分支都要保证lr>=0
            #     lr = 0.00001
        elif self.mode=='step':
            lr = self.lr - 0.0001
            if lr <= 0.0:
                lr = 0.0001
        else:
            raise NotImplemented
        #预热
        # if self.warmup_iters > 0 and T < self.warmup_iters: #由于上述warmup_epochs为0，所以warmup_iters也为0，不会进入该分支
        #     lr=lr*1.0*T / self.warmup_iters

        # if epoch > self.epoch:
        #     print("\n=>Epochs %i,learning_rate=%.4f"%(epoch,lr))
        if lr < 0.00001:
            lr = 0.00001
        assert lr >=0#若lr<0则报错
        self._adjust_learning_rate(optimizer,lr)

    def _adjust_learning_rate(self,optimizer,lr):
        if len(optimizer.param_groups)==1:
            optimizer.param_groups[0]['lr']=lr #长度为7的字典,包括[‘params’, ‘lr’, ‘betas’, ‘eps’, ‘weight_decay’, ‘amsgrad’, ‘maximize’]这7个参数
        else:
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1,len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr']=10 * lr