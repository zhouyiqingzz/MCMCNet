import os
import argparse
from tqdm import tqdm
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.autograd import Variable as V
from models.Our_Model import DARNet
from datasets.Massachusetts_dataset import massachusetts_dataset
from datasets.CHN6_dataset import chn6_dataset
from datasets.DeepGlobe_dataset import deepglobe_dataset
from torch.utils.data.dataloader import DataLoader
from utils.options import Opts
from utils.metrics import Evaluator
import torch.utils.data.distributed as ds
from models.EMA_Update import update_ema_variables
from utils.saver import Saver
import gc
gc.collect()
#可见设置，环境变量使得指定设备对CUDA应用可见
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1"
# os.environ['LOCAL_RANK']="0,1"
torch.backends.cudnn.enabled = True #寻找适合硬件的最佳算法,加上这一行运行时间能缩短很多!!!
torch.backends.cudnn.deterministic = True #由于计算中有随机性，每次网络前馈结果略有差异。设置该语句来避免这种结果波动
torch.backends.cudnn.benchmark = True #为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速

torch.cuda.empty_cache()  #释放显存
torch.cuda.empty_cache()  #释放显存
torch.cuda.empty_cache()  #释放显存
# torch.autograd.set_detect_anomaly(True)#异常调试，实际运行时要设置为False，否则会费时

import logging

logging.basicConfig(filename='distributed_log', level=logging.INFO)
logger = logging.getLogger(__name__)
# 记录信息
logger.info('Starting distributed training...')


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.opts = Opts()
        # Define Saver
        self.saver = Saver(args)
        # Define Dataloader
        if args.dataset == 'Massachusetts':
            data_dir_root = '/home/arsc/tmp/pycharm_project_639/CSA/data/Massachusetts/crops/'#/root/tmp/pycharm_project_968/CDCL/train.py
            labeled_train_dataset, unlabeled_train_dataset, val_dataset = massachusetts_dataset(dir_root=data_dir_root,
                                                                                                image_size=args.image_size,
                                                                                                labeled_ratio=args.label_ratio)
        elif args.dataset == 'CHN6':
            data_dir_root = '/home/arsc/tmp/pycharm_project_639/CSA/data/CHN6/'
            labeled_train_dataset, unlabeled_train_dataset, val_dataset = chn6_dataset(dir_root=data_dir_root,
                                                                                                image_size=args.image_size,
                                                                                                labeled_ratio=args.label_ratio)
        elif args.dataset == 'DeepGlobe':
            data_dir_root = '/home/arsc/tmp/pycharm_project_639/CSA/data/DeepGlobe/train/'
            labeled_train_dataset, unlabeled_train_dataset, val_dataset = deepglobe_dataset(dir_root=data_dir_root,
                                                                                       image_size=args.image_size,
                                                                                       labeled_ratio=args.label_ratio)
        self.labeled_train_sampler = ds.DistributedSampler(labeled_train_dataset)  # 保证一个batch里的数据被均摊到每个进程上，每个进程都能获取到不同的数据
        self.unlabeled_train_sampler = ds.DistributedSampler(unlabeled_train_dataset)
        self.labeled_train_loader = DataLoader(dataset=labeled_train_dataset, batch_size=args.batch_size, num_workers=2,
                                           pin_memory=True, shuffle=False, drop_last=True, sampler=self.labeled_train_sampler)
        self.unlabeled_train_loader = DataLoader(dataset=unlabeled_train_dataset, batch_size=args.batch_size, num_workers=2,
                                            pin_memory=True, shuffle=False, drop_last=True, sampler=self.unlabeled_train_sampler)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=4, num_workers=2, pin_memory=True,
                                      shuffle=False, drop_last=False)

        # Define Model
        self.model = DARNet(self.opts).cuda()
        # 部分模块参数不进行反向传播
        for ema_param in self.model.net_ema.parameters():
            ema_param.requires_grad = False
        # for ema_param in self.model.projector_ema.parameters():
        #     ema_param.requires_grad = False

        self.model = nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True, broadcast_buffers=False)  # 模型分布式多显卡并行,find_unused参数表明若有forward返回值不进行backward，那也不需要在不同进程之间进行通信
        # device_ids=[args.local_rank],output_device=args.local_rank,
        #Define Optimizer
        # self.optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, self.model.parameters()), lr=args.lr, momentum=args.momentum,
        #                                  weight_decay=args.weight_decay, nesterov=args.nesterov)  # 内斯特罗夫（人名)#filter返回满足指定条件的迭代器
        self.optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, self.model.parameters()), lr=args.lr)
        # Define Evaluator
        self.evaluator = Evaluator(num_class=2)
        self.ious, self.mious, self.precisions, self.recalls, self.f1s, self.accuracys, self.acc_classes, self.kappas = \
            [], [], [], [], [], [], [], []

        # self.pLabel_refinement = LabelGuessor(thresh=0.95)
        self.best_IoU = 0.0

    def adjust_lr(self, init_lr, i, max_iters):
        lr_ = init_lr * pow((1.0 - i / max_iters), 0.9) - 0.0001
        if lr_ >= 0.0002:
            lr_ -= 0.0002
        if lr_ <= 0.000005:
            lr_ = 0.000005
        if len(self.optimizer.param_groups)==1:
            self.optimizer.param_groups[0]['lr'] = lr_ #长度为7的字典,包括[‘params’, ‘lr’, ‘betas’, ‘eps’, ‘weight_decay’, ‘amsgrad’, ‘maximize’]这7个参数
        else:
            self.optimizer.param_groups[0]['lr'] = lr_
            for i in range(1,len(self.optimizer.param_groups)):
                self.optimizer.param_groups[i]['lr']=10 * lr_

    def training(self, epoch):
        if epoch % 2 == 0:
            self.labeled_train_sampler.set_epoch(epoch)  # 每个epoch时让训练集采样呈随机状态，且随机种子依赖于epoch
        else:
            self.unlabeled_train_sampler.set_epoch(epoch)
        self.model.to(args.device)
        self.model.train()
        train_loss = 0.0
        tbar = tqdm(zip(self.labeled_train_loader, self.unlabeled_train_loader))

        for i, ((labeled_img, labeled_seg_mask, labeled_ske_mask), (unlabeled_img, _, _)) in enumerate(tbar):
            labeled_seg_mask, labeled_ske_mask = labeled_seg_mask.unsqueeze(1), labeled_ske_mask.unsqueeze(1) #(4,1,512,512)
            labeled_img, labeled_seg_mask, labeled_ske_mask, unlabeled_img = V(labeled_img).cuda(non_blocking=True), \
                V(labeled_seg_mask).cuda(non_blocking=True),V(labeled_ske_mask).cuda(non_blocking=True), V(unlabeled_img).cuda(non_blocking=True)
            self.optimizer.zero_grad()
            loss_tot, loss_contra, loss_conform = self.model(labeled_img=labeled_img, labeled_seg_mask=labeled_seg_mask, labeled_ske_mask=labeled_ske_mask,
                                  unlabeled_img=unlabeled_img, mode='semi')
            loss_tot.backward()
            self.optimizer.step()
            train_loss += loss_tot.item()
            lr_now = self.optimizer.param_groups[0]['lr']
            tbar.set_description("epoch:%d, Train_loss:%.4f, Contra_loss:%.4f, Conform_loss:%.4f, lr:%.6f" % (epoch, train_loss / (i + 1), loss_contra, loss_conform, lr_now))
        if epoch >= 0:
            self.validation(epoch)
        update_ema_variables(self.model.module.net_ema, self.model.module.net_main, epoch, alpha=0.9)  # 分布式寻找模块时要加module
        self.adjust_lr(init_lr=args.lr, i=epoch, max_iters=args.epochs)

    def validation(self, epoch):
        self.model.eval() #该命令会让model.training=False
        self.evaluator.reset()
        tbar=tqdm(self.val_loader)
        for i, (imgs, labels, skeleton) in enumerate(tbar):
            labels, skeleton = labels.unsqueeze(1), skeleton.unsqueeze(1)
            imgs, labels, skeleton = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True), skeleton.cuda(non_blocking=True)
            with torch.no_grad():
                labeled_seg_pred = self.model(labeled_img=imgs, labeled_seg_mask=labels, labeled_ske_mask=skeleton,
                                                                                 unlabeled_img=None, mode='val')
            # 输出通道是2时
            preds = torch.argmax(labeled_seg_pred.data.cpu(), dim=1, keepdim=True)
            preds = preds.detach().cpu().numpy().astype(float)
            labels_n = labels.cpu().numpy().astype(float)
            # 输出通道数是1时
            # preds = labeled_seg_pred.data.detach().cpu().numpy()
            # labels_n = labels.cpu().numpy()
            # preds[preds >= 0.5] = 1
            # preds[preds < 0.5] = 0

            # Add batch sample into evaluator
            self.evaluator.add_batch(labels_n, preds)

        # Fast test during the training
        val_IoU = self.evaluator.Intersection_over_Union()
        val_Precision = self.evaluator.Pixel_Precision()
        val_Recall = self.evaluator.Pixel_Recall()
        val_F1 = self.evaluator.Pixel_F1()
        val_Kappa = self.evaluator.kappa_score()

        print('Validation:')
        print("Val>>> IoU:{}, Precision:{}, Recall:{}, F1:{}, Kappa:{}".format(val_IoU, val_Precision, val_Recall, val_F1, val_Kappa))

        # 保存模型参数文件和度量指标
        # args.is_best = False
        metric_str = str(epoch) + ': ' + str(val_F1) + ' ' + str(val_IoU) + ' ' + str(val_Kappa)
        args.is_best = False
        if epoch >= 120 and val_IoU > self.best_IoU and args.is_best:
            self.best_IoU = val_IoU
            self.saver.save_checkpoint({
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_IoU,
            }, epoch, metric_str)
        # record_file = os.path.join(args.save_dir, args.dataset, args.dataset + '_metric_' + str(int(args.label_ratio*100)) + '.txt')
        # with open(record_file, 'a') as f:
        #     f.write(str(val_F1) + ' ' + str(val_IoU) + ' ' + str(val_Kappa) + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CSA Training')
    parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')#不同版本对local_rank中下划线要求不同
    parser.add_argument('--backbone', type=str, default='resnet34', help='backbone network name(default:resnet34)')
    parser.add_argument('--out-stride', type=int, default=8, help='network output stride(default:8)')
    parser.add_argument('--dataset', type=str, default='Massachusetts', help='dataset name(DeepGlobe or Massachusetts or CHN6)')
    parser.add_argument('--checkname', type=str, default='first_run', help='set the checkpoint name')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads(default:16)')#进程数=节点数*显卡数
    parser.add_argument('--image-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--loss-type', type=str, default='con_ce', help='loss func type')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch(default:0)')
    parser.add_argument('--batch-size', type=int, default=4, help='input batch size for training(default:16)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate(default:0.001)')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='lr scheduler mode(default:poly)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum(default:0.9)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='wheter use nesterov(default:False)')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval(default:1)')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--ft', action='store_true', default=False, help='finetune on differnet dataset')
    parser.add_argument('--scheduler-step', type=int, default=10,
                        help='scheduler adjust while training much iters(default:10)')
    parser.add_argument('--world-size', type=int, default=2)#进程数=机器数*机器上显卡数，可同时进行的任务数
    parser.add_argument('--gpus', type=int, default=2)#机器上显卡数
    parser.add_argument('--nodes', type=int, default=2)#机器数
    parser.add_argument('--step_epoch',type=int,default=2)#每经过5个epoch后学习率调整
    parser.add_argument('--cuda',type=bool,default=True)
    parser.add_argument('--model-name', type=str, default='our_model' , help='Ours or DLinkNet or Others')
    parser.add_argument('--save-step', type=int, default=1) #保存参数文件的间隔数
    parser.add_argument('--is-best', type=bool, default=False)  # 与training函数的False形成对比，决定了后面saver需不需要保存参数文件
    parser.add_argument('--device')
    parser.add_argument('--label-ratio', type=float, default=0.2) #标签率
    parser.add_argument('--save-dir', type=str, default='/home/arsc/tmp/pycharm_project_698/DA_Road/run_model_experiments/our_model')  # 标签率
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
    args = parser.parse_args()

    torch.distributed.init_process_group(backend='nccl',init_method='env://')#设置通信方式和环境变量#用于初始化GPU通信方式(NCCL:实现GPU通信)和参数的获取方式
    local_rank=torch.distributed.get_rank()
    print(f"Start running basic DDP example on rank {local_rank}.")
    args.local_rank=local_rank
    args.world_size=torch.cuda.device_count()#2
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    args.device=device
    print(args.device)
    torch.manual_seed(1)
    trainer = Trainer(args)

    for epoch in range(trainer.args.start_epoch,trainer.args.epochs):
        trainer.training(epoch)
        if epoch >= 150:
            break
        # trainer.validation(epoch)
        # print("Val MaxScores:")
        # print("iou:{},miou:{},precision:{},recall:{},acc_class:{},f1:{}".format(max(trainer.ious),max(trainer.mious),max(trainer.precisions),
        #                                                                         max(trainer.recalls),max(trainer.acc_classes),max(trainer.f1s)))

