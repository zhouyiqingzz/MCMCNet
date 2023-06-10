import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from datasets.MassachuRoad_dataset import MyDataset
import numpy as np
from utils.losses import dice_bce_loss
from tqdm import tqdm
from models.CSA_model import CSANet
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
import datasets.custom_transforms as tr

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        # self.summary = TensorboardSummary(directory='experiments')
        # self.writer = self.summary.create_summary()
        # Define Dataloader
        train_dataset = MyDataset('train')
        test_dataset=MyDataset('val')
        # val_dataset=MyDataset('val')
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       pin_memory=False, drop_last=True, num_workers=args.workers)
        self.test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,
                                    pin_memory=False, drop_last=False, num_workers=args.workers)
        # Define Model
        self.model = CSANet().cuda()
        # Define optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay, nesterov=args.nesterov)  # 内斯特罗夫（人名)
        # Define Criterion
        self.criterion = dice_bce_loss()
        # self.model,self.optimizer=model,optimizer
        self.model = self.model.cuda()
        # Define Evaluator
        self.evaluator = Evaluator(num_class=2)
        # Resuming from checkpoint
        self.best_pred = 0.0
        self.ious,self.mious,self.precisions,self.recalls,self.f1s,self.accuracys,self.acc_classes=\
            [],[],[],[],[],[],[]
        if args.resume is not None:
            if not os.path.exists(args.resume):
                raise RuntimeError("=>no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.cuda = True
            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.optimizer.param_groups[0]['lr'] = args.lr
            print("=> loader checkpoint '{} (epoch {})' with lr :{}".format(args.resume, args.start_epoch,
                                                                            self.optimizer.param_groups[0]['lr']))
        # Define LR_Scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, self.optimizer.param_groups[0]['lr'],
                                      args.epochs, len(self.train_loader))
        if args.ft:  # 微调
            args.start_epoch = 0

    def training(self, epoch):
        self.model.to(device)
        self.model.train()
        self.evaluator.reset()

        num_train_imgs = len(self.train_loader)
        tbar = tqdm(self.train_loader)
        train_loss = 0.0

        for i, (imgs, labels) in enumerate(tbar):
            labels = labels.unsqueeze(1)
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = self.model(imgs)

            self.optimizer.zero_grad()
            loss = self.criterion(labels, outputs)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            lr_now=self.optimizer.param_groups[0]['lr']
            tbar.set_description("epoch:%d, Train_loss:%.3f, lr:%.4f" % (
                epoch, train_loss / (i + 1), lr_now))
            # self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_train_imgs * epoch)
            # save checkpoint every iter of epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
            # preds = outputs.data.cpu().numpy()
            # targets_c = labels.cpu().numpy()
            # preds[preds >= 0.5] = 1
            # preds[preds < 0.5] = 0
            # self.evaluator.add_batch(targets_c, preds)
        trainer.validation(epoch)
        self.scheduler(self.optimizer, i, epoch)#epoch和args.epoch不一样，args.epoch一开始加载后就固定不动了
        # Acc = self.evaluator.Pixel_Accuracy()
        # Acc_class = self.evaluator.Pixel_Accuracy_Class()
        # IoU = self.evaluator.Intersection_over_Union()
        # mIoU = self.evaluator.Mean_Intersection_over_Union()
        # Precision = self.evaluator.Pixel_Precision()
        # Recall = self.evaluator.Pixel_Recall()
        # F1 = self.evaluator.Pix_F1()
        # print("Acc:{},Acc_class:{},IoU:{},mIoU:{},Precision:{},Recall:{},F1:{}".format(Acc,
        #                                                                                Acc_class, IoU, mIoU,
        #                                                                                Precision, Recall, F1))

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        val_loss=0.0
        num_img_tr = len(self.test_loader)

        for i, (imgs, labels) in enumerate(tbar):
            labels = labels.unsqueeze(1)
            imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.model(imgs)

            loss = self.criterion(labels, outputs)
            val_loss += loss.item()

            preds = outputs.data.cpu().numpy()
            labels_n = labels.cpu().numpy()
            # Add batch sample into evaluator
            preds[preds > 0.5] = 1
            preds[preds <= 0.5] = 0
            self.evaluator.add_batch(labels_n, preds)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()
        self.ious.append(IoU)
        self.mious.append(mIoU)
        self.precisions.append(Precision)
        self.recalls.append(Recall)
        self.f1s.append(F1)
        self.accuracys.append(Acc)
        self.acc_classes.append(Acc_class)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d, Loss: %.3f]' % (epoch, i * self.args.batch_size + imgs.data.shape[0],val_loss))
        print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
              .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1))

        new_pred = IoU #作为best_score的度量指标
        if new_pred > self.best_pred:
            is_best = True#与training函数的False形成对比
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CSA Training')
    parser.add_argument('--backbone', type=str, default='resnet34', help='backbone network name(default:resnet34)')
    parser.add_argument('--out-stride', type=int, default=8, help='network output stride(default:8)')
    parser.add_argument('--dataset', type=str, default='DeepGlobe', help='dataset name(default:DeepGlobe)')
    parser.add_argument('--checkname', type=str, default='first_run', help='set the checkpoint name')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads(default:16)')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--loss-type', type=str, default='con_ce', help='loss func type')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch(default:0)')
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training(default:16)')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate(default:0.001)')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='lr scheduler mode(default:poly)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum(default:0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='w-dcay(default:5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='wheter use nesterov(default:False)')
    parser.add_argument('--resume', type=str,default=None,help='checkpoint path')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval(default:1)')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--ft', action='store_true', default=False, help='finetune on differnet dataset')
    parser.add_argument('--scheduler_step', type=int, default=1000,
                        help='scheduler adjust while training much iters(default:1000)')

    args = parser.parse_args()

    torch.manual_seed(1)
    trainer=Trainer(args)
    for epoch in range(trainer.args.start_epoch,trainer.args.epochs):
        trainer.training(epoch)
        # trainer.validation(epoch)
        print("Max Scores:")
        print("iou:{},miou:{},precision:{},recall:{},acc_class:{},f1:{}".format(max(trainer.ious),max(trainer.mious),max(trainer.precisions),
                                                                                max(trainer.recalls),max(trainer.acc_classes),max(trainer.f1s)))

