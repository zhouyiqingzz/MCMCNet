import torch
import math
import numpy as np

class LabelGuessor(object):
    def __init__(self, thresh): #default:0.95
        self.thresh = thresh

    def __call__(self, model, ims, balance, delT):
        is_train = model.training
        with torch.no_grad():
            model.train()
            all_probs = []
            logits, _, _ = model(ims)
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)

            mask = torch.ones_like(lbs,dtype=torch.float)
            labels, counts = np.unique(lbs.cpu(),return_counts=True)
            mxCount = max(counts)

            if balance > 0:
                if balance == 1 or balance == 4:
                    idx = (mask == 0.0)
                else:
                    idx = (scores > self.thresh)
                    delT = 0
                if mxCount > 0:
                    ratios = [x/mxCount for x in counts]
                    for i in range(len(labels)):
                        tmp = (scores*(lbs==labels[i]).float()).ge(self.thresh - delT*(1-ratios[i])) # Which elements
                        idx = idx | tmp
                    idx = idx.int()

                    if balance > 2:
                        labels, counts = np.unique(lbs.cpu() * idx.cpu(),return_counts=True)
                    ratio = torch.zeros_like(mask,dtype=torch.float)
                    for i in range(len(labels)):
                        ratio += ((1/counts[i])*(lbs==labels[i]).float())  # Magnitude of mask elements
                    Z = torch.sum(mask * idx)
                    mask = ratio * idx
                    if Z > 0:
                        mask = Z * mask / torch.sum(mask)

            lbs_1 = lbs * idx
            # print(lbs_1.shape, mask.shape)

        if is_train:
            model.train()
        else:
            model.eval()

        return lbs_1.detach(), idx, mask

class LabelGuessor_ske(object):
    def __init__(self, thresh): #default:0.95
        self.thresh = thresh

    def __call__(self, model, ims, balance, delT):
        is_train = model.training
        with torch.no_grad():
            model.train()
            all_probs = []
            _, logits, _ = model(ims)
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)

            mask = torch.ones_like(lbs,dtype=torch.float)
            labels, counts = np.unique(lbs.cpu(),return_counts=True)
            mxCount = max(counts)

            if balance > 0:
                if balance == 1 or balance == 4:
                    idx = (mask == 0.0)
                else:
                    idx = (scores > self.thresh)
                    delT = 0
                if mxCount > 0:
                    ratios = [x/mxCount for x in counts]
                    for i in range(len(labels)):
                        tmp = (scores*(lbs==labels[i]).float()).ge(self.thresh - delT*(1-ratios[i])) # Which elements
                        idx = idx | tmp
                    idx = idx.int()

                    if balance > 2:
                        labels, counts = np.unique(lbs.cpu() * idx.cpu(),return_counts=True)
                    ratio = torch.zeros_like(mask,dtype=torch.float)
                    for i in range(len(labels)):
                        ratio += ((1/counts[i])*(lbs==labels[i]).float())  # Magnitude of mask elements
                    Z = torch.sum(mask * idx)
                    mask = ratio * idx
                    if Z > 0:
                        mask = Z * mask / torch.sum(mask)

            lbs_1 = lbs * idx
            # print(lbs_1.shape, mask.shape)

        if is_train:
            model.train()
        else:
            model.eval()

        return lbs_1.detach(), idx, mask