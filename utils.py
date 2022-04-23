# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import numpy as np


def cal_score(pred,gt,eps=0.):
    gt_mean = gt.mean((-1,-2,-3))
    weight = (1./(gt_mean + eps))**0.5
    diff = np.abs(pred-gt)
    diff = diff.mean((-1,-2,-3))
    return (diff*weight).mean()


