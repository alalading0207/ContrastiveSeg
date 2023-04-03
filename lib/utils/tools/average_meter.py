#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Utils to store the average and current value.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):   # val是value的意思
        self.val = val       # 当前值
        self.sum += val * n  # 总值
        self.count += n      # 当前数
        self.avg = self.sum / self.count   # 平均值
