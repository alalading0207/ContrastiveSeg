from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.loss.loss_helper import FSAuxCELoss, FSAuxRMILoss, FSCELoss
from lib.utils.tools.logger import Logger as Log


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.temperature = self.configer.get('contrast', 'temperature')   # temperature = 0.1
        self.base_temperature = self.configer.get('contrast', 'base_temperature')   # base_temperature = 0.07

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']   # -1

        self.max_samples = self.configer.get('contrast', 'max_samples')   # 1024
        self.max_views = self.configer.get('contrast', 'max_views')       # 100

    def _hard_anchor_sampling(self, X, y_hat, y):   # X, y_hat, y --> embed [1,32768,256], labels, predict  [1,32768]
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]   # this y = this labels = [32768]
            this_classes = torch.unique(this_y)   # 识别独立不重复元素  [6]
            this_classes = [x for x in this_classes if x != self.ignore_label]  # 去除ignore_label    [5]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views] # 这步筛选该类点数 > max_views=100的类
            #  先让label == 这个类this_classes[x]的给值true, 再获得是true的位置的索引号， 再计算是true的个数
            classes.append(this_classes)   # bs的第一张图的有效类为: 2,5,7,8
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None
        # 设置一个bs中最多样例点个数为max_samples=1024    1024//4 = 256   计算出每张图的每类平均可使用多少样例点
        n_view = self.max_samples // total_classes    
        n_view = min(n_view, self.max_views)    # 每张图的 每类 最多可使用  max_views个样例点   只能小不能更多  （如果n_view < max_views）那么肯定也达不到max_samples

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()   # [总classes数，每类样例点个数，256]
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size): 
            this_y_hat = y_hat[ii]       # 该bs中 第ii张影像的 label    [32768]
            this_y = y[ii]               # 该bs中 第ii张影像的 predict  [32768]
            this_classes = classes[ii]   # 该bs中 第ii张影像的 类别个数

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()  # label是但predict不是这个类(cls_id)的点被认为是 hard sample
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()  # label和predict都是这个类(cls_id)  的点的索引位置

                num_hard = hard_indices.shape[0]   # hard sample的个数
                num_easy = easy_indices.shape[0]
                # num_hard 和 num_easy 需要一半一半  便被称为分段感知 hard sample
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:    # hard和easy点都多
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:                             # hard点多，而easy点 < n_view/2时
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:                             # easy点多，而hard点 < n_view/2时
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)      # 返回一个 0 到 n-1 的整数的随机排列
                hard_indices = hard_indices[perm[:num_hard_keep]]      # 取hard点  [76,1]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]      # 取easy点  [24,1]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)    # x:embed  在embed中取第ii个图像的索引位置为indices 的点
                y_[X_ptr] = cls_id   # cls_id代表该类
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):   # feats_ [4, 100, 256]    labels_ [4] 
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)   # [4, 1]
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()   # 一个横着2578一个竖着2578 取是否相等 则[4,4]的对角线都为1

        contrast_count = n_view  # 在0维上 对100个切片相加  以便之后计算4个类上的第一个点   进而一一计算这400个点   所以contrast_feature[400, 256]    每个点有256的长度向量
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)   # torch.unbind对某一个维度进行长度为1的切片  切开100(n_view)个切片，每个切片的大小为[4,256]

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)   # (i*i)/tau    [400, 400]    对角线为10.0001\9.9998\9.999\10.0002等 不统一
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # [400,1]
        logits = anchor_dot_contrast - logits_max.detach()   # detach后被认为是减了一个常数   每列都减最大数：对角线为0，其他值变为负数    因为对角线是自己和自己的矩阵乘

        mask = mask.repeat(anchor_count, contrast_count)   # 对张量进行重复扩充 mask.repeat(100,100) 行重复100倍 列重复100倍:  [400, 400]
        neg_mask = 1 - mask   

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)   # 将src中数据根据index中的索引按照dim的方向进行填充 ：对角线为0 其他都为1
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask    # i与i-的exp   这一类的锚点和其他类的锚点算exp
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)   # i与i+的exp

        log_prob = logits - torch.log(exp_logits + neg_logits)   # log(exp(logits))=logits

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)   # 除以所有锚点的个数

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()   # labels增加一维 [1, 1, 512, 1024]
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')  # [1, 1, 128, 256]
        labels = labels.squeeze(1).long()  # labels减少一维 [1, 128, 256]
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)   # 对labels进行深拷贝，并view为二维向量  [1, 32768]
        predict = predict.contiguous().view(batch_size, -1) # 对predict进行深拷贝，并view为二维向量  [1, 32768]
        feats = feats.permute(0, 2, 3, 1)  # [1, 256, 128, 256] --> [1, 128, 256, 256]
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])  # 对embed进行深拷贝，并view为二维向量  [1, 32768, 256]
        # feats_ [4, 100, 256] 4类，每类100个点，每个点的embed为256      labels_ [4] 该类的类值
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)   # 获得 hard sampling

        loss = self._contrastive(feats_, labels_)
        return loss


class ContrastCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastCELoss, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = self.configer.get('contrast', 'loss_weight')
        self.use_rmi = self.configer.get('contrast', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)

        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "embed" in preds

        seg = preds['seg']
        embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)

        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training


class ContrastAuxCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastAuxCELoss, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = self.configer.get('contrast', 'loss_weight')   # 0.1
        self.use_rmi = self.configer.get('contrast', 'use_rmi')           # false

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSAuxCELoss(configer=configer)    # FSAuxCELoss

        self.contrast_criterion = PixelContrastLoss(configer=configer)   # PixelContrastLoss

    def forward(self, preds, target, with_embed=False):  # preds: {'seg': out, 'seg_aux': out_aux, 'embed': emb}
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "seg_aux" in preds
        assert "embed" in preds

        seg = preds['seg']     # [1, 19, 128, 256]
        seg_aux = preds['seg_aux']
        embedding = preds['embed']   # [1, 256, 128, 256]

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)            # [1, 19, 512, 1024]  seg
        pred_aux = F.interpolate(input=seg_aux, size=(h, w), mode='bilinear', align_corners=True)    # [1, 19, 512, 1024]  seg_aux
        loss = self.seg_criterion([pred_aux, pred], target)

        _, predict = torch.max(seg, 1)  # 在dim=1的维度上返回最大值     
        # embedding  [1, 256, 128, 256]     target [1, 512, 1024]      predict  [1, 128, 256]
        loss_contrast = self.contrast_criterion(embedding, target, predict)     

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training
