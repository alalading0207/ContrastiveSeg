##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret, JingyiXie, LangHuang
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import os
import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from lib.utils.distributed import get_world_size, get_rank, is_distributed
# from mmcv.cnn import get_model_complexity_info


class Trainer(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """

    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()   # 计算和存储 time平均值与当前值
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()  # 计算和存储 train_ losses 平均值与当前值
        self.val_losses = AverageMeter()    
        self.seg_visualizer = SegVisualizer(configer)  # 只初始化一下SegVisualizer  
        self.loss_manager = LossManager(configer)   
        self.module_runner = ModuleRunner(configer) # 送入device、导入模型、模型传参、保存模型、固定BN、修剪梯度、获取lr
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)  # 数据预处理、数据导入模式、loader器
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)  # ？
        self.evaluator = get_evaluator(configer, self)  # ？  Using evaluator: StandardEvaluator

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None

        self._init_model()

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()   # 初始化网络

        try:
            flops, params = get_model_complexity_info(self.seg_net, (3, 512, 512))   # mmcv的代码
            split_line = '=' * 30
            print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
                split_line, (3, 512, 512), flops, params))
            print('!!!Please be cautious if you use the results in papers. '
                  'You may need to check if all ops are supported and verify that the '
                  'flops computation is correct.')
        except:
            pass

        self.seg_net = self.module_runner.load_net(self.seg_net)   # module_runner  上传网络到gpu上

        Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))  # None
        if self.configer.get('optim', 'group_method') == 'decay': 
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = self._get_parameters()      # 设置学习率参数

        # 初始化  优化器  学习率调整策略
        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)
        # 数据集导入
        self.train_loader = self.data_loader.get_trainloader()  # 获取 dataloader : trainloader
        self.val_loader = self.data_loader.get_valloader()
        # loss声明
        self.pixel_loss = self.loss_manager.get_seg_loss() 
        if is_distributed():
            self.pixel_loss = self.module_runner.to_device(self.pixel_loss)   # 上传loss到gpu上

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self):   #网络不同部分的学习率设置
        bb_lr = []
        nbb_lr = []
        fcn_lr = []
        params_dict = dict(self.seg_net.named_parameters())   # 网络的参数名称字典
        for key, value in params_dict.items():
            if 'backbone' in key:       # 如果是backbone型的参数  就把参数名放入bb_lr列表
                bb_lr.append(value)     # 例如：module.backbone.conv1.weight
            elif 'aux_layer' in key or 'upsample_proj' in key:
                fcn_lr.append(value)
            else:
                nbb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': fcn_lr, 'lr': self.configer.get('lr', 'base_lr') * 10},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]  # nbb_mult默认为1
        return params

    def __train(self):
        """
          Train function of every epoch during train phase. 每个epoch的train函数
        """
        self.seg_net.train()
        self.pixel_loss.train()
        start_time = time.time()
        scaler = torch.cuda.amp.GradScaler()

        # if "swa" in self.configer.get('lr', 'lr_policy'):    # 学习率策略为'swa'时才执行
        #     normal_max_iters = int(self.configer.get('solver', 'max_iters') * 0.75)
        #     swa_step_max_iters = (self.configer.get('solver', 'max_iters') - normal_max_iters) // 5 + 1

        if hasattr(self.train_loader.sampler, 'set_epoch'):   # train_loader为前面获得的train数据加载器
            self.train_loader.sampler.set_epoch(self.configer.get('epoch'))  # 此步作用：分布式训练时需要在导入数据之前就suffle

        # 开始训练
        for i, data_dict in enumerate(self.train_loader):    # data_dict里有img\labelmap\meta\name
            self.optimizer.zero_grad()                            # SGD.zero_grad()
            if self.configer.get('lr', 'metric') == 'iters':
                self.scheduler.step(self.configer.get('iters'))   # get('iters')是获取当前迭代次数   学习率向前一步
            else:
                self.scheduler.step(self.configer.get('epoch'))

            # if self.configer.get('lr', 'is_warm'):
            #     self.module_runner.warm_lr(
            #         self.configer.get('iters'),
            #         self.scheduler, self.optimizer, backbone_list=[0, ]
            #     )

            # 数据获得
            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)   # 从数据字典里剥离inputs 和 targets 
            self.data_time.update(time.time() - start_time)

            foward_start_time = time.time()
            # with torch.cuda.amp.autocast():  #运行网络
            outputs = self.seg_net(*inputs)
            self.foward_time.update(time.time() - foward_start_time)

            loss_start_time = time.time()
            if is_distributed():
                import torch.distributed as dist
                def reduce_tensor(inp):
                    """
                    Reduce the loss from all processes so that 
                    process with rank 0 has the averaged results.
                    """
                    world_size = get_world_size()
                    if world_size < 2:
                        return inp
                    with torch.no_grad():
                        reduced_inp = inp
                        dist.reduce(reduced_inp, dst=0)
                    return reduced_inp

                with torch.cuda.amp.autocast():
                    loss = self.pixel_loss(outputs, targets)
                    backward_loss = loss
                    display_loss = reduce_tensor(backward_loss) / get_world_size()
            else:  # 单卡 计算loss
                backward_loss = display_loss = self.pixel_loss(outputs, targets,
                                                               gathered=self.configer.get('network', 'gathered'))

            self.train_losses.update(display_loss.item(), batch_size)
            self.loss_time.update(time.time() - loss_start_time)

            backward_start_time = time.time()

            # backward_loss.backward()  梯度回传
            # self.optimizer.step()
            scaler.scale(backward_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            self.backward_time.update(time.time() - backward_start_time)

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')  # 迭代次数增加1

            # Print the log info & reset the states.打印时间和loss
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0 and \
                    (not is_distributed() or get_rank() == 0):   # 每10次一输出
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Forward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\t'
                         'Backward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\t'
                         'Loss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    self.configer.get('epoch'), self.configer.get('iters'),
                    self.configer.get('solver', 'display_iter'),   # 每几个step显示一次总时间
                    self.module_runner.get_lr(self.optimizer), batch_time=self.batch_time,  # 第3个参数显示学习率
                    foward_time=self.foward_time, backward_time=self.backward_time, loss_time=self.loss_time,
                    data_time=self.data_time, loss=self.train_losses))  # loss=self.train_losses   loss.val是？
                self.batch_time.reset()
                self.foward_time.reset()
                self.backward_time.reset()
                self.loss_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            # # save checkpoints for swa （swa微调策略时才使用）
            # if 'swa' in self.configer.get('lr', 'lr_policy') and \
            #         self.configer.get('iters') > normal_max_iters and \
            #         ((self.configer.get('iters') - normal_max_iters) % swa_step_max_iters == 0 or \
            #          self.configer.get('iters') == self.configer.get('solver', 'max_iters')):
            #     self.optimizer.update_swa()

            # 到最大迭代次数后退出该循环
            if self.configer.get('iters') == self.configer.get('solver', 'max_iters'):
                break

            # Check to val the current model.
            # if self.configer.get('epoch') % self.configer.get('solver', 'test_interval') == 0:
            if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:  # 每 test_interval 验证一次
                self.__val()

        self.configer.plus_one('epoch')

    def __val(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        self.pixel_loss.eval()
        start_time = time.time()
        replicas = self.evaluator.prepare_validaton()    # 获得None

        data_loader = self.val_loader if data_loader is None else data_loader   # data_loader 使用 val_loader
        # 验证开始
        for j, data_dict in enumerate(data_loader):
            if j % 10 == 0:
                if is_distributed(): dist.barrier()  # Synchronize all processes
                Log.info('{} images processed\n'.format(j))

    
            if self.configer.get('dataset') == 'lip': 
                (inputs, targets, inputs_rev, targets_rev), batch_size = self.data_helper.prepare_data(data_dict,
                                                                                                       want_reverse=True)
            else:   # 拆分字典   数据获得
                (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            with torch.no_grad():
                if self.configer.get('dataset') == 'lip':
                    inputs = torch.cat([inputs[0], inputs_rev[0]], dim=0)
                    outputs = self.seg_net(inputs)
                    if not is_distributed():
                        outputs_ = self.module_runner.gather(outputs)
                    else:
                        outputs_ = outputs
                    if isinstance(outputs_, (list, tuple)):
                        outputs_ = outputs_[-1]
                    outputs = outputs_[0:int(outputs_.size(0) / 2), :, :, :].clone()
                    outputs_rev = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), :, :, :].cloxne()
                    if outputs_rev.shape[1] == 20:
                        outputs_rev[:, 14, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 15, :, :]
                        outputs_rev[:, 15, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 14, :, :]
                        outputs_rev[:, 16, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 17, :, :]
                        outputs_rev[:, 17, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 16, :, :]
                        outputs_rev[:, 18, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 19, :, :]
                        outputs_rev[:, 19, :, :] = outputs_[int(outputs_.size(0) / 2):int(outputs_.size(0)), 18, :, :]
                    outputs_rev = torch.flip(outputs_rev, [3])
                    outputs = (outputs + outputs_rev) / 2.
                    self.evaluator.update_score(outputs, data_dict['meta'])

                elif self.data_helper.conditions.diverse_size:
                    if is_distributed():
                        outputs = [self.seg_net(inputs[i]) for i in range(len(inputs))]
                    else:
                        outputs = nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)

                    for i in range(len(outputs)):
                        loss = self.pixel_loss(outputs[i], targets[i].unsqueeze(0))
                        self.val_losses.update(loss.item(), 1)
                        outputs_i = outputs[i]
                        if isinstance(outputs_i, torch.Tensor):
                            outputs_i = [outputs_i]
                        self.evaluator.update_score(outputs_i, data_dict['meta'][i:i + 1])

                # 计算loss
                else:
                    outputs = self.seg_net(*inputs)  # 运行网络

                    try:   # 计算loss
                        loss = self.pixel_loss(
                            outputs, targets,
                            gathered=self.configer.get('network', 'gathered')
                        )
                    except AssertionError as e:
                        print(len(outputs), len(targets))

                    # if not is_distributed():
                    #     outputs = self.module_runner.gather(outputs)  # 从指定设备上的不同gpu收集张量  单机单卡用不到
                    self.val_losses.update(loss.item(), batch_size)  # 更新val_loss

                    # 如果是字典型输出
                    if isinstance(outputs, dict):   
                        try:
                            outputs = outputs['pred']
                        except:
                            outputs = outputs['seg']
                    self.evaluator.update_score(outputs, data_dict['meta'])   # ？

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        self.evaluator.update_performance()   # 计算metric   先进base

        self.configer.update(['val_loss'], self.val_losses.avg)
        # 保存模型
        self.module_runner.save_net(self.seg_net, save_mode='performance')   # save_mode 参数：保存最佳的模型
        self.module_runner.save_net(self.seg_net, save_mode='val_loss')
        cudnn.benchmark = True

        # Print the log info & reset the states.  记录时间
        self.evaluator.reduce_scores()
        if not is_distributed() or get_rank() == 0:
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            self.evaluator.print_scores()

        self.batch_time.reset()
        self.val_losses.reset()
        self.evaluator.reset()
        self.seg_net.train()
        self.pixel_loss.train()

    def train(self):                # 先判断train的类型：从头/断点
        # cudnn.benchmark = True
        # self.__val()
        if self.configer.get('network', 'resume') is not None:  # resume给值了的情况
            if self.configer.get('network', 'resume_val'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
                return
            elif self.configer.get('network', 'resume_train'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='train'))
                return
            # return

        if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
            self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
            return

        # 进入训练
        while self.configer.get('iters') < self.configer.get('solver', 'max_iters'):   # 当前迭代次数 0<40000
            self.__train()   

        # use swa to average the model
        if 'swa' in self.configer.get('lr', 'lr_policy'):
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.seg_net)

        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))

    def summary(self):
        from lib.utils.summary import get_model_summary
        import torch.nn.functional as F
        self.seg_net.eval()

        for j, data_dict in enumerate(self.train_loader):
            print(get_model_summary(self.seg_net, data_dict['img'][0:1]))
            return


if __name__ == "__main__":
    pass
