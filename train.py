import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm

from utils import get_lr_scheduler, get_optimizer, linear_rampup,Med_Sup_Loss,DiceLoss, update_ema_variables, get_current_consistency_weight
from val import test_acdc,test_lidc
from utils import BCEDiceLoss

def Supervise(model, train_loader, test_loader, args):
    """有监督训练函数（未直接处理无标签数据，但为半监督训练提供基础逻辑）"""
    optimizer = get_optimizer(args=args, model=model)  # 初始化优化器
    lr_scheduler = get_lr_scheduler(args=args, optimizer=optimizer)  # 学习率调度器

    # 计算最大训练轮次（根据总迭代次数和每个epoch的batch数）
    max_epoch = args.total_itrs // len(train_loader) + 1
    args.logger.info("==============> max_epoch :{}".format(max_epoch))

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # 交叉熵损失（忽略255类）
    dice_loss = DiceLoss(args.num_classes)  # Dice损失（用于医学图像分割）

    model.train()  # 设置为训练模式
    cur_itrs = 0  # 当前迭代次数
    train_loss = 0.0  # 累计训练损失
    best_dice = 0.0  # 最佳Dice分数

    # 加载预训练模型（如果有）
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        state_dict = torch.load(args.ckpt)
        cur_itrs = state_dict["cur_itrs"]
        model = state_dict["model"]
        optimizer = state_dict["optimizer"]
        best_dice = state_dict["best_score"]

    # 开始训练循环
    for epoch in range(max_epoch):
        for i, (img_labeled, target_label) in enumerate(tqdm(train_loader)):
            cur_itrs += 1
            # 数据移至GPU
            img_labeled = img_labeled.to(args.device).float()
            target_label = target_label.to(args.device).long()

            # 前向传播（仅使用有标签数据）
            pseudo_labeled = model(img_labeled)
            
            # 计算损失（组合交叉熵和Dice损失）
            loss_ce = criterion(pseudo_labeled, target_label)
            loss_dice = dice_loss(pseudo_labeled, target_label.unsqueeze(1), softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice  # 加权求和

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            # 记录训练指标
            train_loss += loss.item()
            args.writer.add_scalar('supervise/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('supervise/lr', lr, cur_itrs)

            # 定期验证模型性能
            if cur_itrs % args.step_size == 0:
                dice, hd95 = test_acdc(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs)
                args.writer.add_scalar('supervise/{}_dice'.format(args.name), dice, cur_itrs)
                args.writer.add_scalar('supervise/{}_hd95'.format(args.name), hd95, cur_itrs)
                args.logger.info("epoch:{} \t dice:{:.5f} \t hd95:{:.5f} ".format(epoch, dice, hd95))

                # 保存最佳模型
                if dice > best_dice:
                    best_dice = dice
                    torch.save({
                        "cur_itrs": cur_itrs,
                        "best_dice": best_dice,
                        "model": model,
                        "optimizer": optimizer,
                        "lr_scheduler": lr_scheduler,
                    }, args.model_save_path)

                model.train()  # 验证后恢复训练模式

            # 终止条件
            if cur_itrs > args.total_itrs:
                return

        # 每轮次日志输出
        args.logger.info("Train [{}/{} ({:.0f}%)]\t loss: {:.5f} ".format(
            cur_itrs, args.total_itrs, 100. * cur_itrs / args.total_itrs, train_loss
        ))
        train_loss = 0  # 重置损失