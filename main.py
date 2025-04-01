"""
作者的主要内容
"""
import os.path
import numpy as np
import torch
from copy import deepcopy
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
from medpy import metric
from scipy.ndimage import zoom
from utils import loadyaml, _get_logger, mk_path, get_current_consistency_weight, DiceLoss, update_ema_variables, \
    linear_rampup
from utils import build_lr_scheduler, build_optimizer, Med_Sup_Loss, Dense_Loss, BoxMaskGenerator
from model import build_model
from datasets import build_loader



def main():
    path=r"./config/hpfg_unet_plus_30k_224x224_ACDC.yaml" # 加载配置文件 load profile
    root = os.path.dirname(os.path.realpath(__file__))  # 获取绝对路径 Obtain absolute path
    args = loadyaml(os.path.join(root, path))  # 加载yaml Load yaml

    # 设置参数
    if args.cuda:
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        args.device = torch.device("cpu")

    args.save_path = os.path.join(root, args.save_path)
    mk_path(args.save_path)  # 创建文件保存位置
    # 创建 tensorboardX日志保存位置
    mk_path(os.path.join(args.save_path, "tensorboardX"))
    mk_path(os.path.join(args.save_path, "model"))  # 创建模型保存位置
    args.model1_save_path = os.path.join(args.save_path, "model", "model1.pth")
    args.model2_save_path = os.path.join(args.save_path, "model", "model2.pth")
    args.ema_model_save_path = os.path.join(args.save_path, "model", "ema_model.pth")  # 设置模型名称

    args.writer = SummaryWriter(os.path.join(args.save_path, "tensorboardX"))
    args.logger = _get_logger(os.path.join(args.save_path, "log.log"), "info")
    args.tqdm = os.path.join(args.save_path, "tqdm.log")
    torch.manual_seed(args.seed)  # 设置随机种子
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.deterministic = False  # 单卡的不需要分布式
    torch.backends.cudnn.benchmark = True  # 寻找最佳 的训练路径，但是会降低一些训练速度
    
    # 构建数据集
    label_loader, unlabel_loader, test_loader = build_loader(args)  # 构建数据集
    args.epochs = args.total_itrs // args.step_size  # 设置模型epoch
    args.logger.info("==========> train_loader length:{}".format(len(label_loader.dataset)))
    args.logger.info("==========> unlabel_loader length:{}".format(len(unlabel_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader.dataset)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))

    # 构建模型，网络1和网络2都是UNet，但是作者这里提到的是unet plus，可能是UNet的一种变种
    model1 = build_model(args=args.model1).to(device=args.device)  
    model2 = build_model(args=args.model2).to(device=args.device) 

    # 创建ema_model，这里需要使用deepcopy，不共享内存，这样对后续的参数更新不会影响
    # 设置ema_model的参数不需要通过梯度下降更新，因为他的参数是通过滑动平均更新的
    ema_model = deepcopy(model2)  
    for name, param in ema_model.named_parameters():
        param.requires_grad = False

    # 移动到设备
    ema_model.to(device=args.device)

    # 训练模型，这里model1和model2都是UNet，ema_model是权重参数无法通过梯度更新的model2，只能通过EMA的方式更新梯度
    HPFG(model1, model2, ema_model, label_loader, unlabel_loader, test_loader, args)


def update_ema_variables_backbone(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)

    for ema_param, param in zip(ema_model.encoder.parameters(), model.encoder.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    for ema_param, param in zip(ema_model.decoder.parameters(), model.decoder.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def HPFG(model1, model2, ema_model, label_loader, unlabel_loader, test_loader, args):
    """
    半监督医学图像分割的主训练函数
    
    参数:
        model1: 第一个分割模型
        model2: 第二个分割模型
        ema_model: 指数移动平均模型(用于教师模型)
        label_loader: 有标签数据加载器
        unlabel_loader: 无标签数据加载器
        test_loader: 测试数据加载器
        args: 配置参数
    """
    # 初始化优化器和学习率调度器
    optimizer1 = build_optimizer(args=args.model1, model=model1)
    lr_scheduler1 = build_lr_scheduler(args=args.model1, optimizer=optimizer1)
    optimizer2 = build_optimizer(args=args.model2, model=model2)
    lr_scheduler2 = build_lr_scheduler(args=args.model2, optimizer=optimizer2)

    # 计算epoch数
    max_epoch = args.total_itrs // len(unlabel_loader) + 1
    args.logger.info("==============> max_epoch :{}".format(max_epoch))

    # 初始化密集对比损失
    dense_loss = Dense_Loss(args.batch_size + args.unlabel_batch_size, args.device)

    # 设置其他的损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(args.num_classes)

    # 设置为训练模式
    model1.train()
    model2.train()

    # 初始化变量
    cur_itrs = 0  # 当前迭代次数
    best_dice1 = 0.0  # 模型1的最佳Dice分数
    best_dice2 = 0.0  # 模型2的最佳Dice分数
    best_ema_dice = 0.0  # EMA模型的最佳Dice分数

    # CutMix配置
    class config:
        cutmix_mask_prop_range = (0.25, 0.5)  # CutMix掩码比例范围
        cutmix_boxmask_n_boxes = 4  # 生成掩码的框数量
        cutmix_boxmask_fixed_aspect_ratio = False  # 是否固定宽高比
        cutmix_boxmask_by_size = False  # 是否按大小比例生成
        cutmix_boxmask_outside_bounds = False  # 是否允许超出边界
        cutmix_boxmask_no_invert = False  # 是否反转掩码

    # 初始化掩码生成器
    mask_generator = BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range,
                                      n_boxes=config.cutmix_boxmask_n_boxes,
                                      random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                      prop_by_area=not config.cutmix_boxmask_by_size,
                                      within_bounds=not config.cutmix_boxmask_outside_bounds,
                                      invert=not config.cutmix_boxmask_no_invert)

    pbar = tqdm(total=args.total_itrs)

    # 初始化有标签数据的迭代器
    label_iter1 = iter(label_loader)
    label_iter = iter(label_loader)

    # 训练循环
    for epoch in range(max_epoch):
        run_loss = 0.0

        # 遍历无标签数据
        for idx, (img_unlabel, _) in enumerate(unlabel_loader):
            cur_itrs += 1

            try:
                # 尝试获取有标签数据
                label_img, target_label = next(label_iter)
                label_img1, target_label1 = next(label_iter1)
            except StopIteration:
                # 如果迭代器耗尽，重新初始化
                label_iter = iter(label_loader)
                label_img, target_label = next(label_iter)
                label_iter1 = iter(label_loader)
                label_img1, target_label1 = next(label_iter1)

            # 获取批量大小
            label_bs = label_img.shape[0]  # 有标签批量大小
            unlabel_bs = img_unlabel.shape[0]  # 无标签批量大小
            
            # 将数据移动到指定设备
            label_img = label_img.to(args.device).float()
            img_unlabel = img_unlabel.to(args.device).float()
            
            # 重复有标签数据以匹配无标签批量大小
            label_img1 = label_img1.repeat(int(unlabel_bs // label_bs), 1, 1, 1).to(args.device).float()
            target_label1 = target_label1.repeat(int(unlabel_bs // label_bs), 1, 1).to(args.device).long()
            target_label = target_label.to(args.device).long()

            # 生成CutMix掩码
            cutmix_mask = mask_generator.generate_params(n_masks=unlabel_bs,
                                                         mask_shape=(args.train_crop_size[0], args.train_crop_size[1]))
            cutmix_mask = torch.tensor(cutmix_mask, dtype=torch.float).to(args.device)

            # 通过CutMix混合有标签数据和无标签数据
            batch_un_mix = label_img1 * (1.0 - cutmix_mask) + img_unlabel * cutmix_mask
            batch_mix = torch.cat([label_img, batch_un_mix], dim=0).to(args.device).float()

            # 模型1向前传播，这里的模型1应该是辅助网络
            outputs1, _, _ = model1(batch_mix)
            outputs_soft1 = torch.softmax(outputs1, dim=1) # 获取Softmax概率

            # 模型2向前传播
            volume_batch = torch.cat([label_img, img_unlabel], dim=0).to(args.device).float()
            outputs2, h1, h2 = model2(volume_batch) # 可以看一下model代码，检查一下为什么能输出这么多
            outputs_soft2 = torch.softmax(outputs2, dim=1) # 获取Softmax概率

            # EMA模型前向传播(不计算梯度)
            with torch.no_grad():
                ema_output, ema_h1, ema_h2 = ema_model(volume_batch)
                ema_output_soft = torch.softmax(ema_output.detach(), dim=1)

            # ========== 计算损失 ==========
            
            # 监督损失(两个模型)
            loss1 = 0.5 * (criterion(outputs1[:label_bs], target_label) + 
                          dice_loss(outputs_soft1[:label_bs], target_label.unsqueeze(1)))
            loss2 = 0.5 * (criterion(outputs2[:label_bs], target_label) + 
                          dice_loss(outputs_soft2[:label_bs], target_label.unsqueeze(1)))
            loss_sup = loss1 + loss2
            # 密集对比损失
            loss_constrivate = dense_loss(h1, ema_h1) + dense_loss(h2, ema_h2)

            # 4个半监督损失=两个cps loss+两个mean_teacher loss
            # 准备CutMix掩码
            cutmix_mask = cutmix_mask.squeeze(1)

            # 后面的一个损失函数
            pseudo_outputs1 = torch.argmax(ema_output_soft[label_bs:], dim=1, keepdim=False)
            pseudo_outputs1 = target_label1 * (1.0 - cutmix_mask) + pseudo_outputs1 * cutmix_mask
            pseudo_supervision1 = dice_loss(outputs_soft1[label_bs:], pseudo_outputs1.unsqueeze(1))

            # mean teacher一致性损失权重
            consistency_weight_cps = args.consistency * linear_rampup(cur_itrs // 150, args.consistency_rampup)
            consistency_weight_mt = args.consistency * linear_rampup(cur_itrs // 150, args.consistency_rampup)

            # 在前一千个item不应用一致性损失
            if cur_itrs < 1000:
                consistency_loss1 = 0.0
                consistency_loss2 = 0.0
            else:
                # consistency_loss1 = torch.mean((outputs_soft1[label_bs:] - ema_output_soft[label_bs:]) ** 2)
                consistency_loss2 = torch.mean((outputs_soft2[label_bs:] - ema_output_soft[label_bs:]) ** 2)

            # 计算模型1和模型2的半监督损失
            model1_loss = 7 * consistency_weight_cps * pseudo_supervision1 + consistency_weight_mt * consistency_loss1
            model2_loss = consistency_weight_mt * consistency_loss2 + consistency_weight_mt * loss_constrivate
            loss_semi = model1_loss + model2_loss

            # 总损失=监督损失+半监督损失
            loss = loss_sup + loss_semi
            run_loss += loss.item()

            # ========== 反向传播和优化 ==========
            
            # 清空梯度
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer1.step()
            optimizer2.step()
            
            # 更新EMA模型参数
            update_ema_variables_backbone(model1, model2, args.ema_decay, cur_itrs)
            update_ema_variables(model2, ema_model, args.ema_decay, cur_itrs)
            
            # 更新学习率
            lr_scheduler1.step()
            lr_scheduler2.step()
            
            # 记录学习率
            lr1 = optimizer1.param_groups[0]["lr"]
            lr2 = optimizer2.param_groups[0]["lr"]

            args.writer.add_scalar('HPFG/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('HPFG/loss_semi', loss_semi.item(), cur_itrs)
            args.writer.add_scalar('HPFG/loss_sup', loss_sup.item(), cur_itrs)
            args.writer.add_scalar('HPFG/lr1', lr1, cur_itrs)
            args.writer.add_scalar('HPFG/lr2', lr2, cur_itrs)
            args.writer.add_scalar('HPFG/consistency_weight_cps', consistency_weight_cps, cur_itrs)
            args.writer.add_scalar('HPFG/consistency_weight_mt', consistency_weight_mt, cur_itrs)

            # ========== 定期评估模型 ==========

            if cur_itrs % args.step_size == 0:
                # 测试模型1
                mean_dice, mean_hd952 = test_acdc(model=model1, test_loader=test_loader, args=args, cur_itrs=cur_itrs,
                                                  name="model1")
                args.writer.add_scalar('HPFG/model1_dice', mean_dice, cur_itrs)
                args.writer.add_scalar('HPFG/model1_hd95', mean_hd952, cur_itrs)
                args.logger.info("model1_dice: {:.4f} model1_hd95: {:.4f}".format(mean_dice, mean_hd952))

                # 保存最佳模型
                if mean_dice > best_dice1:
                    best_dice1 = mean_dice
                    torch.save({
                        "model": model1.state_dict(),
                        "optimizer": optimizer1.state_dict(),
                        "lr_scheduler": lr_scheduler1.state_dict(),
                        "cur_itrs": cur_itrs,
                        "best_dice": best_dice1
                    }, args.model1_save_path)
                model1.train()

                # 测试模型2
                mean_dice, mean_hd952 = test_acdc(model=model2, test_loader=test_loader, args=args, cur_itrs=cur_itrs,
                                                  name="model2")
                args.writer.add_scalar('S4CVnet/model2_dice', mean_dice, cur_itrs)
                args.writer.add_scalar('S4CVnet/model2_hd95', mean_hd952, cur_itrs)
                args.logger.info("model2_dice: {:.4f} model2_hd95: {:.4f}".format(mean_dice, mean_hd952))

                # 保存最佳模型
                if mean_dice > best_dice2:
                    best_dice2 = mean_dice
                    torch.save({
                        "model": model2.state_dict(),
                        "optimizer": optimizer2.state_dict(),
                        "lr_scheduler": lr_scheduler2.state_dict(),
                        "cur_itrs": cur_itrs,
                        "best_dice": best_dice2
                    }, args.model2_save_path)

                # 恢复训练模式
                model2.train()

                # 测试EMA模型
                mean_dice, mean_hd952 = test_acdc(model=ema_model, test_loader=test_loader, args=args,
                                                  cur_itrs=cur_itrs, name="model1")
                args.writer.add_scalar('S4CVnet/ema_dice', mean_dice, cur_itrs)
                args.writer.add_scalar('S4CVnet/ema_hd95', mean_hd952, cur_itrs)
                args.logger.info("ema_dice: {:.4f} ema_hd95: {:.4f}".format(mean_dice, mean_hd952))

                # 保存最佳的EMA模型
                if mean_dice > best_ema_dice:
                    best_ema_dice = mean_dice
                    torch.save({
                        "model": ema_model.state_dict(),
                        "optimizer": optimizer2.state_dict(),
                        "lr_scheduler": lr_scheduler2.state_dict(),
                        "cur_itrs": cur_itrs,
                        "best_dice": best_ema_dice
                    }, args.ema_model_save_path)

                # 恢复训练模式
                ema_model.train()

                # 记录最佳分数
                args.logger.info("model1 best_dice: {:.4f} model2 best_dice: {:.4f} ema best_dice: {:.4f}".format(
                    best_dice1, best_dice2, best_ema_dice))

            # 检查是否达到最大迭代次数
            if cur_itrs > args.total_itrs:
                return
            pbar.update(1)

        args.logger.info(
            "Train  [{}/{} ({:.0f}%)]\t loss: {:.5f}".format(cur_itrs, args.total_itrs,
                                                             100. * cur_itrs / args.total_itrs,
                                                             run_loss / len(unlabel_loader)))


def test_acdc(model, test_loader, args, cur_itrs, name="test"):
    """
    测试模型
    :param model: 模型
    :param test_loader:
    :param args:
    :param cur_itrs:
    :return:
    """
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(test_loader):
        image = sampled_batch[0].to(args.device)
        label = sampled_batch[1].to(args.device)
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=args.test_crop_size)
        metric_list += np.array(metric_i)

        if i_batch == 0:
            slice = image[0, 0, :, :].cpu().detach().numpy()
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (args.test_crop_size[0] / x, args.test_crop_size[1] / y), order=0)
            img = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(args.device)

            label_pred = torch.argmax(torch.softmax(model.val(img), dim=1), dim=1, keepdim=False).squeeze(0)
            label_pred = label_pred.cpu().detach().numpy()
            label_pred = zoom(label_pred, (x / args.test_crop_size[0], y / args.test_crop_size[1]), order=0)
            label_pred = test_loader.dataset.label_to_img(label_pred)

            label_true = label[0, 0, :, :].squeeze().cpu().detach().numpy()
            label_true = test_loader.dataset.label_to_img(label_true)

            args.writer.add_image('{}/Image'.format(name), img[0], cur_itrs)
            args.writer.add_image('{}/label_pred'.format(name), label_pred, cur_itrs, dataformats='HWC')
            args.writer.add_image('{}/label_true'.format(name), label_true, cur_itrs, dataformats='HWC')

    metric_list = metric_list / len(test_loader.dataset)
    performance2 = np.mean(metric_list, axis=0)[0]
    mean_hd952 = np.mean(metric_list, axis=0)[1]
    return performance2, mean_hd952


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net.val(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


if __name__ == "__main__":
    main()
