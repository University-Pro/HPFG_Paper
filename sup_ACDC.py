# 导入必要的库和模块
import os.path  # 用于处理文件路径
import numpy as np  # 科学计算库
import torch  # PyTorch深度学习框架
from tensorboardX import SummaryWriter  # 用于可视化训练过程
from tqdm import tqdm  # 进度条显示
from utils import loadyaml, _get_logger, mk_path, Med_Sup_Loss  # 自定义工具函数
from model import build_model  # 模型构建模块
from datasets import build_loader  # 数据加载器构建模块
from utils import build_lr_scheduler, build_optimizer  # 学习率调度器和优化器构建模块
from val import test_acdc  # 验证函数

def main():
    # 配置文件路径
    path = r"config/unet_30k_224x224_ACDC.yaml"
    # 获取当前脚本的绝对路径
    root = os.path.dirname(os.path.realpath(__file__))
    # 加载YAML配置文件
    args = loadyaml(os.path.join(root, path))
    
    # 设置计算设备（GPU或CPU）
    if args.cuda:
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        args.device = torch.device("cpu")
    
    # 设置随机种子以保证实验可重复性
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 禁用确定性算法以加速训练（单GPU不需要分布式）
    torch.backends.cudnn.deterministic = False
    # 启用自动寻找最优算法
    torch.backends.cudnn.benchmark = True
    
    # 再次获取当前路径（冗余代码，可优化）
    root = os.path.dirname(os.path.realpath(__file__))
    # 设置模型保存路径
    args.save_path = os.path.join(root, args.save_path)
    # 创建保存目录
    mk_path(args.save_path)
    
    # 创建tensorboard日志目录
    mk_path(os.path.join(args.save_path, "tensorboardX"))
    # 创建模型保存目录
    mk_path(os.path.join(args.save_path, "model"))
    # 设置不同训练阶段的模型保存路径
    args.finetune_save_path = os.path.join(args.save_path, "model", "finetune_model.pth")
    args.pretrain_save_path = os.path.join(args.save_path, "model", "pretrain_model.pth")
    args.supervise_save_path = os.path.join(args.save_path, "model", "supervise_model.pth")
    
    # 初始化tensorboard写入器
    args.writer = SummaryWriter(os.path.join(args.save_path, "tensorboardX"))
    # 初始化日志记录器
    args.logger = _get_logger(os.path.join(args.save_path, "log.log"), "info")
    # 设置tqdm日志路径
    args.tqdm = os.path.join(args.save_path, "tqdm.log")
    
    # 步骤1：构建数据集
    train_loader, test_loader = build_loader(args)
    # 计算总epoch数
    args.epochs = args.total_itrs // len(train_loader) + 1
    # 记录数据集信息
    args.logger.info("==========> train_loader length:{}".format(len(train_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader.dataset)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))
    
    # 步骤2：构建模型
    model = build_model(args=args).to(device=args.device)
    
    # 步骤3：训练模型
    Supervise(model=model, train_loader=train_loader, test_loader=test_loader, args=args)

def Supervise(model, train_loader, test_loader, args):
    # 构建优化器
    optimizer = build_optimizer(args=args, model=model)
    # 构建学习率调度器
    lr_scheduler = build_lr_scheduler(args=args, optimizer=optimizer)
    # 计算最大epoch数
    max_epoch = args.total_itrs // len(train_loader) + 1
    args.logger.info("==============> max_epoch :{}".format(max_epoch))
    
    # 定义损失函数
    criterion = Med_Sup_Loss(args.num_classes)
    
    # 设置模型为训练模式
    model.train()
    # 初始化变量
    cur_itrs = 0  # 当前迭代次数
    train_loss = 0.0  # 训练损失
    best_dice = 0.0  # 最佳dice分数
    
    # 加载预训练模型（如果有）
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        state_dict = torch.load(args.ckpt)
        cur_itrs = state_dict["cur_itrs"]
        model = state_dict["model"]
        optimizer = state_dict["optimizer"]
        lr_scheduler = state_dict["lr_scheduler"]
        best_dice = state_dict["best_score"]
    
    # 开始训练循环
    for epoch in range(max_epoch):
        # 使用tqdm显示进度条
        for i, (img, label_true) in enumerate(tqdm(train_loader)):
            cur_itrs += 1  # 更新迭代计数器
            
            # 将数据移动到指定设备
            img = img.to(args.device).float()
            label_true = label_true.to(args.device).long()
            
            # 前向传播
            label_pred = model(img)
            # 计算损失
            loss = criterion(label_pred, label_true)
            
            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            lr_scheduler.step()  # 更新学习率
            
            # 获取当前学习率
            lr = optimizer.param_groups[0]["lr"]
            # 累计训练损失
            train_loss += loss.item()
            
            # 记录损失和学习率到tensorboard
            args.writer.add_scalar('supervise/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('supervise/lr', lr, cur_itrs)
            
            # 定期验证模型
            if cur_itrs % args.step_size == 0:
                # 在验证集上测试
                dice, hd95 = test_acdc(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs)
                # 记录指标
                args.writer.add_scalar('supervise/{}_dice'.format(args.name), dice, cur_itrs)
                args.writer.add_scalar('supervise/{}_hd95'.format(args.name), hd95, cur_itrs)
                args.logger.info("epoch:{} \t dice:{:.5f} \t hd95:{:.5f} ".format(epoch, dice, hd95))
                
                # 保存最佳模型
                if dice > best_dice:
                    best_dice = dice
                    torch.save({
                        "cur_itrs": cur_itrs,
                        "best_dice": best_dice,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }, os.path.join(args.save_path, "model", "model_{:.4f}.pth".format(best_dice)))
                
                # 恢复训练模式
                model.train()
            
            # 检查是否达到最大迭代次数
            if cur_itrs > args.total_itrs:
                return
        
        # 记录训练进度
        args.logger.info("Train [{}/{} ({:.0f}%)]\t loss: {:.5f}\t best_dice:{:.5f} ".format(
            cur_itrs, args.total_itrs,
            100. * cur_itrs / args.total_itrs,
            train_loss, best_dice
        ))
        # 重置训练损失
        train_loss = 0

if __name__ == "__main__":
    main()