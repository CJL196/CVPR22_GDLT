import torch
import numpy as np
import options
import argparse
import os
from datasets import RGDataset
from torch.utils.data import DataLoader
from models import model, loss
import train
from test import test_epoch
import optuna
from tensorboardX import SummaryWriter


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optim(model, args):
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception("Unknown optimizer")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=args.epoch - args.warmup, eta_min=args.lr * args.decay_rate)
        elif args.lr_decay == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epoch - 30], gamma=args.decay_rate)
        else:
            raise Exception("Unknown Scheduler")
    else:
        scheduler = None
    return scheduler


def objective(trial, args):
    # 设置超参数搜索空间
    args.lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    args.n_decoder = trial.suggest_int('n_decoder', 1, 4)
    args.n_query = trial.suggest_int('n_query', 2, 8)
    args.decay_rate = trial.suggest_float('decay_rate', 0.001, 0.1, log=True)
    args.dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # 设置随机种子确保可复现
    setup_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 加载数据
    if args.dataset.lower() == 'finefs':
        from FineFSLoader import get_FineFS_loader
        if args.split.lower() == 'sp':
            indices_path = 'indices/indices_sp_0.pkl'
        elif args.split.lower() == 'fs':
            indices_path = 'indices/indices_fs_0.pkl'
        train_data, test_data, train_loader, test_loader = get_FineFS_loader(args.train_label_path, args.video_path, indices_path, args.batch, type=args.type)
    elif args.dataset.lower() == 'gdlt':
        train_data = RGDataset(args.video_path, args.train_label_path, clip_num=args.clip_num,
                           action_type=args.action_type)
        train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=8)

        test_data = RGDataset(args.video_path, args.test_label_path, clip_num=args.clip_num,
                           action_type=args.action_type, train=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
    else:
        raise Exception("Unknown dataset")
    
    # 加载模型
    model_instance = model.GDLT(args.in_dim, args.hidden_dim, args.n_head, args.n_encoder,
                       args.n_decoder, args.n_query, args.dropout).to(device)
    loss_fn = loss.LossFun(args.alpha, args.margin)
    train_fn = train.train_epoch
    
    # 设置优化器和学习率调度器
    optim = get_optim(model_instance, args)
    scheduler = get_scheduler(optim, args)
    
    # 训练和评估
    best_coef = -1
    
    for epc in range(args.search_epochs):
        if args.warmup and epc < args.warmup:
            warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
            warmup.step()
            
        avg_loss, train_coef = train_fn(epc, model_instance, loss_fn, train_loader, optim, None, device, args)
        
        if scheduler is not None and (args.lr_decay != 'cos' or epc >= args.warmup):
            scheduler.step()
            
        test_loss, test_coef = test_epoch(epc, model_instance, test_loader, None, device, args)
        
        if test_coef > best_coef:
            best_coef = test_coef
        
        # 向Optuna报告当前结果
        trial.report(test_coef, epc)
        
        # # 如果表现不佳，提前终止
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
    
    return best_coef


def parse_args():
    # 先获取原始的参数解析器
    parser = options.parser
    
    # 添加我们自己的参数
    parser.add_argument('--num_trials', type=int, default=10000, help='Number of trials for hyperparameter search')
    parser.add_argument('--search_epochs', type=int, default=100, help='Number of epochs per trial')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workspace_name', type=str, default='optuna_search', help='Workspace name')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建工作目录
    args.workspace_dir = os.path.join('./workspace', args.workspace_name)
    if not os.path.exists(args.workspace_dir):
        os.makedirs(args.workspace_dir)
    
    return args


def main():
    args = parse_args()
    setup_seed(args.seed)
    
    # 创建Optuna研究
    storage_name = f'sqlite:///{args.workspace_dir}/optuna.db'
    study = optuna.create_study(
        direction='maximize',  # 最大化相关系数
        pruner=optuna.pruners.MedianPruner(),
        study_name='gdlt_hyperparameter_search',
        storage=storage_name,
        load_if_exists=True
    )
    
    # 开始优化
    study.optimize(lambda trial: objective(trial, args), n_trials=args.num_trials)
    
    # 打印最佳结果
    print('最佳超参数组合:')
    print(study.best_params)
    print(f'最佳性能: {study.best_value}')
    
    # 保存最佳超参数到文件
    with open(os.path.join(args.workspace_dir, 'best_params.txt'), 'w') as f:
        f.write(f'最佳性能: {study.best_value}\n')
        for key, value in study.best_params.items():
            f.write(f'{key}: {value}\n')


if __name__ == '__main__':
    main()
