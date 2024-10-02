import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import argparse
from dataset_new import SwinPrognosisDataset, custom_collate_fn
from transformer_test import Transformer
from loss_func import NLLSurvLoss
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm
import os
import pandas as pd
import optuna
import logging
import random
import wandb
from utils import *

# 设置模型、损失函数和优化器

# 主训练循环，加入外部测试步骤
# def train_model(train_indices, val_indices, test_indices, external_indices, args, model_params, criterion_params, fold):
def train_model(train_indices, val_indices, test_indices, external_indices, args, model_params, criterion_params, fold, trial_number, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_dataset = SwinPrognosisDataset(args.train_csv, data_dir=args.data_dir)
    external_dataset = SwinPrognosisDataset(args.external_csv, data_dir=args.external_data_dir)
    train_subset = CustomSubset(full_dataset, train_indices, is_training=True)
    val_subset = CustomSubset(full_dataset, val_indices, is_training=False)
    test_subset = CustomSubset(full_dataset, test_indices, is_training=False)
    external_subset = CustomSubset(external_dataset, external_indices, is_training=False)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=args.val_batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_subset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    external_loader = DataLoader(external_subset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    model, criterion, optimizer = setup_model_and_optimizer(model_params, criterion_params, learning_rate, args.weight_decay)
    model = nn.DataParallel(model)
    model.to(device)

    # 先定义 experiment_id 和 experiment_dir
    experiment_id = f"trial_{trial_number}_fold_{fold}_lr_{learning_rate}_dropout_{model_params['dropout']}_depth_{model_params['depth']}_heads_{model_params['heads']}_dim_{model_params['dim']}"
    experiment_dir = os.path.join(args.save_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # 在定义 fold_dir 后，再配置日志
    fold_dir = os.path.join(experiment_dir, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    # 现在可以配置日志记录
    logging.basicConfig(
        filename=os.path.join(experiment_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    save_params_to_txt(args, model_params, criterion_params, fold_dir)

    # 后续代码保持不变
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)
    best_val_c_index = -1
    patience_counter = 0
    early_stopping_patience = 20
    results = []

    for epoch in range(args.max_epochs):
        train_loss, train_c_index = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, fold)
        val_loss, val_c_index = validate_one_epoch(model, criterion, val_loader, device, epoch, fold)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train C-Index: {train_c_index:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val C-Index: {val_c_index:.4f}")

        scheduler.step(val_c_index)
        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        results.append({
            'Epoch': epoch,
            'Train_Loss': train_loss,
            'Train_C_Index': train_c_index,
            'Val_Loss': val_loss,
            'Val_C_Index': val_c_index
        })

    # 保存训练指标
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(fold_dir, 'training_metrics.csv'), index=False)

    # 测试和外部测试
    print("Evaluating on test set...")
    test_c_index = test_model(model, test_loader, device)
    print(f"Test C-Index: {test_c_index:.4f}")

    with open(os.path.join(fold_dir, 'test_c_index.txt'), 'w') as f:
        f.write(f"Test C-Index: {test_c_index:.4f}")

    print("Evaluating on external test set...")
    external_c_index = external_test_model(model, external_loader, device, args.save_dir, fold, experiment_id)
    print(f"External Test C-Index: {external_c_index:.4f}")

    with open(os.path.join(fold_dir, 'external_test_c_index.txt'), 'w') as f:
        f.write(f"External Test C-Index: {external_c_index:.4f}")

    return test_c_index, external_c_index

# 使用 Optuna 调参 + 五折交叉验证
# objective 函数
def objective(trial, args, train_indices, val_indices, test_indices, external_indices, model_params, criterion_params):
    import copy
    # 使用 Optuna 调参
    learning_rate = trial.suggest_categorical('learning_rate', [1e-5])
    # dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    dropout = trial.suggest_categorical('dropout', [0.3])
    # depth = trial.suggest_categorical('depth', [1, 2, 4])
    depth = trial.suggest_categorical('depth', [1])
    # heads = trial.suggest_categorical('heads', [1, 2, 4])
    heads = trial.suggest_categorical('heads', [1])
    dim = trial.suggest_categorical('dim', [256])
    mlp_dim = trial.suggest_categorical('mlp_dim', [64])
    dim_head = trial.suggest_categorical('dim_head', [32])
    use_DropKey = trial.suggest_categorical('use_DropKey', [False])
    mask_ratio = trial.suggest_categorical('mask_ratio', [0.1, 0.2, 0.3, 0.4, 0.5])
    alpha = trial.suggest_categorical('alpha', [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    # 使用深拷贝防止参数在不同试验之间相互影响
    model_params = copy.deepcopy(model_params)
    criterion_params = copy.deepcopy(criterion_params)

    # 更新模型和损失函数参数
    model_params.update({
        'dim': dim,
        'depth': depth,
        'heads': heads,
        'mlp_dim': mlp_dim,
        'dim_head': dim_head,
        'dropout': dropout,
        'use_DropKey': use_DropKey,
        'mask_ratio': mask_ratio
    })
    criterion_params['alpha'] = alpha

    # 初始化 WandB
    wandb.init(
        project='Your Project Name',  # 请替换为您的项目名称
        name=f"trial_{trial.number}",
        config={
            'learning_rate': learning_rate,
            'dropout': dropout,
            'depth': depth,
            'heads': heads,
            'dim': dim,
            'mlp_dim': mlp_dim,
            'dim_head': dim_head,
            'use_DropKey': use_DropKey,
            'mask_ratio': mask_ratio,
            'alpha': alpha,
            'batch_size': args.batch_size,
            'val_batch_size': args.val_batch_size,
            'test_batch_size': args.test_batch_size,
            'weight_decay': args.weight_decay,
            'max_epochs': args.max_epochs,
            'seed': args.seed,
            'trial_number': trial.number,
        }
    )

    # 如果使用 7:1.5:1.5 划分，不需要交叉验证
    if args.mode == 'train_val_test':
        fold = 1  # 单次划分，不需要多个 fold
        print(f"Starting 7:1.5:1.5 split for Trial {trial.number}...")
        
        # 直接传入从 `train_val_test_split` 中传递的 `train_indices` 和 `val_indices`
        _, val_c_index = train_model(
            train_indices=train_indices,  
            val_indices=val_indices,    
            test_indices=test_indices,
            external_indices=external_indices,
            args=args,
            model_params=model_params,
            criterion_params=criterion_params,
            fold=fold,
            trial_number=trial.number,
            learning_rate=learning_rate
        )
        avg_c_index = val_c_index  # 没有多个 fold，直接返回单次的结果

    # 否则，使用五折交叉验证
    else:
        full_dataset = SwinPrognosisDataset(args.train_csv, data_dir=args.data_dir)
        labels = [full_dataset[i][1] for i in range(len(full_dataset))]

        skf = StratifiedKFold(n_splits=5)
        val_c_index_avg = 0

        for fold, (train_indices, val_indices) in enumerate(skf.split(np.arange(len(full_dataset)), labels)):
            print(f"Starting Fold {fold + 1} of Trial {trial.number}...")
            _, val_c_index = train_model(
                train_indices,
                val_indices,
                test_indices,
                external_indices,
                args,
                model_params,
                criterion_params,
                fold=fold + 1,
                trial_number=trial.number,
                learning_rate=learning_rate
            )
            val_c_index_avg += val_c_index

        avg_c_index = val_c_index_avg / 5  # 五折的平均 C-index

    # 保存试验结果
    trial_result = {
        'trial_number': trial.number,
        'learning_rate': learning_rate,
        'dropout': dropout,
        'depth': depth,
        'heads': heads,
        'dim': dim,
        'mlp_dim': mlp_dim,
        'dim_head': dim_head,
        'use_DropKey': use_DropKey,
        'mask_ratio': mask_ratio,
        'alpha': alpha,
        'avg_val_c_index': avg_c_index
    }

    # 将结果保存到一个 CSV 文件，追加模式
    results_file = os.path.join(args.save_dir, 'all_trials_results.csv')
    results_df = pd.DataFrame([trial_result])
    if not os.path.exists(results_file):
        results_df.to_csv(results_file, index=False)
    else:
        results_df.to_csv(results_file, mode='a', header=False, index=False)

    # 结束 WandB 运行
    wandb.finish()

    return avg_c_index


# 使用 10% 数据作为测试集，剩下 90% 数据用于五折交叉验证
def train_val_test_split(args, model_params, criterion_params):
    full_dataset = SwinPrognosisDataset(args.train_csv, data_dir=args.data_dir)
    labels = [full_dataset[i][1] for i in range(len(full_dataset))]

    # 按 7:1.5:1.5 进行数据划分
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        np.arange(len(full_dataset)), labels, test_size=0.15, stratify=labels, random_state=args.seed
    )

    # 将训练+验证数据集划分为 70% 训练集和 15% 验证集
    train_indices, val_indices, train_labels, val_labels = train_test_split(
        train_val_indices, train_val_labels, test_size=0.1765, stratify=train_val_labels, random_state=args.seed
    )

    external_indices = np.arange(len(SwinPrognosisDataset(args.external_csv, data_dir=args.external_data_dir)))

    # 使用 Optuna 调参，传递 train_indices 和 val_indices
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args, train_indices, val_indices, test_indices, external_indices, model_params, criterion_params), n_trials=30)

    print("最佳超参数: ", study.best_params)
    print("最佳外部测试集 C-Index: ", study.best_value)


def cross_validation_with_test_set(args, model_params, criterion_params):
    full_dataset = SwinPrognosisDataset(args.train_csv, data_dir=args.data_dir)
    labels = [full_dataset[i][1] for i in range(len(full_dataset))]

    # 将 10% 数据分为测试集，剩余 90% 数据用于五折交叉验证
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        np.arange(len(full_dataset)), labels, test_size=0.1, stratify=labels, random_state=42
    )

    # External dataset indices
    external_indices = np.arange(len(SwinPrognosisDataset(args.external_csv, data_dir=args.external_data_dir)))

    # 使用 Optuna 调参
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args, test_indices, external_indices, model_params, criterion_params), n_trials=30)

    print("最佳超参数: ", study.best_params)
    print("最佳外部测试集 C-Index: ", study.best_value)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 更新主函数
def main():
    parser = argparse.ArgumentParser(description="Training Transformer for Survival Analysis")
    parser.add_argument('--data_dir', type=str, default='/mnt/usb5/jijianxin/', help='Directory containing the data files')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--external_csv', type=str, required=True, help='Path to the external test CSV file')
    parser.add_argument('--external_data_dir', type=str, required=True, help='Directory for the external dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation DataLoader')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch size for test DataLoader')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the best model and results')
    parser.add_argument('--alpha', type=float, default=0.3, help='Weighting factor for the uncensored loss term')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, choices=['train_val_test', 'cross_validation'], required=True, help='Choose between 7:1.5:1.5 split or cross-validation')
    args = parser.parse_args()
    
    set_seed(args.seed)

    model_params = {
        'num_classes': 4,
        'input_dim': 1024,
        'dim': 256,
        'depth': 1,
        'heads': 1,
        'mlp_dim': 128,
        'pool': 'cls',
        'dim_head': 32,
        'dropout': 0.3,
        'emb_dropout': 0.3,
        'use_DropKey': False,
        'mask_ratio': 0.1
    }
    criterion_params = {'alpha': args.alpha}

    logging.info("Starting training with the following parameters:")
    logging.info(f"Arguments: {args}")
    logging.info(f"Model Parameters: {model_params}")
    logging.info(f"Criterion Parameters: {criterion_params}")

    # 根据 mode 选择使用的划分方式
    if args.mode == 'train_val_test':
        train_val_test_split(args, model_params, criterion_params)
    elif args.mode == 'cross_validation':
        cross_validation_with_test_set(args, model_params, criterion_params)

if __name__ == "__main__":
    main()
