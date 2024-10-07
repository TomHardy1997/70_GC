import torch
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sksurv.metrics import concordance_index_censored
import os
import pandas as pd
from dataset_new import SwinPrognosisDataset, custom_collate_fn  
from transformer_test import Transformer  
from loss_func import NLLSurvLoss, coxph_loss
import logging
import wandb
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_model_and_optimizer(model_params, criterion_params, learning_rate, weight_decay, warmup_steps=1000):
    model = Transformer(**model_params)
    criterion = NLLSurvLoss(**criterion_params)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 定义一个线性热身调度器
    def warmup_lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

    # 使用 ReduceLROnPlateau 进行进一步的学习率调整
    lr_scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)

    return model, criterion, optimizer, lr_scheduler_warmup, lr_scheduler_plateau

def l1_reg_all(model):
    l1_loss = 0
    # 遍历模型的所有可训练参数
    for param in model.parameters():
        if param.requires_grad:  # 只对可训练参数进行正则化
            l1_loss += torch.sum(torch.abs(param))  # 计算L1范数
    return l1_loss




def train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, fold, lr_scheduler=None, use_l1_loss=False, lambda_reg=1e-4):
    model.train()
    total_loss = 0
    risk_scores = []
    event_times = []
    events = []

    logging.info(f"Epoch {epoch}: Starting training...")
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        optimizer.zero_grad()
        path_features, label, sur_time, censor, _, mask = batch
        path_features, sur_time, censor, mask = path_features.to(device), sur_time.to(device), censor.to(device), mask.to(device)

        outputs = model(path_features, mask)
        loss = criterion(h=outputs, y=label, t=sur_time, c=censor)

        # 如果启用了 l1_loss，则加上 L1 正则化项
        if use_l1_loss:
            l1_loss = l1_reg_all(model)
            loss += lambda_reg * l1_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if lr_scheduler is not None:
            lr_scheduler.step()

        with torch.no_grad():
            risk = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1).cpu().numpy()
            risk_scores.extend(risk)
            event_times.extend(sur_time.cpu().numpy())
            events.extend(censor.cpu().numpy().astype(bool))

    avg_loss = total_loss / len(train_loader)
    # import ipdb;ipdb.set_trace()
    train_c_index = concordance_index_censored((1 - np.array(events)).astype(bool), event_times, np.array(risk_scores))[0]

    logging.info(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train C-Index: {train_c_index:.4f}")
    wandb.log({'Train Loss': avg_loss, 'Train C-Index': train_c_index, 'Epoch': epoch, 'Fold': fold})
    
    return avg_loss, train_c_index



def validate_one_epoch(model, criterion, val_loader, device, epoch, fold):
    model.eval()
    total_loss = 0
    risk_scores = []
    event_times = []
    events = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch}"):
            # import ipdb;ipdb.set_trace()
            path_features, label, sur_time, censor, _ ,mask= batch
            path_features, sur_time, censor, mask = path_features.to(device), sur_time.to(device), censor.to(device), mask.to(device)
            # import ipdb;ipdb.set_trace()
            outputs = model(path_features,mask)
            
            loss = criterion(h=outputs, y=label, t=sur_time, c=censor)
            # loss = criterion(risk=outputs, phase=sur_time, censors=censor)
            total_loss += loss.item()

            risk = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1).cpu().numpy()
            risk_scores.extend(risk)
            # risk_scores.extend(outputs.cpu().numpy())
            event_times.extend(sur_time.cpu().numpy())
            events.extend(censor.cpu().numpy().astype(bool))
    
    avg_loss = total_loss / len(val_loader)
    val_c_index = concordance_index_censored((1-np.array(events)).astype(bool), event_times, np.array(risk_scores))[0]
    # val_c_index = concordance_index_censored(np.array(events), event_times, np.array(risk_scores).reshape(-1))[0]
    wandb.log({'Val Loss': avg_loss, 'Val C-Index': val_c_index, 'Epoch': epoch, 'Fold': fold})
    return avg_loss, val_c_index

# 测试集评估
def test_model(model, test_loader, device):
    model.eval()
    risk_scores = []
    event_times = []
    events = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing"):
            path_features, label, sur_time, censor, _ ,mask= batch
            path_features, sur_time, censor,mask = path_features.to(device), sur_time.to(device), censor.to(device), mask.to(device)
            outputs = model(path_features,mask)
            
            risk = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1).cpu().numpy()
            risk_scores.extend(risk)
            event_times.extend(sur_time.cpu().numpy())
            events.extend(censor.cpu().numpy().astype(bool))

    test_c_index = concordance_index_censored((1-np.array(events)).astype(bool), event_times, np.array(risk_scores))[0]

    return test_c_index

def external_test_model(model, test_loader, device, external_csv_path, fold, experiment_id):
    model.eval()
    risk_scores = []
    labels = []
    event_times = []
    events = []
    patient_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"External Testing Fold {fold}"):
            path_features, label, sur_time, censor, patient_id, mask = batch
            path_features, label, sur_time, censor, mask = path_features.to(device), label.to(device), sur_time.to(device), censor.to(device), mask.to(device)
            # path_features, sur_time, censor = path_features.to(device), sur_time.to(device), censor.to(device)
            outputs = model(path_features,mask)

            risk = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1).cpu().numpy()
            risk_scores.extend(risk)
            # risk_scores.extend(outputs.cpu().numpy())
            labels.extend(label.cpu().numpy())
            event_times.extend(sur_time.cpu().numpy())
            events.extend(censor.cpu().numpy().astype(bool))
            patient_list.extend(patient_id)
    
    # external_c_index = concordance_index_censored((1 - np.array(events)).astype(bool), event_times, np.array(risk_scores))[0]
    external_c_index = concordance_index_censored((1-np.array(events)).astype(bool), event_times, np.array(risk_scores))[0]
    # external_c_index = concordance_index_censored(np.array(events), event_times, np.array(risk_scores).reshape(-1))[0]
    result_df = pd.DataFrame({
        'Patient_ID': patient_list,
        'Label': labels,
        'Survival_Time': event_times,
        'Censor': events,
        'Risk_Score': risk_scores
    })
    # result_csv_path = os.path.join(external_csv_path, f'external_test_results_fold_{fold}.csv')
    # result_csv_path = os.path.join(external_csv_path, f'external_test_results_{experiment_id}.csv')
    result_csv_path = os.path.join(external_csv_path, experiment_id, f'external_test_results_fold_{fold}.csv')
    os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
    result_df.to_csv(result_csv_path, index=False)
    # result_df.to_csv(result_csv_path, index=False)
    print(f"Saved external test results to {result_csv_path}")
    
    return external_c_index

class CustomSubset(Subset):
    def __init__(self, dataset, indices, is_training=True):
        super().__init__(dataset, indices)
        self.is_training = is_training

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        return (*item, self.is_training)


def save_params_to_txt(args, model_params, criterion_params, save_dir):
    param_file_path = os.path.join(save_dir, 'parameters.txt')
    with open(param_file_path, 'w') as f:
        f.write("===== Training Arguments =====\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n===== Model Parameters =====\n")
        for key, value in model_params.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n===== Criterion Parameters =====\n")
        for key, value in criterion_params.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved parameters to {param_file_path}")