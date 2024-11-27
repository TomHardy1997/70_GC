import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset, DataLoader
from sksurv.metrics import concordance_index_censored
import os
import pandas as pd
from dataset_position import SwinPrognosisDataset
from model_utils import custom_collate_fn
from transformer_context import Transformer
from loss_func import CombinedSurvLoss
from tqdm import tqdm
import logging
import wandb
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch.optim as optim


def setup_model_and_optimizer(model_params,
                              criterion_params,
                              learning_rate,
                              weight_decay,
                              warmup_steps=1000,
                              total_steps=10000):
    model = Transformer(**model_params)
    criterion = CombinedSurvLoss(**criterion_params)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def warmup_lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    
    lr_scheduler_warmup = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    lr_scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    return model, criterion, optimizer, lr_scheduler_warmup, lr_scheduler_cosine

def train_one_epoch(model, 
                    criterion, 
                    optimizer,
                    train_loader,
                    device,
                    epoch,
                    fold,
                    lr_scheduler_warmup=None,
                    lr_scheduler_cosine=None,
                    use_l1_loss=False,
                    lambda_reg=1e-4,
                    warmup_steps=1000):
    model.train()
    total_loss = 0
    risk_scores = []
    event_times = []
    events = []
    step = epoch * len(train_loader)
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch}'):
        optimizer.zero_grad()

        patient, gender, age, label, sur_time, censor, feature, _, num_patches, mask = batch
        gender = gender.to(device)
        age = age.to(device)
        label = label.to(device)
        sur_time = sur_time.to(device)
        censor = censor.to(device)
        feature = feature.to(device)
        # num_patches = num_patches.to(device)
        mask = mask.to(device)
        # import ipdb;ipdb.set_trace()
        outputs = model(feature, gender, age, mask)
        loss = criterion(outputs=outputs, y=label, t=sur_time, c=censor)

        
        if use_l1_loss:
            l1_regularization = lambda_reg * sum(torch.abs(param).sum() for param in model.parameters())
            loss += l1_regularization
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()

        # 调整学习率：只在 warmup 阶段使用 lr_scheduler_warmup，在之后使用 lr_scheduler_cosine
        step += 1
        if step < warmup_steps and lr_scheduler_warmup is not None:
            lr_scheduler_warmup.step()
            # print(f"Step {step}: Warmup phase. Current LR: {optimizer.param_groups[0]['lr']:.8f}")
        elif lr_scheduler_cosine is not None:
            lr_scheduler_cosine.step()
            # print(f"Step {step}: Post-warmup phase. Current LR: {optimizer.param_groups[0]['lr']:.8f}")

        # GPU 同步
        # torch.cuda.synchronize()

        with torch.no_grad():
            risk = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1).cpu().numpy()
            risk_scores.extend(risk)
            event_times.extend(sur_time.cpu().numpy())
            events.extend(censor.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    train_c_index = concordance_index_censored((1 - np.array(events)).astype(bool), event_times, np.array(risk_scores))[0]
    print(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train C-Index: {train_c_index:.4f}")
    wandb.log({'Train Loss': avg_loss, 'Train C-Index': train_c_index, 'Fold': fold, 'Epoch': epoch})

    return avg_loss, train_c_index

def validate_one_epoch(model, criterion, val_loader, device, epoch, fold):
    model.eval()
    total_loss = 0
    risk_scores = []
    event_times = []
    events = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validating Epoch {epoch}'):
            patient, gender, age, label, sur_time, censor, feature, _, num_patches, mask = batch
            gender = gender.to(device)
            age = age.to(device)
            label = label.to(device)
            sur_time = sur_time.to(device)
            censor = censor.to(device)
            feature = feature.to(device)
            # num_patches = num_patches.to(device)
            mask = mask.to(device)

            outputs = model(feature, gender, age, mask)
            loss = criterion(outputs=outputs, y=label, t=sur_time, c=censor)


            total_loss += loss.item()

            risk = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1).cpu().numpy()
            risk_scores.extend(risk)

            event_times.extend(sur_time.cpu().numpy())
            events.extend(censor.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        val_c_index = concordance_index_censored((1 - np.array(events)).astype(bool), event_times, np.array(risk_scores))[0]
        wandb.log({'Val Loss': avg_loss, 'Val C-Index': val_c_index, 'Epoch': epoch, 'Fold': fold})
        
        return avg_loss, val_c_index
    

def test_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    risk_scores = []
    event_times = []
    events = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            patient, gender, age, label, sur_time, censor, feature, _, num_patches, mask = batch
            gender = gender.to(device)
            age = age.to(device)
            label = label.to(device)
            sur_time = sur_time.to(device)
            censor = censor.to(device)
            feature = feature.to(device)
            # num_patches = num_patches.to(device)
            mask = mask.to(device)

            outputs = model(feature, gender, age, mask)
            risk = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1).cpu().numpy()
            risk_scores.extend(risk)

            event_times.extend(sur_time.cpu().numpy())
            events.extend(censor.cpu().numpy())

    test_c_index = concordance_index_censored((1 - np.array(events)).astype(bool), event_times, np.array(risk_scores))[0]
    return test_c_index

def external_test_model(model, test_loader, device, external_csv_path, fold, experiment_id, external_type="external"):
    model.eval()
    risk_scores = []
    labels = []
    event_times = []
    events = []
    patient_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Testing {external_type}'):
            patient, gender, age, label, sur_time, censor, feature, _, num_patches, mask = batch
            gender = gender.to(device)
            age = age.to(device)
            label = label.to(device)
            sur_time = sur_time.to(device)
            censor = censor.to(device)
            feature = feature.to(device)
            mask = mask.to(device)

            outputs = model(feature, gender, age, mask)
            risk = -torch.sum(torch.cumprod(1 - torch.sigmoid(outputs), dim=1), dim=1).cpu().numpy()
            risk_scores.extend(risk)
            labels.extend(label.cpu().numpy())
            event_times.extend(sur_time.cpu().numpy())
            events.extend(censor.cpu().numpy())
            patient_list.extend(patient)

    # 在这里筛选NaN，确保所有列表长度一致
    risk_scores = np.array(risk_scores)
    labels = np.array(labels)
    event_times = np.array(event_times)
    events = np.array(events)

    valid_indices = ~np.isnan(risk_scores)
    risk_scores = risk_scores[valid_indices]
    labels = labels[valid_indices]
    event_times = event_times[valid_indices]
    events = events[valid_indices]
    patient_list = [patient_list[i] for i in range(len(patient_list)) if valid_indices[i]]

    external_c_index = concordance_index_censored((1 - np.array(events)).astype(bool), event_times, np.array(risk_scores))[0]

    result_df = pd.DataFrame({
        'patient': patient_list, 
        'risk_score': risk_scores, 
        'label': labels, 
        'survival_time': event_times, 
        'censor': events
    })
    
    # 创建不同的保存路径避免替换文件
    result_csv_path = os.path.join(external_csv_path, experiment_id, f'{external_type}_test_results_fold_{fold}.csv')
    os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
    result_df.to_csv(result_csv_path, index=False)
    print(f"Saved {external_type} test results to {result_csv_path}")

    return external_c_index



def train_model(train_indices, val_indices, test_indices, external_indices, args, model_params, criterion_params, fold, trial_number, learning_rate, use_l1_loss):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_mode = 'pt'
    full_dataset = SwinPrognosisDataset(args.train_csv, pt_dir=args.data_dir)
    external_dataset = SwinPrognosisDataset(args.external_csv, pt_dir=args.external_data_dir)

    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)
    external_subset = Subset(external_dataset, external_indices)
    

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=lambda batch: custom_collate_fn(batch, load_mode))
    val_loader = DataLoader(val_subset, batch_size=args.val_batch_size, shuffle=False, num_workers=4, collate_fn=lambda batch: custom_collate_fn(batch, load_mode))
    test_loader = DataLoader(test_subset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, collate_fn=lambda batch: custom_collate_fn(batch, load_mode))
    external_loader = DataLoader(external_subset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, collate_fn=lambda batch: custom_collate_fn(batch, load_mode))
    

    model, criterion, optimizer, lr_scheduler_warmup, lr_scheduler_cosine = setup_model_and_optimizer(
        model_params, criterion_params, learning_rate, args.weight_decay
    )
    print("Model initialized with the following parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    model = nn.DataParallel(model)
    model.to(device)

    # 创建实验目录
    experiment_id = f"trial_{trial_number}_fold_{fold}_lr_{learning_rate}_dropout_{model_params['dropout']}_depth_{model_params['depth']}_heads_{model_params['heads']}_dim_{model_params['dim']}"
    experiment_dir = os.path.join(args.save_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    fold_dir = os.path.join(experiment_dir, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(experiment_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    save_params_to_txt(args, model_params, criterion_params, fold_dir)

    best_val_c_index = -1
    patience_counter = 0
    early_stopping_patience = 10
    results = []
    warmup_steps = 1000

    for epoch in range(args.max_epochs):
        train_loss, train_c_index = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch, fold,
            lr_scheduler_warmup=lr_scheduler_warmup,
            lr_scheduler_cosine=lr_scheduler_cosine,
            use_l1_loss=use_l1_loss, warmup_steps=warmup_steps
        )

        val_loss, val_c_index = validate_one_epoch(model, criterion, val_loader, device, epoch, fold)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train C-Index: {train_c_index:.4f}, Val Loss: {val_loss:.4f}, Val C-Index: {val_c_index:.4f}")

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

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(fold_dir, 'training_metrics.csv'), index=False)

    # 测试集评估
    print("Evaluating on test set...")
    test_c_index = test_model(model, test_loader, device)
    print(f"Test C-Index: {test_c_index:.4f}")
    with open(os.path.join(fold_dir, 'test_c_index.txt'), 'w') as f:
        f.write(f"Test C-Index: {test_c_index:.4f}")

    # External Test Set 评估
    print("Evaluating on external test set...")
    external_c_index = external_test_model(model, external_loader, device, args.save_dir, fold, experiment_id, "external")
    print(f"External Test C-Index: {external_c_index:.4f}")
    with open(os.path.join(fold_dir, 'external_test_c_index.txt'), 'w') as f:
        f.write(f"External Test C-Index: {external_c_index:.4f}")

    
    return test_c_index, val_c_index, external_c_index



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
