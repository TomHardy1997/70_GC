import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import logging
import argparse
import ast

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SwinPrognosisDataset(Dataset):
    def __init__(self, df, data_dir, is_training=True):
        self.df = pd.read_csv(df)
        self.patient = self.df['case_id']
        self.label = self.df['label']
        self.censor = self.df['censor']
        # self.censor = self.df['status']
        self.time = self.df['survival_months']
        self.wsi = self.df['slide_id'].apply(lambda x: ast.literal_eval(x.strip()))
        self.data_dir = data_dir
        logging.info("SwinPrognosisDataset initialized with {} samples".format(len(self.patient)))
    
    def __len__(self):
        return len(self.patient)

    def __getitem__(self, idx):
        patient = self.patient[idx]
        label = self.label[idx]
        censor = self.censor[idx]
        sur_time = self.time[idx]
        slide_ids = self.wsi[idx]
        path_features = []
        for slide_id in slide_ids:
            slide_id = slide_id.strip()
            wsi_path = os.path.join(self.data_dir, slide_id)
            try:
                wsi_bag = torch.load(wsi_path, weights_only=True)
                path_features.append(wsi_bag)
            except FileNotFoundError:
                logging.error(f"File not found: {wsi_path}")
                continue
            except RuntimeError as e:
                logging.error(f"Error loading file {wsi_path}: {e}")
                continue
        num_patches = len(path_features) 
        if path_features:
            path_features = torch.cat(path_features, dim=0)
        else:
            path_features = torch.tensor([])


        return path_features, label, sur_time, censor, patient, num_patches


import torch.nn.utils.rnn as rnn_utils

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None 

    path_features_list, label_list, sur_time_list, censor_list, patient_list, mask_list = [], [], [], [], [], []

    # 获取当前批次的最大补丁数量
    max_patch_count = max(len(item[0]) for item in batch)
 
    for item in batch:

        path_features, label, sur_time, censor, patient, _ = item


        # 进行补零
        if path_features.size(0) < max_patch_count:
            padding = torch.zeros(max_patch_count - path_features.size(0), path_features.size(1))
            path_features = torch.cat([path_features, padding], dim=0)

        path_features_list.append(path_features)
        label_list.append(label)
        sur_time_list.append(sur_time)
        censor_list.append(censor)
        patient_list.append(patient)

        # 创建掩码
        mask = torch.ones(max_patch_count, dtype=torch.float)
        if len(path_features) < max_patch_count:
            mask[len(path_features):] = 0  # 填充部分掩码设置为0
        mask_list.append(mask)

    path_features_tensor = torch.stack(path_features_list)
    label_tensor = torch.tensor(label_list, dtype=torch.float)
    sur_time_tensor = torch.tensor(sur_time_list, dtype=torch.float)
    censor_tensor = torch.tensor(censor_list, dtype=torch.float)

    mask_tensor = torch.stack(mask_list) if mask_list else None
    return path_features_tensor, label_tensor, sur_time_tensor, censor_tensor, patient_list, mask_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tumor and SwinPrognosis Dataset")
    parser.add_argument('--df', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of the data')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    args = parser.parse_args()

    dataset = SwinPrognosisDataset(df=args.df, data_dir=args.data_dir, is_training=True)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )


    for data in data_loader:
        print(data)
        import ipdb;ipdb.set_trace()
        break
