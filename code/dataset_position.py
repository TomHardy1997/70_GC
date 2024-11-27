import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import logging
import argparse
import ast
import h5py
import numpy as np
from model_utils import custom_collate_fn


class SwinPrognosisDataset(Dataset):
    def __init__(self, df, pt_dir, h5_dir=None, load_mode='pt'):
        """
        Args:
            df: Path to CSV file containing the metadata.
            pt_dir: Directory containing the .pt files.
            h5_dir: Directory containing the .h5 files (only required if load_mode is 'h5').
            load_mode: Specify whether to load 'pt' files or 'h5' files. Default is 'pt'.
        """
        self.df = pd.read_csv(df)
        self.patient = self.df["case_id"]
        self.gender = self.df["gender"]
        self.age = pd.to_numeric(self.df["age_at_index"], errors='coerce')
        self.label = self.df["label"]
        self.time = self.df['survival_months']
        self.censor = self.df['censor']
        self.pt_dir = pt_dir
        self.h5_dir = h5_dir
        self.load_mode = load_mode
        self.wsi = self.df['slide_id'].apply(lambda x: ast.literal_eval(x.strip()))
        # import ipdb;ipdb.set_trace()
    def __len__(self):
        return len(self.patient)

    def __getitem__(self, idx):
        patient = self.patient[idx]
        gender = self.gender[idx]
        age = self.age[idx]
        label = self.label[idx]
        sur_time = self.time[idx]
        censor = self.censor[idx]
        slide_ids = self.wsi[idx]
        path_features = []
        path_coords = []

        
        if self.load_mode == 'pt':
            
            for slide_id in slide_ids:
                slide_id = slide_id.strip()
                wsi_path = os.path.join(self.pt_dir, slide_id)
                try:
                    wsi_bag = torch.load(wsi_path, weights_only=True)
                    path_features.append(wsi_bag)
                except FileNotFoundError as e:
                    print(f'File not found {wsi_path}: {e}')
                    continue
                except RuntimeError as e:
                    print(f'RuntimeError {wsi_path}: {e}')
                    continue

            if path_features:
                features = torch.cat(path_features, dim=0)
            else:
                features = torch.tensor([])

            num_patches = features.shape[0]
            
            return patient, gender, age, label, sur_time, censor, features, None, num_patches

        elif self.load_mode == 'h5':
            
            for slide_id in slide_ids:
                h5_id = slide_id.strip().replace('.pt', '.h5')
                h5_path = os.path.join(self.h5_dir, h5_id)
                try:
                    with h5py.File(h5_path, 'r') as f:
                        features = f['features'][:]
                        coords = f['coords'][:]
                        path_features.append(torch.tensor(features))
                        path_coords.append(torch.tensor(coords))
                except FileNotFoundError as e:
                    print(f'File not found {h5_path}: {e}')
                    continue
                except RuntimeError as e:
                    print(f'RuntimeError {h5_path}: {e}')
                    continue

            if path_features:
                features = torch.cat(path_features, dim=0)
                coords = torch.cat(path_coords, dim=0)
            else:
                features = torch.tensor([])
                coords = torch.tensor([])

            num_patches = features.shape[0]
            
            return patient, gender, age, label, sur_time, censor, features, coords, num_patches




if __name__ == '__main__':
    df = 'TCGA_ALL_PT_LABEL.csv'
    pt_dir = '/home/stat-jijianxin/SCI_GC_OS/40x_tcga_merged_folder'
    load_mode = 'pt'  
    dataset = SwinPrognosisDataset(df, pt_dir, load_mode=load_mode)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=lambda batch: custom_collate_fn(batch, load_mode))
    import ipdb;ipdb.set_trace()



