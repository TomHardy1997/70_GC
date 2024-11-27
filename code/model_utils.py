import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
from typing import Optional


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def custom_collate_fn(batch, load_mode):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    path_features_list, label_list, sur_time_list, censor_list, patient_list = [], [], [], [], []
    gender_list, age_list, num_patch_list, mask_list = [], [], [], []
    
    max_patch_count = max(len(item[6]) for item in batch)

    if load_mode == 'h5':
        coords_list = []
    
    for item in batch:
        patient, gender, age, label, sur_time, censor, features, coords, num_patches = item

        if features.size(0) < max_patch_count:
            padding = torch.zeros(max_patch_count - features.size(0), features.size(1))
            features = torch.cat((features, padding), dim=0)
        
        patient_list.append(patient)
        gender_list.append(1 if gender == "male" else 0) 
        age_list.append(age)
        label_list.append(label)
        sur_time_list.append(sur_time)
        censor_list.append(censor)
        path_features_list.append(features)
        num_patch_list.append(num_patches)

        if load_mode == 'h5' and coords is not None:
            if coords.size(0) < max_patch_count:
                coords_padding = torch.zeros(max_patch_count - coords.size(0), coords.size(1))
                coords = torch.cat((coords, coords_padding), dim=0)
            coords_list.append(coords)

        mask = torch.ones(max_patch_count, dtype=torch.float)
        mask[num_patches:] = 0
        mask_list.append(mask)

    gender_list = torch.tensor(gender_list, dtype=torch.long)
    age_list = torch.tensor(age_list, dtype=torch.float)
    label_list = torch.tensor(label_list, dtype=torch.float)
    sur_time_list = torch.tensor(sur_time_list, dtype=torch.float)
    censor_list = torch.tensor(censor_list, dtype=torch.float)
    path_features = torch.stack(path_features_list, dim=0)
    num_patch_list = torch.tensor(num_patch_list, dtype=torch.long)
    mask_list = torch.stack(mask_list, dim=0)

    if load_mode == 'h5':
        coords_list = torch.stack(coords_list, dim=0)
        return patient_list, gender_list, age_list, label_list, sur_time_list, censor_list, path_features, coords_list, num_patch_list, mask_list
    else:
        return patient_list, gender_list, age_list, label_list, sur_time_list, censor_list, path_features, None, num_patch_list, mask_list


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x, context=None, **kwargs):
        x = self.norm(x)
        if self.norm_context is not None and context is not None:
            # 先对上下文信息进行标准化
            normed_context = self.norm_context(context)
            # 上下文信息在patch维度上进行扩展，确保与x维度一致
            x = torch.cat((x, normed_context.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        return self.fn(x, **kwargs)


class PerceiverAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim  # 如果没有给定 context_dim，则使用 query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = context or x  # 如果没有传递 context，使用 x 自身作为 context
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = mask.to(torch.bool)
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_positions: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_positions, embedding_dim)

    def forward(self, x, mask):
        batch_size, sequence_length, _ = x.shape
        
        # 生成位置信息
        positions = torch.arange(0, sequence_length, device=x.device).unsqueeze(0)  # 形状 (1, sequence_length)
        position_embeddings = self.position_embeddings(positions)  # shape: (1, sequence_length, embedding_dim)

        # 将 mask 用于遮蔽无效位置
        if mask is not None:
            mask = mask.unsqueeze(-1)  # 扩展到 (batch_size, sequence_length, 1)
            position_embeddings = position_embeddings * mask  # 遮蔽无效位置

        return position_embeddings