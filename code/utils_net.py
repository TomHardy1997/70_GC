import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, input, mask):
        # input 形状为 (batch_size, sequence_length, feature_dim)
        batch_size, sequence_length, _ = input.shape

        # 动态设置 num_embeddings
        self.num_embeddings = sequence_length
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        sequence_input = input[:, :, 0]  # 只取序列的第一个维度生成位置信息
        positions = self._make_positions(sequence_input, mask)
        return self.embedding(positions)
    
    def _make_positions(self, tensor, mask):
        # 使用 mask 生成位置索引，mask 为 1 的地方生成位置信息
        return torch.cumsum(mask.long(), dim=1) * mask.long()  # 生成位置信息，填充区域保持为 0
    
    def max_positions(self):
        return self.num_embeddings
    


