import torch
import torch.nn as nn
from einops import repeat, rearrange
from model_utils import PreNorm, FeedForward, PerceiverAttention, LearnedPositionalEmbedding


class TransformerBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, PerceiverAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ])
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_classes,
        input_dim=2048,
        dim=512,
        depth=2,
        heads=8,
        mlp_dim=512,
        pool='cls',
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        max_positions=12600
    ):
        super(Transformer, self).__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, heads * dim_head, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim + 2), nn.Linear(dim + 2, num_classes))  # dim + 2 用于加入 age 和 gender

        self.transformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = LearnedPositionalEmbedding(heads * dim_head, max_positions=max_positions)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.apply(self.init_weights)

    def forward(self, x, age, gender, mask=None):
        b, n, _ = x.shape
        x = self.projection(x)
        x = x + self.pos_embedding(x, mask)
        # import ipdb;ipdb.set_trace()
        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            if mask is not None:
                cls_mask = torch.ones(b, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat((cls_mask, mask), dim=1)

        x = self.dropout(x)
        x = self.transformer(x, mask=mask)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # 将 age 和 gender 拼接到 Transformer 的输出
        context = torch.stack((age, gender), dim=-1)  # (b, 2)
        x = torch.cat((x, context), dim=-1)  # 拼接上下文

        return self.mlp_head(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)