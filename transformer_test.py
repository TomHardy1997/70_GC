import torch
import torch.nn as nn
from einops import repeat, rearrange
# from torch.utils.checkpoint import checkpoint

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        input_dim=1024,
        dim=512,
        depth=4,
        heads=8,
        mlp_dim=1024,
        pool='cls',
        dim_head=128,
        dropout=0.1,
        emb_dropout=0.1,
        use_DropKey=False,  # Added DropKey option
        mask_ratio=0.1      # Mask ratio for DropKey
    ):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (class token) or mean pooling'

        self.projection = nn.Sequential(nn.Linear(input_dim, dim, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.transformer = nn.ModuleList([TransformerBlock(dim, heads, dim_head, mlp_dim, dropout, use_DropKey, mask_ratio) for _ in range(depth)])

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        b, n, d = x.shape
        # import ipdb;ipdb.set_trace()-
        x = self.projection(x)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            mask = torch.cat((torch.ones(b, 1, device=mask.device), mask), dim=1)
            # mask = torch.cat((torch.ones(b, 1, device=x.device), mask), dim=1) 
        x = self.dropout(x)
        for layer in self.transformer:
            x = layer(x, mask=mask)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(self.norm(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, use_DropKey, mask_ratio):
        super().__init__()
        # import ipdb;ipdb.set_trace()
        self.attn = PreNorm(dim, Attention(dim, heads, dim_head, dropout, use_DropKey, mask_ratio))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout))

    def forward(self, x, mask=None):
        x = self.attn(x, mask)  
        x = self.ff(x)     
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # import ipdb;ipdb.set_trace()
        return self.fn(self.norm(x), mask=mask)

class Attention(nn.Module):
    def __init__(self, dim=128, heads=8, dim_head=32, dropout=0.1, use_DropKey=False, mask_ratio=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_DropKey = use_DropKey  # Whether to use DropKey
        self.mask_ratio = mask_ratio    # Mask ratio for DropKey

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # import ipdb;ipdb.set_trace()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots = dots.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = self.attend(dots)
        
        # Apply DropKey if enabled
        if self.use_DropKey:
            m_r = torch.ones_like(attn) * self.mask_ratio
            attn = attn + torch.bernoulli(m_r) * -1e12  # Use -1e12 to mask dropped keys
        
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        return self.to_out(x)

class FeedForward(nn.Module):
    def __init__(self, dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()  
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        return self.net(x)
