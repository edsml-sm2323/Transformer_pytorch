import torch
import torch.nn as nn
from selfAttention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 论文中forward_expansion选的是4
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # 第一个残差结构
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)

        # 第二个残差结构
        out = self.dropout(self.norm2(forward + x))

        return out
