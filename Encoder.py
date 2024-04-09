import torch
import torch.nn as nn
from selfAttention import SelfAttention
from TransformerBlock import TransformerBlock

class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size =embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                ) for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        position = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # 对输入进行embedding和position向量的计算, 并且最后相加
        out = self.dropout(self.word_embedding(x) + self.position_embedding(position))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
