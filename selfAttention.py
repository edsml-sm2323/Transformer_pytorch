import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        # embed_size: 嵌入向量的大小，即输入数据的维度
        # heads：头的数量，即要将嵌入向量分割成多少个部分来并行计算注意力。
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads

        # 为了避免除不尽的情况
        assert (self.heads_dim*heads == embed_size), "embed_size和heads除不尽"

        # q,k,v 三个重要的向量。 它们是输入数据的不同表示，用于计算注意力得分
        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)

        # Fully-Connected层，用于将多头注意力的输出再次合并成原始嵌入维度的向量。
        self.fc_out = nn.Linear(heads*self.heads_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # 相当于batch size
        N = query.shape[0]
        # 分别是值、键和查询序列的长度
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 接下来将embedding分割为heads个pieces
        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)
        queries = query.reshape(N, query_len, self.heads, self.heads_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算q, k 得到 a'
        energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # 计算a', v 得到 b
        # 并且进行shape的重塑， 以便进行最后的全连接层处理。
        out = torch.einsum("nhql,nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.heads_dim
        )

        # 全连接层
        out = self.fc_out(out)
        return out