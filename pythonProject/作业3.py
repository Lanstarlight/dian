import torch
import torch.nn.functional as F


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = torch.nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, return_attention_weights=False):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into multiple heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Calculate the attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Apply attention to the values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        if return_attention_weights:
            return out, attention
        else:
            return out


# 下面的代码用来测试和展示如何使用修改后的MultiHeadAttention
embed_size = 256
heads = 8
values = torch.rand((5, 60, embed_size))
keys = torch.rand((5, 60, embed_size))
queries = torch.rand((5, 40, embed_size))

attention_model = MultiHeadAttention(embed_size, heads)
output, attention_weights = attention_model(values, keys, queries, return_attention_weights=True)

print("Output shape:", output.shape)  # Expected: [5, 40, 256]
print("Attention weights shape:", attention_weights.shape)  # Expected: [5, num_heads, 40, 60] or similar based on heads
