import torch
import torch.nn as nn
import math

class InfiniteAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(InfiniteAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask=None):
        N, query_len, embed_size = queries.shape
        key_len = keys.shape[1]
        value_len = values.shape[1]

        # Разделение эмбеддингов на num_heads голов
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)

        queries = self.queries(queries)  # (N, query_len, num_heads, head_dim)
        keys = self.keys(keys)            # (N, key_len, num_heads, head_dim)
        values = self.values(values)      # (N, value_len, num_heads, head_dim)

        # Вычисление энегрии (attention scores)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, num_heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, num_heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = InfiniteAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class GPTCustom(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=4096,
        num_layers=48,
        num_heads=64,
        dropout=0.1,
        forward_expansion=4,
        max_length=2048,
    ):
        super(GPTCustom, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(x.device)
        out = self.dropout(self.token_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        logits = self.fc_out(out)
        return logits

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Модель сохранена в {path}")

    def load_model(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"Модель загружена из {path}")

    def generate_text(self, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95, device='cuda'):
        self.eval()
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = tokens

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(generated, mask=None)
                logits = outputs[:, -1, :] / temperature

                # Применение top-k и top-p фильтрации
                filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probabilities = torch.softmax(filtered_logits, dim=-1)

                next_token = torch.multinomial(probabilities, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        return tokenizer.decode(generated[0], skip_special_tokens=True)

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=50, top_p=0.95, filter_value=-float('Inf')):
        """ Функция фильтрации logits с использованием top-k и top-p (nucleus) семплинга """
        assert logits.dim() == 2  # batch_size x vocab_size

        top_k = min(top_k, logits.size(-1))  # Защита от случая, когда top_k > vocab_size

        # Top-K фильтрация
        if top_k > 0:
            # Находим порог top_k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
            logits[indices_to_remove] = filter_value

        # Top-P фильтрация
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Устанавливаем индексы, где накопленная вероятность превышает top_p
            sorted_indices_to_remove = cumulative_probs > top_p

            # Сдвигаем маску на один вправо, чтобы первый токен с cumulative_prob > top_p не был удалён
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            # Возвращаем индексы к исходному расположению
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value

        return logits
