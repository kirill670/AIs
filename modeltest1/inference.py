import torch
from transformers import AutoTokenizer
from GPTCustom import GPTCustom  # Предполагается, что модель сохранена в your_model_file.py

# Параметры
model_path = "path_to_saved_model.pt"
vocab_size = 50257
embed_size = 4096
num_layers = 48
num_heads = 64
dropout = 0.1
forward_expansion = 4
max_length = 2048

# Инициализация модели
model = GPTCustom(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_layers=num_layers,
    num_heads=num_heads,
    dropout=dropout,
    forward_expansion=forward_expansion,
    max_length=max_length
)
model.load_model(model_path)
model.eval()
model.to('cuda')  # Перенос модели на GPU, если доступно

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    generated = inputs

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated, mask=None)
            logits = outputs[:, -1, :] / temperature

            # Применение top-k и top-p семплинга
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

def top_k_top_p_filtering(logits, top_k=50, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch_size, vocab_size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering)
            top_p <1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering)
    """
    assert logits.dim() == 2  # batch_size x vocab_size

    top_k = min(top_k, logits.size(-1))  # Safety check

    # Top-K filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    # Top-P filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits

if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_text = generate_text(prompt, max_length=100)
    print(generated_text)
