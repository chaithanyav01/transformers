import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1. DATA
# -----------------------------
text = open("input.txt", "r", encoding="utf-8").read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y


# -----------------------------
# 2. MODEL
# -----------------------------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        mask = torch.tril(torch.ones(T, T, device=x.device))
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        
        pos = torch.arange(T, device=x.device)
        
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        return self.head(x)


# -----------------------------
# 3. TRAIN
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(
    vocab_size=vocab_size,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    block_size=64
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

block_size = 64
batch_size = 32

# for step in range(1000):
#     x, y = get_batch('train', block_size, batch_size)
#     x, y = x.to(device), y.to(device)
    
#     logits = model(x)
    
#     B, T, V = logits.shape
#     loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if step % 200 == 0:
#         print(f"step {step} loss {loss.item():.4f}")

# Save the model
# torch.save(model.state_dict(), "model.pt")

# -----------------------------
# 4. GENERATION
# -----------------------------
def generate(model, start, max_new_tokens=100):
    model.eval()
    x = torch.tensor([encode(start)], dtype=torch.long).to(device)
    
    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]
        logits = model(x_cond)
        
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        x = torch.cat([x, next_token], dim=1)
    
    return decode(x[0].tolist())



# load the model for inference
model.load_state_dict(torch.load("model.pt"))

# -----------------------------
# 5. TEST
# -----------------------------
print("\nGenerated:\n")
print(generate(model, start="Shubham Education", max_new_tokens=300))