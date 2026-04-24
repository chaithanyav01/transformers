
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT
from get_batch import get_batch
from data import vocab_size

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    block_size=64
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

block_size = 64
batch_size = 32

for step in range(2000):
    x, y = get_batch('train', block_size, batch_size)
    x, y = x.to(device), y.to(device)
    
    logits = model(x)
    
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 200 == 0:
        print(f"step {step} loss {loss.item():.4f}")