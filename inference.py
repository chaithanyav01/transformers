import torch
import torch.nn.functional as F
from model import GPT
from data import encode, decode

device = "cuda" if torch.cuda.is_available() else "cpu"

model =  torch.load("model.pt")
model.to(device)

def generate(model, start, max_new_tokens=100):
    model.eval()
    
    x = torch.tensor([encode(start)], dtype=torch.long).to(device)
    
    for _ in range(max_new_tokens):
        x_cond = x[:, -64:]  # block_size
        
        logits = model(x_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        
        next_token = torch.multinomial(probs, 1)
        x = torch.cat([x, next_token], dim=1)
    
    return decode(x[0].tolist())


print(generate(model, "The ", 200))