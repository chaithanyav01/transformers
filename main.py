import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1) # q - (B, T, C), k - (B, T, C), v - (B, T, C)
        
        # reshape → (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # attention scores
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, heads, T, T) - attention scores for each head 
        
        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)) # (T, T) - lower triangular matrix to prevent attending to future tokens
        attn = attn.masked_fill(mask == 0, float('-inf')) # (B, heads, T, T) - set future token attention scores to -inf
        
        attn = F.softmax(attn, dim=-1) # (B, heads, T, T) - normalized attention scores
        
        out = attn @ v  # (B, heads, T, head_dim) - weighted sum of values based on attention scores
        
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C) - concatenate heads and reshape back to original embedding dimension
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, expansion * embed_dim),
            nn.GELU(),
            nn.Linear(expansion * embed_dim, embed_dim)
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
        # Pre-norm attention
        x = x + self.attn(self.ln1(x))
        
        # Pre-norm feedforward
        x = x + self.ff(self.ln2(x))
        
        return x

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_length = 5
    embed_dim = 16
    num_heads = 4
    
    x = torch.randn(batch_size, seq_length, embed_dim, device='cpu') # B T C - B sentences, T tokens, C embedding dimension
    block = TransformerBlock(embed_dim, num_heads)
    
    out = block(x)
    print(out.shape)  # Should be (batch_size, seq_length, embed_dim) - B T C - B sentences, T tokens, C embedding dimension