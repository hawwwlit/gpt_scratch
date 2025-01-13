import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import BaseModel

import math

# transformer: wte, wpe, (decoder blocks), layer norm
# ffn
# --
# decoder: ln -> self attn -> ln -> mlp

class GPTConfig(BaseModel):
    vocab_size: int = 40478
    n_embed: int = 768
    n_layer: int = 12
    n_head: int = 12
    seq_len: int = 512

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x) # apply GELU to learn non-linearity
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.n_embed % config.n_head == 0
        
        # q, k, v
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer('mask', torch.tril(torch.ones(config.seq_len, config.seq_len))
                        .view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x):
        batch_size, seq_len, n_embed = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2) # qkv shape is (batch_size, sequence_length, 3 * n_embed), hence dim=2
        # (B, nh, T, hs)
        q = q.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(2, 1)
        k = k.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(2, 1)
        v = v.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(2, 1)

        # Shape (B, nh, T, T)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.mask[:,:,seq_len, seq_len] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        y = attn @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embed)
        y = self.c_proj(y)
        return y 

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x += self.attn(self.ln1(x))
        x += self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict(dict(
            tokens_embed = nn.Embedding(config.vocab_size, config.n_embd),
            positions_embed = nn.Embedding(config.seq_len, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln = nn.LayerNorm(config.n_embd)
        ))
        self.ffn = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()

        tok_embed = self.transformer.tokens_embed(idx)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_embed = self.transformer.positions_embed(pos)

        x = tok_embed + pos_embed

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln(x)
        logits = self.ffn(x) # (batch_size, seq_len, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # F.cross_entropy expects the target tensor to be a 1D tensor of class indices

        return logits, loss

