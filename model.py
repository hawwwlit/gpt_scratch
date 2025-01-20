import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import BaseModel

# transformer: wte, wpe, (decoder blocks), layer norm
# ffn
# --
# decoder: ln -> self attn (residual) -> ln -> mlp (residual)

class GPTConfig(BaseModel):
    # 100288 is a better number as it can divided by 32
    vocab_size: int = 100288 # for cl100k_base 
    n_embed: int = 768
    n_layer: int = 12
    n_head: int = 12
    max_seq_len: int = 1024

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

        self.register_buffer('mask', torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                        .view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x):
        batch_size, seq_len, n_embed = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2) # qkv shape is (batch_size, sequence_length, 3 * n_embed), hence dim=2
        # (B, nh, T, hs)
        q = q.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(2, 1)
        k = k.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(2, 1)
        v = v.view(batch_size, seq_len, self.n_head, self.n_embed // self.n_head).transpose(2, 1)

        # Shape (B, nh, T, T)
        # attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # attn = attn.masked_fill(self.mask[:,:,:seq_len, :seq_len] == 0, float('-inf'))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict(dict(
            tokens_embed = nn.Embedding(config.vocab_size, config.n_embed),
            positions_embed = nn.Embedding(config.max_seq_len, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        # weight sharing
        self.transformer.tokens_embed.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights)

        for param_name, param in self.named_parameters():
            if param_name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0., std=0.02/math.sqrt(2*self.config.n_layer))
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of params: {n_params/1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)


    def forward(self, idx, targets=None):

        batch_size, seq_len = idx.size()

        assert seq_len <= self.config.max_seq_len, f"Cannot forward sequence len {seq_len}, max allowed is {self.config.max_seq_len}" 

        tok_embed = self.transformer.tokens_embed(idx)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_embed = self.transformer.positions_embed(pos)

        x = tok_embed + pos_embed

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln(x)
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # flatten out the logits tensor 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # F.cross_entropy expects the target tensor to be a 1D tensor of class indices

        return logits, loss