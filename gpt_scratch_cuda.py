import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import math
import tiktoken
import time
import inspect 

from model import GPT, GPTConfig

os.environ['CC'] = 'gcc'

# set seed before model initialization! 
# all ddp processes will share the same seed
torch.manual_seed(922)
if torch.cuda.is_available():
    torch.cuda.manual_seed(922)
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(922)

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "need CUDA for DDP"
    ddp_rank = int(os.environ.get('RANK'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_local_rank == 0
    dist.init_process_group('nccl')
else:
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    master_process = True
    print(f"Using device: {device}")


max_steps = 50
total_batch_size = 2**19
B, T = 16, 1024
assert total_batch_size % (B*T*ddp_world_size) == 0, "total batch size should be dvisible by B*T*ddp_world_size"
grad_accum_step = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"desired batch size: {total_batch_size} => calculated grad accumulated steps: {grad_accum_step}")


# initialize the model with random weights
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class DataLoaderLite:
    def __init__(self, input_file, B, T, rank, num_processes, enc='cl100k_base'):
        enc = tiktoken.get_encoding(enc)
        with open(input_file, 'r') as f:
            text = f.read()
        self.tokens = torch.tensor(enc.encode(text))
        if rank == 0:
            print(f"Loaded {len(self.tokens)} tokens")
            print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        self.B = B
        self.T = T
        self.rank = rank
        self.num_processes = num_processes
        self.enc = enc
        self.current_position = rank * B * T
    
    def next_batch(self):
        buf = self.tokens[self.current_position: self.current_position+self.B*self.T+1]
        x, y = buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)
        self.current_position += self.B*self.T*self.num_processes
        if self.current_position+self.B*self.T> len(self.tokens):
            self.current_position = self.rank * self.B * self.T
        return x, y 

train_loader = DataLoaderLite('input.txt', B=16, T=1024, rank=ddp_rank, num_processes=ddp_world_size)

# torch.set_float32_matmul_precision('high')
extra_kargs = dict(fused=True) if 'fused' in inspect.signature(torch.optim.AdamW).parameters and device=='cuda' else dict()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), **extra_kargs)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=50)

for i in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.
    for j in range(grad_accum_step):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_step
        loss_accum += loss.detach() # by default, in DDP, it will only be accum_loss of master process, so need to do something
        if ddp:
            model.require_backward_grad_sync = (j == grad_accum_step-1)
        loss.backward() 
        # by default with DDP, the gradient syncing will happen after the loss.backward() where the graidents are calculated. 

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # clipped_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters()]))
    # import code; code.interact(local=locals())

    """
    >>> optimizer.param_groups[0].keys()
    dict_keys(['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach', 'maximize', 'capturable', 'differentiable', 'fused'])
    """
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) * grad_accum_step * ddp_world_size / (t1-t0)
    
    if i == 0:         # sanity check
        print(f"expected initial loss is {-math.log(1/(train_loader.enc.max_token_value+1)):.3f} loss; actual is {loss_accum:.3f}")
    
    if master_process:
        print(f"Step: {i} | loss: {loss_accum.item():.5f} | norm: {norm:.2f} | lr: {scheduler.get_last_lr()[0]:.4e} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    dist.destroy_process_group()

import sys; sys.exit(0)
batch_size = 5
max_seq_len = 30
tokens = enc.encode("Here in Seattle, ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(batch_size, 1)
x = tokens.to(device)

model.eval()
t0 = time.time()
while x.size()[-1] < max_seq_len:
    with torch.no_grad():
        logits, _ = model(x) # (b, t, vocab_size)
        logits = logits[:, -1, :] # (b, vocab_size)
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, dim=1, index=ix) # returns the same result with dim=-1
        x = torch.cat((x, xcol), dim=1)
t1 = time.time()
print(f"inference done after {(t1-t0)*1000:.3f} milliseconds")
for i in range(batch_size):
    tokens = x[i, :max_seq_len].tolist()
    decoded = enc.decode(tokens)
    print('>', decoded)