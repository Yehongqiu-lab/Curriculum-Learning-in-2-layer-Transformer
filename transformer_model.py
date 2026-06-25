"""
References:
nanoGPT:
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """LayerNorm with an optional bias"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd=256, n_head=4, bias=False, dropout=0.1, block_size=128):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        #causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension (n_embd)

        # compute query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # all in size (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side-by-side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
        
class MLP(nn.Module):

    def __init__(self, n_embd=256, bias=False, dropout=0.1):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, n_embd=256, n_head=4, bias=False, dropout=0.1, block_size=128):
        super().__init__()
        self.ln_1 = LayerNorm(ndim=n_embd, bias=bias)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            block_size=block_size
        )
        self.ln_2 = LayerNorm(ndim=n_embd, bias=bias)
        self.mlp = MLP(n_embd=n_embd, bias=bias, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, vocab_size=50304, n_layer=2,
                 n_embd=256, n_head=4, bias=False, dropout=0.1, block_size=128, pad_token=50303):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size
        self.pad_token = pad_token

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(
                n_embd=n_embd, n_head=n_head, bias=bias, dropout=dropout, block_size=block_size
            ) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias)
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying

        # weight initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        # count model parameter size
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (LayerNorm, torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        # validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        param_dict = {pn: p for pn, p in self.named_parameters()}
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer
    
    def forward(self, idx, targets=None, return_features=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # shape (b, t, n_embd)
        if return_features:
            return x

        if targets is not None:
            logits = self.lm_head(x) # shape (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.pad_token)
        else:
            # only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # using list [-1] to preserve the time dim
            loss = None
        return logits, loss

class EarlyStopping:
   def __init__(self, patience=5, min_delta=0.01):
       self.patience = patience
       self.min_delta = min_delta
       self.counter = 0
       self.best_score = None
       self.early_stop = False
   def __call__(self, val_loss, model):
       if self.best_score is None:
           self.best_score = val_loss
           torch.save(model.state_dict(), "best_chkpt")
       elif val_loss > self.best_score - self.min_delta:
           self.counter += 1
           if self.counter >= self.patience:
               self.early_stop = True
       else:
           self.best_score = val_loss
           torch.save(model.state_dict(), "best_chkpt")
           self.counter = 0
           self.early_stop = False


