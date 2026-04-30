"""
GPT-style decoder-only transformer. Frozen — the candidate does not modify this file.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Config ----
VOCAB_SIZE = 256       # byte-level vocab,
N_LAYER    = 4
N_HEAD     = 4
D_MODEL    = 128
D_HEAD     = D_MODEL // N_HEAD
MAX_SEQ    = 512
SEED       = 1337


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv  = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        # Causal mask, registered as buffer so it moves with .to()
        mask = torch.tril(torch.ones(MAX_SEQ, MAX_SEQ)).view(1, 1, MAX_SEQ, MAX_SEQ)
        self.register_buffer("mask", mask)

    def forward(self, x, past_kv=None):
        """
        x: (B, T, D)
        past_kv: tuple (past_k, past_v) each of shape (B, H, T_past, Dh)
        Returns: (out, new_kv) where new_kv is (k, v) with the FULL accumulated
                 keys/values (i.e. past concatenated with current).
        """
        B, T, _ = x.shape
        qkv = self.qkv(x)                                 # (B, T, 3D)
        q, k, v = qkv.split(D_MODEL, dim=-1)
        # reshape into heads: (B, H, T, Dh)
        q = q.view(B, T, N_HEAD, D_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEAD, D_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, D_HEAD).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)             # (B, H, T_past+T, Dh)
            v = torch.cat([past_v, v], dim=2)

        T_full = k.size(2)
        T_past = T_full - T

        # attention: (B, H, T, T_full)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D_HEAD)
        # Slice the precomputed causal mask. Query positions are
        # [T_past, T_past+T), key positions are [0, T_full).
        causal = self.mask[:, :, T_past:T_past + T, :T_full]
        att = att.masked_fill(causal == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v                                     # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        return self.proj(out), (k, v)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc   = nn.Linear(D_MODEL, 4 * D_MODEL, bias=False)
        self.proj = nn.Linear(4 * D_MODEL, D_MODEL, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(D_MODEL)
        self.mlp  = MLP()

    def forward(self, x, past_kv=None):
        a, new_kv = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ, D_MODEL)
        self.blocks  = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(D_MODEL)
        self.head    = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, input_ids, past_kvs=None):
        """
        input_ids: (B, T)
            past_kvs:  None, or list of length N_LAYER of (k, v) tuples.
            Returns: logits (B, T, V), new_kvs (list of (k, v))
            Position embeddings are indexed from T_past onward, so this works
            whether you pass the full sequence each step (naive) or just the
            new token (with cache).
        """
        B, T = input_ids.shape
        T_past = 0 if past_kvs is None else past_kvs[0][0].size(2)
        assert T_past + T <= MAX_SEQ, "sequence too long"

        positions = torch.arange(T_past, T_past + T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions).unsqueeze(0)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            past = None if past_kvs is None else past_kvs[i]
            x, kv = block(x, past_kv=past)
            new_kvs.append(kv)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_kvs


def build_model(seed: int = SEED) -> TinyGPT:
    """Deterministic constructor. Same seed -> same weights, every time."""
    torch.manual_seed(seed)
    model = TinyGPT()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model