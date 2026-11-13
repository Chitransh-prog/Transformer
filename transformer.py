import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Positional Encoding  (Section 3.4) ────────────────────────────────────────
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, d_model]
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Scaled Dot-Product Attention  (Section 3.2) ───────────────────────────────
# Attention(Q,K,V) = softmax( Q K^T / √d_k ) V

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)

    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)

    return torch.matmul(weights, v), weights


# ── Multi-Head Attention  (Section 3.2) ───────────────────────────────────────
# MultiHead(Q,K,V) = Concat(head_1, …, head_h) W_O
# head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k     = d_model // n_heads
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # [B, T, d_model] → [B, h, T, d_k]
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q, k, v = self._split_heads(self.w_q(q)), \
                  self._split_heads(self.w_k(k)), \
                  self._split_heads(self.w_v(v))

        x, self.attn_weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # [B, h, T, d_k] → [B, T, d_model]
        x = x.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.w_o(x)


# ── Feed-Forward Network  (Section 3.3) ───────────────────────────────────────
# FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ── Encoder Layer  (Section 3.1) ──────────────────────────────────────────────
# Sub-layer 1: Self-attention  |  Sub-layer 2: FFN
# Each wrapped in:  LayerNorm( x + Dropout( Sublayer(x) ) )

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff        = FeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ── Decoder Layer  (Section 3.1) ──────────────────────────────────────────────
# Sub-layer 1: Masked self-attention
# Sub-layer 2: Cross-attention  (Q from decoder, K/V from encoder)
# Sub-layer 3: FFN

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff         = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


# ── Full Transformer  (Section 3) ─────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer.

    Paper defaults: d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1
    College scale:  d_model=128, n_heads=4, n_layers=2, d_ff=512
    """
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model:  int   = 128,
        n_heads:  int   = 4,
        n_layers: int   = 2,
        d_ff:     int   = 512,
        dropout:  float = 0.1,
        max_len:  int   = 512,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)

        # Output projection: maps hidden states → vocabulary logits
        self.output_proj = nn.Linear(d_model, tgt_vocab)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, emb_layer, tokens):
        # Scale embeddings by √d_model (Section 3.4)
        return self.pos_enc(emb_layer(tokens) * math.sqrt(self.d_model))

    def encode(self, src, src_mask):
        x = self._embed(self.src_emb, src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self._embed(self.tgt_emb, tgt)
        for layer in self.decoder:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.dec_norm(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.output_proj(dec_out)   # [B, T, tgt_vocab]


# ── Mask Utilities ────────────────────────────────────────────────────────────

def pad_mask(seq, pad_idx=0):
    """[B, T] → [B, 1, 1, T]  (True = keep, False = ignore <pad>)"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def causal_mask(size, device):
    """Lower-triangular [1,1,T,T] mask preventing look-ahead in the decoder."""
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(1)

def make_masks(src, tgt, pad_idx=0):
    src_mask = pad_mask(src, pad_idx)
    tgt_mask = pad_mask(tgt, pad_idx) & causal_mask(tgt.size(1), tgt.device)
    return src_mask, tgt_mask


# ── Warm-up Learning Rate Schedule  (Section 5.3) ────────────────────────────
# lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))

class WarmupScheduler:
    def __init__(self, optimizer, d_model: int, warmup: int = 400):
        self.opt     = optimizer
        self.d_model = d_model
        self.warmup  = warmup
        self.step_n  = 0

    def step(self):
        self.step_n += 1
        s = self.step_n
        lr = self.d_model ** -0.5 * min(s ** -0.5, s * self.warmup ** -1.5)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()

