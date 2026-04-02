# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import Embedding, Linear
import torch.nn.functional as F
#from flash_attn.flash_attn_interface import flash_attn_func

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1536

    w_bias: bool = False # use bias tuning
    w_lora: bool = False # use lora tuning
    lora_rank: int = 16
    w_new_gate: bool = False # for compatibility


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.w_bias
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.wq.bias.data, 0)
            nn.init.constant_(self.wo.bias.data, 0)

        self.w_lora = args.w_lora
        if args.w_lora:
           self.lora_wq_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wq_l2 = Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wk_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wk_l2 = Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wv_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wv_l2 = Linear(args.lora_rank, args.dim, bias=False)

           self.lora_wo_l1 = Linear(args.dim, args.lora_rank, bias=False)
           self.lora_wo_l2 = Linear(args.lora_rank, args.dim, bias=False)
           nn.init.constant_(self.lora_wq_l2.weight.data, 0)
           nn.init.constant_(self.lora_wk_l2.weight.data, 0)
           nn.init.constant_(self.lora_wv_l2.weight.data, 0)
           nn.init.constant_(self.lora_wo_l2.weight.data, 0)

        self.cache_k = None
        self.cache_v = None

        self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
        self.w_new_gate = args.w_new_gate
        if args.w_new_gate:
            self.new_gate = torch.nn.Parameter(torch.ones(1, 1, 1, 1))
            
        L = 0.975*2048  # Assume max length represents the high-percentile length
        #g_init 
        self.qk_gate = nn.Parameter( torch.tensor(math.log2(L**2 - L), dtype=torch.float32 ) )

    def train(self, mode: bool = True):
        # Compute L (97.5th percentile sequence length)
        if mode:
            self.cache_k = None
            self.cache_v = None
        else:
            self.cache_k = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
        return super().train(mode)


    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.w_lora:
           xq = xq + self.lora_wq_l2(self.lora_wq_l1(x))
           xk = xk + self.lora_wk_l2(self.lora_wk_l1(x))
           xv = xv + self.lora_wv_l2(self.lora_wv_l1(x))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not self.training:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            assert start_pos==0
            keys = xk
            values = xv

        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_v = self.wv(adapter).view(bsz, adapter_len, self.n_local_heads, self.head_dim)
            adapter_v = adapter_v.transpose(1, 2)

            if adapter_len > 1:
                adapter_k = self.wk(adapter).view(bsz, adapter_len, self.n_local_heads, self.head_dim)
                adapter_k = F.normalize(adapter_k, dim=-1)
                adapter_k = adapter_k.transpose(1, 2)

        #print(f"xq.shape, keys.shape, values.shape {xq.shape} {keys.shape} {values.shape}")
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # print("xq.dtype", xq.dtype, torch.isfinite(xq).all() )
        # print("keys.dtype", keys.dtype, torch.isfinite(keys).all() ) 
        # print("values.dtype", values.dtype, torch.isfinite(values).all() )

        # attempting to do the adapter attn and softmax in fp32 and cast ba ckfp16
        # dtypes
        # xq.dtype torch.float16 tensor(True, device='cuda:0')
        # keys.dtype torch.float16 tensor(True, device='cuda:0')
        # values.dtype torch.float16 tensor(True, device='cuda:0')
        # scores.dtype torch.float16 tensor(True, device='cuda:0')
        # adapter_k.dtype torch.float32 tensor(True, device='cuda:0')
        # adapter_v.dtype torch.float16 tensor(True, device='cuda:0')
        # print("xq.dtype xq.shape", xq.dtype, xq.shape)
        # print("keys.dtype keys.shape", keys.dtype, keys.shape)
        # print("values.dtype values.shape", values.dtype, values.shape)

        # print("scores.dtype", scores.dtype, torch.isfinite(scores).all() )
        
        use_sdpa = (not self.training)
        
        if use_sdpa:
            S_q = xq.shape[2]
            S_k = keys.shape[2]
            
            # decode step with KV cache typically has S_q == 1 and S_k > 1
            is_decode = (S_q == 1 and S_k > 1)
        
            # In decode, causal masking is unnecessary (the query is the last token)
            is_causal = (not is_decode)
            
            # print("flash:", torch.backends.cuda.flash_sdp_enabled())
            # print("mem-efficient:", torch.backends.cuda.mem_efficient_sdp_enabled())
            # print("math:", torch.backends.cuda.math_sdp_enabled())
            # print("is_decode", is_decode)
            # print("is_causal", is_causal)
            
            # Use SDPA (fused attention when available)
            xq = xq.contiguous()
            keys = keys.contiguous()
            values = values.contiguous()

            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=False,
                enable_math=False
            ):
                output = F.scaled_dot_product_attention(
                    xq, keys, values,
                    attn_mask=None,     # safest to start; avoids mask shape issues
                    dropout_p=0.0,
                    is_causal=is_causal
                )  # [B, H, S_q, D]
        
        else:
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
    
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            #print("values.shape", values.shape)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
       
        
        if adapter is not None:
            if adapter_len > 1:
                xq = F.normalize(xq, dim=-1)
                
                # print("adapter_k.dtype",adapter_k.dtype, torch.isfinite(adapter_k).all())
                # print("adapter_v.dtype",adapter_v.dtype, torch.isfinite(adapter_v).all())
                
                adapter_scores = torch.matmul(xq, adapter_k.transpose(2, 3)) # / math.sqrt(self.head_dim)
                adapter_scores = self.qk_gate * adapter_scores
                adapter_scores = self.gate.tanh() * F.softmax(adapter_scores.float(), dim=-1).type_as(xq)
                if self.w_new_gate:
                    adapter_scores = self.new_gate * adapter_scores
                output = output + torch.matmul(adapter_scores, adapter_v)
            else:
                output = output + self.gate.tanh() * adapter_v        
        
        #print("output.shape", output.shape)
        #print(output.shape)
        # if adapter is not None:
        #     if adapter_len > 1:
        #         #print("attention module xq.type", xq.dtype)
        #         adapter_output = flash_attn_func(xq, adapter_k, adapter_v, dropout_p=0.0, causal=True)
        #         #adapter_scores = torch.matmul(xq, adapter_k.transpose(2, 3)) / math.sqrt(self.head_dim)
        #         adapter_output = self.gate.tanh() * adapter_output.type_as(xq)
        #         if self.w_new_gate:
        #             adapter_scores = self.new_gate * adapter_output.type_as(xq) #<- Not sure about type
        #         output = output + adapter_output
        #     else:
        #         output = output + self.gate.tanh() * adapter_v

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        if self.w_lora:
           return self.wo(output) + self.lora_wo_l2(self.lora_wo_l1(output))
        else:
           return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        args: ModelArgs
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=args.w_bias
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.w1.bias.data, 0)
            nn.init.constant_(self.w2.bias.data, 0)
            nn.init.constant_(self.w3.bias.data, 0)

        self.w_lora = args.w_lora
        if args.w_lora:
           self.lora_w1_l1 = Linear(dim, args.lora_rank, bias=False)
           self.lora_w1_l2 = Linear(args.lora_rank, hidden_dim, bias=False)
           self.lora_w2_l1 = Linear(hidden_dim, args.lora_rank, bias=False)
           self.lora_w2_l2 = Linear(args.lora_rank, dim, bias=False)
           self.lora_w3_l1 = Linear(dim, args.lora_rank, bias=False)
           self.lora_w3_l2 = Linear(args.lora_rank, hidden_dim, bias=False)
           nn.init.constant_(self.lora_w1_l2.weight.data, 0)
           nn.init.constant_(self.lora_w2_l2.weight.data, 0)
           nn.init.constant_(self.lora_w3_l2.weight.data, 0)

    def forward(self, x):
        if self.w_lora:
           out = F.silu(self.w1(x) + self.lora_w1_l2(self.lora_w1_l1(x))) * (self.w3(x) + self.lora_w3_l2(self.lora_w3_l1(x)))
           return self.w2(out) + self.lora_w2_l2(self.lora_w2_l1(out))
        else:
           return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, args=args
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        x = x.clamp(min=-65000, max=65000)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)
        #h = h.clamp(min=-65000, max=65000)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        #out = out.clamp(min=-65000, max=65000)
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )
        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
