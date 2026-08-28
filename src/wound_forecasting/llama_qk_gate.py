"""Project-specific query/key gate used by the wound LLaMA adapter.

The upstream OpenGVLab transformer remains an external dependency. This module
replaces only its attention class before model construction, preserving the
learned ``qk_gate`` that was present during wound-adapter training.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def install_qk_gated_attention(llama_module):
    """Install the wound-trained attention variant into an upstream module."""
    if getattr(llama_module.Attention, "_wound_qk_gate", False):
        return llama_module.Attention

    apply_rotary_emb = llama_module.apply_rotary_emb
    upstream_attention = llama_module.Attention

    class QKGatedAttention(upstream_attention):
        _wound_qk_gate = True

        def __init__(self, args):
            super().__init__(args)
            sequence_length = 0.975 * 2048
            self.qk_gate = nn.Parameter(
                torch.tensor(
                    math.log2(sequence_length**2 - sequence_length),
                    dtype=torch.float32,
                )
            )

        def forward(self, x, start_pos, freqs_cis, mask, adapter=None):
            batch_size, sequence_length, _ = x.shape
            query = self.wq(x)
            key = self.wk(x)
            value = self.wv(x)

            if self.w_lora:
                query = query + self.lora_wq_l2(self.lora_wq_l1(x))
                key = key + self.lora_wk_l2(self.lora_wk_l1(x))
                value = value + self.lora_wv_l2(self.lora_wv_l1(x))

            query = query.view(
                batch_size,
                sequence_length,
                self.n_local_heads,
                self.head_dim,
            )
            key = key.view(
                batch_size,
                sequence_length,
                self.n_local_heads,
                self.head_dim,
            )
            value = value.view(
                batch_size,
                sequence_length,
                self.n_local_heads,
                self.head_dim,
            )
            query, key = apply_rotary_emb(query, key, freqs_cis=freqs_cis)

            if not self.training:
                self.cache_k = self.cache_k.to(query)
                self.cache_v = self.cache_v.to(query)
                self.cache_k[
                    :batch_size,
                    start_pos : start_pos + sequence_length,
                ] = key
                self.cache_v[
                    :batch_size,
                    start_pos : start_pos + sequence_length,
                ] = value
                keys = self.cache_k[
                    :batch_size,
                    : start_pos + sequence_length,
                ]
                values = self.cache_v[
                    :batch_size,
                    : start_pos + sequence_length,
                ]
            else:
                if start_pos != 0:
                    raise ValueError("Training attention requires start_pos=0")
                keys = key
                values = value

            adapter_length = 0
            if adapter is not None:
                adapter_length = adapter.shape[1]
                adapter_value = self.wv(adapter).view(
                    batch_size,
                    adapter_length,
                    self.n_local_heads,
                    self.head_dim,
                )
                adapter_value = adapter_value.transpose(1, 2)
                if adapter_length > 1:
                    adapter_key = self.wk(adapter).view(
                        batch_size,
                        adapter_length,
                        self.n_local_heads,
                        self.head_dim,
                    )
                    adapter_key = F.normalize(adapter_key, dim=-1)
                    adapter_key = adapter_key.transpose(1, 2)

            query = query.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)

            if not self.training:
                query_length = query.shape[2]
                key_length = keys.shape[2]
                is_decode = query_length == 1 and key_length > 1
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_mem_efficient=False,
                    enable_math=False,
                ):
                    output = F.scaled_dot_product_attention(
                        query.contiguous(),
                        keys.contiguous(),
                        values.contiguous(),
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=not is_decode,
                    )
            else:
                scores = torch.matmul(query, keys.transpose(2, 3))
                scores = scores / math.sqrt(self.head_dim)
                if mask is not None:
                    scores = scores + mask
                scores = F.softmax(scores.float(), dim=-1).type_as(query)
                output = torch.matmul(scores, values)

            if adapter is not None:
                if adapter_length > 1:
                    normalized_query = F.normalize(query, dim=-1)
                    adapter_scores = torch.matmul(
                        normalized_query,
                        adapter_key.transpose(2, 3),
                    )
                    adapter_scores = self.qk_gate * adapter_scores
                    adapter_scores = self.gate.tanh() * F.softmax(
                        adapter_scores.float(),
                        dim=-1,
                    ).type_as(query)
                    if self.w_new_gate:
                        adapter_scores = self.new_gate * adapter_scores
                    output = output + torch.matmul(
                        adapter_scores,
                        adapter_value,
                    )
                else:
                    output = output + self.gate.tanh() * adapter_value

            output = output.transpose(1, 2).contiguous().view(
                batch_size,
                sequence_length,
                -1,
            )
            if self.w_lora:
                return self.wo(output) + self.lora_wo_l2(
                    self.lora_wo_l1(output)
                )
            return self.wo(output)

    QKGatedAttention.__name__ = "Attention"
    QKGatedAttention.__qualname__ = "Attention"
    llama_module.Attention = QKGatedAttention
    return QKGatedAttention
