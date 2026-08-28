"""Text-conditioned LVM adapter architecture used for wound forecasting.

This module contains the project-specific text encoder and its injection into
the LVM/LLaMA adapter layers. It deliberately imports the underlying
``ModelArgs`` and ``Transformer`` primitives from a user-supplied LVM checkout
instead of duplicating that upstream implementation.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .llama_qk_gate import install_qk_gated_attention


class SuppressTokensLogitsProcessor:
    """Prevent reserved non-image tokens from being sampled."""

    def __init__(self, suppress_tokens, device: str | torch.device = "cpu"):
        self.suppress_tokens = torch.tensor(list(suppress_tokens), device=device)

    def __call__(self, _input_ids=None, scores: Tensor | None = None) -> Tensor:
        if scores is None:
            scores = _input_ids
        vocabulary = torch.arange(scores.shape[-1], device=scores.device)
        mask = torch.isin(vocabulary, self.suppress_tokens.to(scores.device))
        return torch.where(mask, -float("inf"), scores)


class Attention(nn.Module):
    """Small self-attention layer used by the wound text encoder."""

    def __init__(
        self,
        dimension: int = 64,
        number_of_heads: int = 2,
        query_key_value_bias: bool = False,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ):
        super().__init__()
        # Retain the original attribute names because they are part of the
        # released adapter checkpoint's state-dictionary contract.
        self.attn = nn.MultiheadAttention(
            embed_dim=dimension,
            num_heads=number_of_heads,
            bias=query_key_value_bias,
            dropout=attention_dropout,
        )
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        sequence_first = inputs.permute(1, 0, 2)
        output, _ = self.attn(
            sequence_first,
            sequence_first,
            sequence_first,
            key_padding_mask=mask,
        )
        return self.proj_drop(output.permute(1, 0, 2))


class LightweightBlock(nn.Module):
    """Pre-normalized attention/MLP block for concise prompt tokens."""

    def __init__(
        self,
        dimension: int = 64,
        number_of_heads: int = 2,
        mlp_ratio: float = 2.0,
        query_key_value_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attn = Attention(
            dimension=dimension,
            number_of_heads=number_of_heads,
            query_key_value_bias=query_key_value_bias,
            attention_dropout=dropout,
            projection_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = FeedForward(
            input_features=dimension,
            hidden_features=int(dimension * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        inputs = inputs + self.attn(self.norm1(inputs), mask=mask)
        return inputs + self.mlp(self.norm2(inputs))


class FeedForward(nn.Module):
    """MLP with state names matching the trained text-adapter checkpoint."""

    def __init__(
        self,
        *,
        input_features: int,
        hidden_features: int,
        dropout: float,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, input_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.drop1(self.act(self.fc1(inputs)))
        return self.drop2(self.fc2(inputs))


class TextEncoder(nn.Module):
    """Encode concise prompts into learned text-adapter query tokens."""

    def __init__(
        self,
        *,
        vocabulary_size: int = 65_536,
        embedding_dimension: int = 64,
        sequence_length: int = 32,
        number_of_layers: int = 1,
        number_of_heads: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        adapter_sequence_length: int = 32,
        padding_index: int = 16_383,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension,
            padding_idx=padding_index,
        )
        self.positional_embedding = nn.Embedding(
            sequence_length,
            embedding_dimension,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                LightweightBlock(
                    dimension=embedding_dimension,
                    number_of_heads=number_of_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(number_of_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dimension)
        self.query_len = adapter_sequence_length
        self.learned_query = nn.Parameter(
            torch.randn(adapter_sequence_length, embedding_dimension)
        )

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, sequence_length = tokens.shape
        positions = torch.arange(sequence_length, device=tokens.device).unsqueeze(0)
        encoded = self.embedding(tokens) + self.positional_embedding(positions)
        queries = self.learned_query.unsqueeze(0).repeat(batch_size, 1, 1)
        encoded = torch.cat([queries, encoded], dim=1)
        for block in self.encoder_blocks:
            encoded = block(encoded, mask=mask)
        return self.norm(encoded)[:, : self.query_len, :]


def _load_lvm_primitives(source_root: str | Path):
    """Load only the transformer primitives required from upstream LVM."""
    root = Path(source_root).expanduser().resolve()
    required = [root / "llama" / "llama.py", root / "llama" / "utils.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LVM source files are missing: {missing}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    llama_module = importlib.import_module("llama.llama")
    utility_module = importlib.import_module("llama.utils")
    install_qk_gated_attention(llama_module)
    return llama_module.ModelArgs, llama_module.Transformer, utility_module.sample_top_p


class WoundLLaMAAdapter(nn.Module):
    """LVM adapter with concise text queries injected into late layers."""

    def __init__(
        self,
        llama_checkpoint_dir: str | Path,
        *,
        upstream_source_root: str | Path,
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
        query_layer: int = 20,
        img_adapter_len: int = 64,
        text_adapter_len: int = 64,
        use_text: bool = True,
        text_in_layers: int = 8,
        w_bias: bool = False,
        w_lora: bool = False,
        lora_rank: int = 16,
        w_new_gate: bool = False,
        phase: str = "finetune",
        **_unused,
    ):
        super().__init__()
        model_args_class, transformer_class, sample_top_p = _load_lvm_primitives(
            upstream_source_root
        )
        self._sample_top_p = sample_top_p
        checkpoint_directory = Path(llama_checkpoint_dir).expanduser().resolve()
        with (checkpoint_directory / "params.json").open(encoding="utf-8") as stream:
            parameters = json.load(stream)

        w_bias = phase == "finetune"
        model_args = model_args_class(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **parameters,
        )
        self.query_len = img_adapter_len + text_adapter_len
        self.query_layer = query_layer
        self.img_adapter_len = img_adapter_len
        self.text_adapter_len = text_adapter_len
        self.use_text = use_text
        self.text_in_layers = text_in_layers
        self.scale_factor = 1.0

        self.text_dim = 64
        self.text_encoder = TextEncoder(
            vocabulary_size=65_536,
            embedding_dimension=self.text_dim,
            sequence_length=32,
            number_of_layers=1,
            number_of_heads=2,
            mlp_ratio=2.0,
            dropout=0.1,
            adapter_sequence_length=text_adapter_len,
            padding_index=16_383,
        )
        self.text_proj = nn.Linear(self.text_dim, model_args.dim)
        self.text_proj_norm = nn.LayerNorm(model_args.dim, elementwise_affine=True)
        self.adapter_query = nn.Embedding(
            img_adapter_len * query_layer,
            model_args.dim,
        )

        self.pad_id = 8192
        self.ignore_idx = -100
        model_args.w_bias = w_bias
        model_args.w_lora = w_lora
        model_args.lora_rank = lora_rank
        model_args.w_new_gate = w_new_gate
        model_args.vocab_size = 8292

        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        try:
            self.llama = transformer_class(model_args)
        finally:
            torch.set_default_tensor_type(torch.FloatTensor)

        checkpoints = sorted(checkpoint_directory.glob("*.pth"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No LVM base checkpoints found in {checkpoint_directory}"
            )
        for checkpoint_path in checkpoints:
            state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.llama.load_state_dict(state, strict=False)

        suppress_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.suppress_tokens = list(range(8192, 8292))
        self.suppress_tokens_processor = SuppressTokensLogitsProcessor(
            self.suppress_tokens,
            device=suppress_device,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)
        self.phase = phase
        self.get_trainable_params(phase)

    def get_trainable_params(self, phase: str = "finetune") -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        if phase == "finetune":
            for name, parameter in self.named_parameters():
                if name.startswith("llama.") and "norm" in name:
                    parameter.data = parameter.data.float()
                    parameter.requires_grad = True
        elif phase == "pretrain":
            selected = (
                "gate",
                "text_encoder",
                "text_proj",
                "text_proj_norm",
                "adapter_query",
                "qk_gate",
            )
            for name, parameter in self.named_parameters():
                if any(component in name for component in selected):
                    parameter.data = parameter.data.float()
                    parameter.requires_grad = True
        else:
            raise ValueError(f"Unknown model phase: {phase}")

    def forward_text(self, text_tokens: Tensor) -> Tensor:
        text_query = self.text_encoder(text_tokens, mask=None)
        if not torch.isfinite(text_query).all():
            raise RuntimeError("text_encoder produced NaN/Inf")
        return self.text_proj_norm(self.text_proj(text_query))

    def inference_forward_text(self, text_tokens: Tensor) -> Tensor:
        return self.text_proj_norm(self.text_proj(self.text_encoder(text_tokens)))

    def forward(self, tokens: Tensor, labels: Tensor, text_tokens: Tensor):
        text_query = self.forward_text(text_tokens) if self.use_text else None
        batch_size, sequence_length = tokens.shape
        hidden = self.llama.tok_embeddings(tokens)
        frequencies = self.llama.freqs_cis.to(hidden.device)[:sequence_length]
        mask = torch.full(
            (1, 1, sequence_length, sequence_length),
            float("-inf"),
            device=hidden.device,
        )
        mask = torch.triu(mask, diagonal=1).type_as(hidden)

        for layer in self.llama.layers[: -self.query_layer]:
            hidden = checkpoint(
                layer,
                hidden,
                0,
                frequencies,
                mask,
                use_reentrant=False,
            )

        adapters = self.adapter_query.weight.reshape(
            self.query_layer,
            self.img_adapter_len,
            -1,
        ).unsqueeze(1)
        for adapter_index, layer in enumerate(self.llama.layers[-self.query_layer :]):
            dynamic_adapter = adapters[adapter_index].repeat(batch_size, 1, 1)
            if (
                self.use_text
                and text_query is not None
                and adapter_index >= self.query_layer - self.text_in_layers
            ):
                dynamic_adapter = torch.cat(
                    [dynamic_adapter, self.scale_factor * text_query],
                    dim=1,
                )
            if adapter_index < 12:
                hidden = checkpoint(
                    layer,
                    hidden,
                    0,
                    frequencies,
                    mask,
                    dynamic_adapter,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, 0, frequencies, mask, dynamic_adapter)

        hidden = self.llama.norm(hidden.clamp(min=-65_000, max=65_000))
        output = self.suppress_tokens_processor(scores=self.llama.output(hidden))
        output = output[:, :-1, :]
        shifted_labels = labels[:, 1:]
        if shifted_labels.sum() == 0:
            loss = output.mean() * 0
        else:
            loss = self.criterion(
                output.reshape(-1, self.llama.vocab_size),
                shifted_labels.flatten(),
            )
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite LLaMA adapter loss: {loss}")
        return loss, loss

    @torch.inference_mode()
    def forward_inference(
        self,
        tokens: Tensor,
        text_query: Tensor,
        start_position: int,
    ) -> Tensor:
        batch_size, sequence_length = tokens.shape
        hidden = self.llama.tok_embeddings(tokens)
        frequencies = self.llama.freqs_cis.to(hidden.device)[
            start_position : start_position + sequence_length
        ]
        mask = torch.full(
            (1, 1, sequence_length, sequence_length),
            float("-inf"),
            device=hidden.device,
        )
        mask = torch.triu(mask, diagonal=start_position + 1).type_as(hidden)
        for layer in self.llama.layers[: -self.query_layer]:
            hidden = layer(hidden, start_position, frequencies, mask)

        adapters = self.adapter_query.weight.reshape(
            self.query_layer,
            self.img_adapter_len,
            -1,
        ).unsqueeze(1)
        for adapter_index, layer in enumerate(self.llama.layers[-self.query_layer :]):
            dynamic_adapter = adapters[adapter_index].repeat(batch_size, 1, 1)
            if (
                self.use_text
                and adapter_index >= self.query_layer - self.text_in_layers
            ):
                dynamic_adapter = torch.cat(
                    [dynamic_adapter, self.scale_factor * text_query],
                    dim=1,
                )
            hidden = layer(
                hidden,
                start_position,
                frequencies,
                mask,
                dynamic_adapter,
            )

        hidden = self.llama.norm(hidden.clamp(min=-65_000, max=65_000))
        return self.llama.output(hidden[:, -1, :]).float()

    @torch.inference_mode()
    def generate(
        self,
        img_tokens: Tensor,
        text_tokens: Tensor,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ) -> Tensor:
        batch_size = len(img_tokens)
        parameters = self.llama.params
        if batch_size > parameters.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds {parameters.max_batch_size}"
            )
        if len(img_tokens) != len(text_tokens):
            raise ValueError("Image-token and text-token batch sizes differ")
        if self.use_text:
            with torch.amp.autocast("cuda", enabled=img_tokens.is_cuda):
                text_query = self.inference_forward_text(text_tokens)
        else:
            text_query = torch.empty(0, device=img_tokens.device)

        minimum_prompt = min(len(tokens) for tokens in img_tokens)
        maximum_prompt = max(len(tokens) for tokens in img_tokens)
        total_length = min(parameters.max_seq_len, max_gen_len + maximum_prompt)
        tokens = torch.full(
            (batch_size, total_length),
            self.pad_id,
            device=img_tokens.device,
            dtype=torch.long,
        )
        for index, prompt in enumerate(img_tokens):
            tokens[index, : len(prompt)] = prompt.to(tokens.device, dtype=torch.long)

        prompt_mask = tokens != self.pad_id
        previous_position = 0
        for current_position in range(minimum_prompt, total_length):
            logits = self.forward_inference(
                tokens[:, previous_position:current_position],
                text_query,
                previous_position,
            )
            logits = self.suppress_tokens_processor(scores=logits)
            if temperature > 0:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                next_token = self._sample_top_p(probabilities, top_p).reshape(-1)
            else:
                next_token = torch.argmax(logits, dim=-1).reshape(-1)
            next_token = torch.where(
                prompt_mask[:, current_position],
                tokens[:, current_position],
                next_token,
            )
            tokens[:, current_position] = next_token
            previous_position = current_position
        return tokens
