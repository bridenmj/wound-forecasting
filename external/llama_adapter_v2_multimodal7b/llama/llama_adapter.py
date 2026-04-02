import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from torch.utils.checkpoint import checkpoint # Checkpoint first layer

from .llama import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download

from transformers import LogitsProcessor
from typing import List
import math

from timm.models.layers import Mlp

class SuppressTokensLogitsProcessor(LogitsProcessor):

    def __init__(self, suppress_tokens, device: str = "cpu"):
        self.suppress_tokens = torch.tensor(list(suppress_tokens), device=device)

    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        suppress_token_mask = torch.isin(vocab_tensor, self.suppress_tokens)
        scores = torch.where(suppress_token_mask, -float("inf"), scores)
        return scores

# class SuppressTokensLogitsProcessor:
#     def __init__(self, suppress_tokens, device="cuda"):
#         self.suppress_tokens = torch.tensor(list(suppress_tokens), device=device, dtype=torch.long)

#     def __call__(self, scores: torch.Tensor) -> torch.Tensor:
#         # scores: [B, vocab]
#         scores[:, self.suppress_tokens] = -float("inf")
#         return scores
        
class LLaMA_adapter(nn.Module):

    def __init__(self, llama_ckpt_dir,
                 max_seq_len=2048, max_batch_size=1,
                 clip_model='ViT-L/14',
                 v_embed_dim=768, v_depth=8,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 #query_len=128, 
                 query_layer=20,
                 img_adapter_len = 64,
                 text_adapter_len = 64,
                 use_text = True,
                 text_in_layers = 8,
                 w_bias=False, 
                 w_lora=False, lora_rank=16, 
                 w_new_gate=False,
                 phase="finetune"):
        super().__init__()
        print(f"llama_ckpt_dir={llama_ckpt_dir}")
        # load llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        w_bias = phase == "finetune"
        print("params\n", params)
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        ) # max_batch_size only affects inferenc
        #self.query_len = query_len
        self.query_len = img_adapter_len + text_adapter_len  #+ text_adapter_len
        self.query_layer = query_layer
        self.img_adapter_len = img_adapter_len ### Fix this
        self.text_adapter_len = text_adapter_len
        self.use_text = use_text
        
        # 2. Text query Original Params
        # self.text_dim = 128               # Embedding dimension
        # self.text_heads = 4              # Number of attention heads
        # self.text_len = 32               # Text seq_len
        # self.text_depth = 1              # Number of transformer blocks
        # self.text_mlp_ratio = 2.0        # Expansion ratio for the MLP
        # self.text_dropout = 0.1          
        # self.text_vocab = 65536
        # self.text_pad_idx = 16383
        # self.scale_factor = 1 / 1# query_layer
        # self.text_encoder = TextEncoder(
        #                              vocab_size = self.text_vocab , 
        #                              embedding_dim = self.text_dim, # Text output dim 
        #                              model_dim = model_args.dim, # Not used; need to get rid of!!!!!!
        #                              seq_len = self.text_len, # text seq_len
        #                              num_layers = self.text_depth,
        #                              num_heads = self.text_heads, 
        #                              mlp_ratio = self.text_mlp_ratio, # mlp compression ratio
        #                              dropout =self.text_dropout , 
        #                              adapter_seq_len = text_adapter_len, # <- adapter len; fix above!
        #                              pad_idx = self.text_pad_idx)
        
        # Concise Text Encoder
        self.text_dim = 64               # Embedding dimension
        self.text_heads = 2              # Number of attention heads
        self.text_len = 32               # Text seq_len
        self.text_depth = 1              # Number of transformer blocks
        self.text_mlp_ratio = 2.0        # Expansion ratio for the MLP
        self.text_dropout = 0.1          
        self.text_vocab = 65536
        self.text_pad_idx = 16383
        self.scale_factor = 1 / 1# query_layer
        self.text_encoder = TextEncoder(
                                     vocab_size = self.text_vocab , 
                                     embedding_dim = self.text_dim, # Text output dim 
                                     model_dim = model_args.dim, # Not used; need to get rid of!!!!!!
                                     seq_len = self.text_len, # text seq_len
                                     num_layers = self.text_depth,
                                     num_heads = self.text_heads, 
                                     mlp_ratio = self.text_mlp_ratio, # mlp compression ratio
                                     dropout =self.text_dropout , 
                                     adapter_seq_len = text_adapter_len, # <- adapter len; fix above!
                                     pad_idx = self.text_pad_idx)
        
        # text_in layers for concat w/ adapter
        self.text_in_layers = text_in_layers

        self.text_proj = nn.Linear(self.text_dim, model_args.dim)  # Map llama hidden dim
        self.text_proj_norm = nn.LayerNorm(model_args.dim,elementwise_affine=True)  # Normalize the final output
        
        # 3. adapter query
        self.adapter_query = nn.Embedding(
            img_adapter_len * query_layer, model_args.dim)

        # 4. tokenizer
        #self.tokenizer = Tokenizer(model_path=llama_tokenizer)
        #self.pad_id = 0
        self.pad_id = 8192
        self.ignore_idx = -100
        #self.eos_id = 8192
        
        # 5. llama
        model_args.w_bias = w_bias
        model_args.w_lora = w_lora
        model_args.lora_rank = lora_rank
        model_args.w_new_gate = w_new_gate
        model_args.vocab_size = 8292 # Set to LVM vocab_size
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        #print(model_args)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        #print(self.llama)
        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        print("checkpoints:\n",ckpts)
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            load_result = self.llama.load_state_dict(ckpt, strict=False)
            print("Missing keys:", load_result.missing_keys)
            print("Unexpected keys:", load_result.unexpected_keys)
            
        self.suppress_tokens=list(range(8192, 8292))
        self.suppress_tokens_processor = SuppressTokensLogitsProcessor(self.suppress_tokens, device="cuda")
        
        # Fix freq embedding and mask
        # self.freqs_cis = self.llama.freqs_cis
        # self.freqs_cis = self.freqs_cis[:max_seq_len]
        # self.mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
        # self.mask = torch.triu(self.mask, diagonal=0 + 1)
        
        # self.register_buffer('freqs_cis', self.llama.freqs_cis[:max_seq_len])

        # # Register mask as a buffer
        # self.register_buffer('mask', torch.full(
        #     (1, 1, max_seq_len, max_seq_len), float("-inf") ) )
        # self.mask = torch.triu(self.mask, diagonal=1)
        
        # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_idx)#ignore_index=padding)

        # 7. training parameters
        self.phase = phase
        self.get_trainable_params(self.phase)

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #       print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name:
                        para.data = para.data.float()
                        para.requires_grad = True

            # for layer in self.llama.layers[-self.query_layer:]:
            #     for name, module in layer.named_modules():
            #         # LLaMA uses RMSNorm (not nn.LayerNorm)
            #         if 'norm' in name or 'bias' in name:
            #             for p in module.parameters():
            #                 p.data = p.data.float()
            #                 p.requires_grad = True
                    
        elif phase == 'pretrain':
            train_param_name = ['gate', 'clip_proj', 'text_encoder', 'text_proj', 'text_proj_norm','clip_proj_norm', 'visual_query', 'visual_blocks', 'visual_proj','visual_proj_norm', 'adapter_query', 'qk_gate']
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        # if train_name == 'qk_gate':
                        #     print("qk_gate")
                        para.data = para.data.float()
                        para.requires_grad = True
        
        else:
            raise ValueError(f"Unknown model phase: {phase}")
            
    def forward_text(self, text_tokens):

        _bsz, len_ = text_tokens.shape
        # padding_mask = (text_tokens == self.text_pad_idx)
         
        # query_mask = torch.zeros((_bsz,len_), dtype=padding_mask.dtype, device=text_tokens.device)
        # padding_mask = torch.cat([query_mask, padding_mask], dim=1)
        #print(padding_mask)
        #print(f"{padding_mask.shape}, {query_mask.shape}, {padding_mask.shape}")
        # Apply projection
        text_query = self.text_encoder(text_tokens, mask = None) # trying w/o padding mask Feb23
        
        if not torch.isfinite(text_query).all():
            bad = (~torch.isfinite(text_query)).nonzero(as_tuple=False)
            print("[NaN FROM TEXT_ENCODER]")
            print("shape:", text_query.shape, "dtype:", text_query.dtype)
            print("first bad idx:", bad[0])
            print("value:", text_query.flatten()[bad[0,0]].item())
            raise RuntimeError("text_encoder produced NaN/Inf")
        #text_query = self.text_encoder(text_tokens, mask = padding_mask)
        #text_query = torch.clamp(text_query, min=-25, max=25)  # Clamp after projection
        #print(f"self.text_proj(text_query).shape={text_query.shape}")
        # Reshape and normalizexw
        text_query = self.text_proj(text_query)
        
       # text_query = self.text_embedding(text_tokens)
        
        # layernorm disabled for small bsz!!!!!
        #print("text_query b4 norm", text_query.min(), text_query.max())
        text_query = self.text_proj_norm(text_query)  # Stabilize normalization
        #text_query = torch.nn.functional.normalize(text_query, dim=-1)    
        #print("text_query after norm", text_query.min(), text_query.max())
        return text_query

    def inference_forward_text(self, text_tokens):
        _bsz, len_ = text_tokens.shape
        # padding_mask = (text_tokens == self.text_pad_idx)
        # query_mask = torch.zeros((_bsz,len_), dtype=padding_mask.dtype, device=text_tokens.device)
        # padding_mask = torch.cat([query_mask, padding_mask], dim=1)
        text_query = self.text_encoder(text_tokens, mask = None)
        text_query = self.text_proj(text_query)
        text_query = self.text_proj_norm(text_query) # Stabilize normalization
        return text_query


    def forward(self, tokens, labels, text_tokens):
        """text query"""
        #import pdb; pdb.set_trace()
        if self.use_text:
            text_query = self.forward_text(text_tokens)

        #print(f"text_query,min={text_query.min()}, text_query,max={text_query.max()}")
        
        _bsz, seqlen = tokens.shape
        
        # print(f"text_query:{text_query.shape} {text_query.dtype}")
        # print("seqlen:", seqlen)
        
        h = self.llama.tok_embeddings(tokens)
        
        # print("h.shape:", h.shape)
        
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = checkpoint(layer, h, 0, freqs_cis, mask, use_reentrant=False) 
            #h = layer(h, 0, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.img_adapter_len, -1).unsqueeze(1)
        adapter_index = 0
        dynamic_adapter = None
        # print("adapter.shape", adapter.shape, adapter.dtype )
        # print("*"*60)
        count = 0

        for layer in self.llama.layers[-1 * self.query_layer:]:  
            #print("count:",count)
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            #print("dynamic_adapter.shape 1", dynamic_adapter.shape, type(dynamic_adapter))
            #print("text_query.shape", text_query.min().item(), text_query.max().item(), text_query.shape)
            #print("dynamic_adapter.shape", dynamic_adapter.min().item(), dynamic_adapter.max().item(), dynamic_adapter.shape)
            
            # Only inject text into the last `self.text_in_layers` layers
            if self.use_text and adapter_index >= self.query_layer - self.text_in_layers:
                dynamic_adapter = torch.cat([dynamic_adapter, self.scale_factor * text_query], dim = 1)   
            #print("dynamic_adapter.shape", dynamic_adapter.shape, type(dynamic_adapter))
            #print(f"{dynamic_adapter} {count} grad norm: {dynamic_adapter.grad.norm().item()}")
            if adapter_index < 12: 
                h = checkpoint(layer, h, 0, freqs_cis, mask, dynamic_adapter, use_reentrant=False) 
            else:
                h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1
            
            #print("h.min/max", h.min(), h.max())
            
        count += 1  
        #print(f" dyn.adap.max={dynamic_adapter.max()}, text.max={text_query.max()}")
        
        h = h.clamp(min=-65000, max=65000)
        h = self.llama.norm(h)
        output = self.llama.output(h)
        #print("312", output.max(), output.min(), labels.sum()==0)
        
        output = self.suppress_tokens_processor(scores=output)
        
        #print("314", type(output))
        #print("315", output.max(), output.min(), labels.sum()==0)
        #print("hello world")
        
        output = output[:, :-1, :]
        labels = labels[:, 1:]
        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 8292
            #print("output.shape", output.shape)
            #print("torch.argmax(output)", torch.argmax(output,dim=-1))
            #print(f"Label range: min={labels.min()}, max={labels.max()}", f"torch.argmax(output)={torch.argmax(output,dim=-1).max()}")
            #print("self.llama.vocab_size",self.llama.vocab_size)
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())

        if not math.isfinite(c_loss):
            print("Loss is {}, stopping training".format(c_loss))
            print("text_tokens:", text_tokens.shape, text_tokens.min(), text_tokens.max())
            print("output.shape", output.shape)
            print("torch.argmax(output)", torch.argmax(output,dim=-1))
            print(f"Logits range: min={output.min().item()}, max={output.max().item()}")
            print(f"Label range: min={labels.min()}, max={labels.max()}", f"torch.argmax(output)={torch.argmax(output,dim=-1).max()}")
            print("self.llama.vocab_size",self.llama.vocab_size)

        return c_loss, c_loss
        
    """################################## UPDATE QUERY SHAPES #################################"""
    
    @torch.inference_mode()
    def forward_inference(self, tokens, text_token, start_pos: int):
        _bsz, seqlen = tokens.shape
        #print("f.infer tokens.shape", tokens.shape, seqlen)
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        #print("freqs_cis.shape", freqs_cis.shape)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        if len(freqs_cis) < 1:
          print("start_pos : start_pos + seqlen shape", freqs_cis[start_pos : start_pos + seqlen].shape, start_pos, start_pos + seqlen)
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer,self.img_adapter_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            # original inference dynamic adapter 1/5/26
            # dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            # dynamic_adapter = torch.cat([dynamic_adapter, self.scale_factor * text_token], dim = 1)
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)

            if self.use_text and adapter_index >= self.query_layer - self.text_in_layers:
                dynamic_adapter = torch.cat([dynamic_adapter, self.scale_factor * text_token], dim = 1)   
            
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = h.clamp(min=-65000, max=65000)
        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()
    #original 12/23/25
    # def forward_inference(self, visual_query, tokens, start_pos: int):
    #     _bsz, seqlen = tokens.shape
    #     h = self.llama.tok_embeddings(tokens)
    #     freqs_cis = self.llama.freqs_cis.to(h.device)
    #     freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
    #     mask = None
    #     mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
    #     mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

    #     for layer in self.llama.layers[:-1 * self.query_layer]:
    #         h = layer(h, start_pos, freqs_cis, mask)

    #     adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
    #     adapter_index = 0
    #     for layer in self.llama.layers[-1 * self.query_layer:]:
    #         dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
    #         dynamic_adapter = torch.nn.functional.normalize(dynamic_adapter, dim=-1)
    #         dynamic_adapter = dynamic_adapter + visual_query
    #         h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
    #         adapter_index = adapter_index + 1

    #     h = self.llama.norm(h)
    #     output = self.llama.output(h[:, -1, :])

    #     return output.float()

    @torch.inference_mode()
    def generate(
        self, img_tokens,
        text_tokens,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        #print("img_tokens.shape", img_tokens.shape)
        bsz = len(img_tokens)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(img_tokens) == len(text_tokens)
        #print("len(img_tokens), len(text_tokens)", len(img_tokens), len(text_tokens))
        if self.use_text:
            with torch.cuda.amp.autocast():
                text_query = self.inference_forward_text(text_tokens) # cluge, will fix later

        #print(f"len(img_tokens[0]): {len(img_tokens[0])}")
        min_prompt_size = min([len(t) for t in img_tokens])
        max_prompt_size = max([len(t) for t in img_tokens])
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)


        #print("max min prompt, total len", min_prompt_size, max_prompt_size, total_len)


        tokens = torch.full((bsz, total_len), self.pad_id).cuda().long()
        #print("shape of tok after pad", tokens.shape)
        for k, t in enumerate(img_tokens):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        #print("start_pos, prev_pos", start_pos, prev_pos)
        for cur_pos in range(start_pos, total_len):
            #with torch.cuda.amp.autocast():
            # if cur_pos in {start_pos, start_pos + 1, start_pos + 2}:
            #     print(
            #         "slice len:",
            #         tokens[:, prev_pos:cur_pos].shape[1],
            #         "prev_pos",
            #         prev_pos,
            #         "cur_pos",
            #         cur_pos
            #     )

            logits = self.forward_inference(tokens[:, prev_pos:cur_pos],
                                            text_query,
                                            prev_pos)
            logits = self.suppress_tokens_processor(scores=logits)
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            # if bsz == 1 and next_token[0] == self.eos_id:
            #     print(f"{self.eos_id} called at {cur_pos}")
            #     break
            
            prev_pos = cur_pos
        #print(tokens.shape, type(tokens))
        # decoded = []
        # for i, t in enumerate(tokens.tolist()):
        #     # CURRENTLY INCLUDE ALL CONTEXT:
        #     # cut to max gen len
        #     #t = t[len(img_tokens[i]): len(img_tokens[i]) + max_gen_len]
            
        #     # Append Whole Thing For Now
        #     #t = t[len(img_tokens[i]): len(img_tokens[i]) + max_gen_len]
            
        #     # cut to eos tok if any
        #     # try:
        #     #     t = t[: t.index(self.eos_id)]
        #     # except ValueError:
        #     #     pass
        #     # No decoding right now
        #     decoded.append(t)
        #     #decoded.append(self.tokenizer.decode(t))

        # return decoded
        return tokens
        
    # Original 12/23/25
    # def generate(
    #     self, imgs, prompts,
    #     max_gen_len: int = 256,
    #     temperature: float = 0.1,
    #     top_p: float = 0.75,
    # ):
    #     bsz = len(imgs)
    #     params = self.llama.params
    #     assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
    #     assert len(imgs) == len(prompts)

    #     with torch.cuda.amp.autocast():
    #         visual_query = self.forward_visual(imgs)

    #     if isinstance(prompts[0], str):
    #         prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    #     min_prompt_size = min([len(t) for t in prompts])
    #     max_prompt_size = max([len(t) for t in prompts])

    #     total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

    #     tokens = torch.full((bsz, total_len), self.pad_id).cuda().long()

    #     for k, t in enumerate(prompts):
    #         tokens[k, : len(t)] = torch.tensor(t).cuda().long()
    #     input_text_mask = tokens != self.pad_id
    #     start_pos = min_prompt_size
    #     prev_pos = 0
    #     for cur_pos in range(start_pos, total_len):
    #         with torch.cuda.amp.autocast():
    #             logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
    #         if temperature > 0:
    #             probs = torch.softmax(logits / temperature, dim=-1)
    #             next_token = sample_top_p(probs, top_p)
    #         else:
    #             next_token = torch.argmax(logits, dim=-1)
    #         next_token = next_token.reshape(-1)

    #         next_token = torch.where(
    #             input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
    #         )
    #         tokens[:, cur_pos] = next_token
    #         # trick: early stop if bsz==1
    #         if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
    #             break
    #         prev_pos = cur_pos

    #     decoded = []
    #     for i, t in enumerate(tokens.tolist()):

    #         # cut to max gen len
    #         t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
    #         # cut to eos tok if any
    #         try:
    #             t = t[: t.index(self.tokenizer.eos_id)]
    #         except ValueError:
    #             pass
    #         decoded.append(self.tokenizer.decode(t))

    #     return decoded



_MODELS = {
    "BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    "LORA-BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth",
    "CAPTION-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/5088aeb63a89746b90bcfd5cb819e1c7411b2771b267c6d131ce73e250a8abf0_CAPTION-7B.pth",
    "LORA-BIAS-7B-v21": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.1.0/d26d107eec32127ac86ef1997cf7169de1c56a59c539fc1258c6798b969e289c_LORA-BIAS-7B-v21.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}

def available_models():
    return list(_MODELS.keys())

def load(name, llama_dir, llama_type="7B", device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts', max_seq_len=512,
        phase="finetune"):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}"), None

    # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
    # llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})

    model = LLaMA_adapter(
        llama_ckpt_dir,
        max_seq_len=512, max_batch_size=1,
        clip_model='ViT-L/14',
        v_embed_dim=768, v_depth=8,
        v_num_heads=16, v_mlp_ratio=4.0,
        query_len=10, query_layer=31,
        w_bias=model_cfg.get('w_bias', False), 
        w_lora=model_cfg.get('w_lora', False), 
        lora_rank=model_cfg.get('lora_rank', 16),
        w_new_gate=model_cfg.get('w_lora', False), # for compatibility
        phase=phase)

    load_result = model.load_state_dict(ckpt['model'], strict=False)

    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device), model.clip_transform

################################################################################################################################
################################.     Light Weight Attn Encoder,  mask          ################################################
################################################################################################################################

class LightweightBlock(nn.Module):
    def __init__(self, dim=64, num_heads=2, mlp_ratio=2., qkv_bias=False, drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU, drop=drop)

    def forward(self, x, mask = None):
        x = x + self.attn(self.norm1(x), mask = mask)
        x = x + self.mlp(self.norm2(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=qkv_bias, dropout=attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask = None):
        # nn.MultiheadAttention expects input of shape (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask )
        attn_output = attn_output.permute(1, 0, 2)
        return self.proj_drop(attn_output)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, model_dim, seq_len, num_layers, num_heads, mlp_ratio, dropout, adapter_seq_len, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.positional_embedding = nn.Embedding(seq_len, embedding_dim)
        self.encoder_blocks = nn.ModuleList([
            LightweightBlock(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

        # Additional components for adapter_seq_len and output_dim
        self.query_len = adapter_seq_len
        self.learned_query = nn.Parameter(torch.randn(adapter_seq_len, embedding_dim))  # Learned queries

    def forward(self, tokens, mask = None):
        batch_size, seq_len = tokens.shape

        # Token and positional embeddings
        positions = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.positional_embedding(positions)

      # Append learned queries
        queries = self.learned_query.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, adapter_seq_len, embedding_dim)
        #print(queries.shape, x.shape)

        x = torch.cat([queries, x], dim=1)  # Combine queries and encoded tokens
        #print(x.shape)
        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x, mask = mask)

        # Normalize the encoder output
        x = self.norm(x)

        # Slice the first `adapter_seq_len` positions
        x = x[:, :self.query_len, :]  # (batch_size, adapter_seq_len, 4098)


        return x

################################################################################################################################
################################.     Light Weight Attn Encoder, no mask        ################################################
################################################################################################################################

# class LightweightBlock(nn.Module):
#     def __init__(self, dim=64, num_heads=2, mlp_ratio=2., qkv_bias=False, drop=0., norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=drop, proj_drop=drop)
        
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU, drop=drop)

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=qkv_bias, dropout=attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         # nn.MultiheadAttention expects input of shape (seq_len, batch_size, embed_dim)
#         x = x.permute(1, 0, 2)  # Convert to (seq_len, batch_size, dim)
#         #with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
#         attn_output, _ = self.attn(x, x, x)  # Self-attention
#         attn_output = attn_output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, dim)
#         return self.proj_drop(attn_output)

# class TextEncoder(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, model_dim, seq_len, num_layers, num_heads, mlp_ratio, dropout, adapter_seq_len, pad_idx):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
#         self.positional_embedding = nn.Embedding(seq_len, embedding_dim)
#         self.encoder_blocks = nn.ModuleList([
#             LightweightBlock(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=dropout)
#             for _ in range(num_layers)
#         ])
#         self.norm = nn.LayerNorm(embedding_dim)

#         # Additional components for adapter_seq_len and output_dim
#         self.query_len = adapter_seq_len
#         self.learned_query = nn.Parameter(torch.randn(adapter_seq_len, embedding_dim))  # Learned queries

#     def forward(self, tokens):
#         batch_size, seq_len = tokens.shape

#         # Token and positional embeddings
#         positions = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0)
#         x = self.embedding(tokens) + self.positional_embedding(positions)

#       # Append learned queries
#         queries = self.learned_query.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, adapter_seq_len, embedding_dim)


#         x = torch.cat([queries, x], dim=1)  # Combine queries and encoded tokens
#         # Encoder blocks
#         for block in self.encoder_blocks:
#             x = block(x)

#         # Normalize the encoder output
#         x = self.norm(x)

#         # Slice the first `adapter_seq_len` positions
#         x = x[:, :self.query_len, :]  # (batch_size, adapter_seq_len, 4098)

#         # Projection to desired output shape
#         #x = self.projection(x)  # (batch_size, seq_len + adapter_seq_len, 4098)
#         #x = self.projection_norm(x)
        
#         return x 
     
# import torch.nn.functional as F

# class MultilayerProjection(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim):
#         """
#         Args:
#             input_dim (int): Dimensionality of input features.
#             hidden_dims (list of int): Dimensions of intermediate hidden layers.
#             output_dim (int): Dimensionality of the output features.
#         """
#         super(MultilayerProjection, self).__init__()
        
#         # Define layers
#         layers = []
#         in_dim = input_dim
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(in_dim, hidden_dim))
#             layers.append(nn.LeakyReLU())  # Non-linearity
#             in_dim = hidden_dim
        
#         # Add final projection layer
#         layers.append(nn.Linear(in_dim, output_dim))
        
#         # Register layers in a sequential container
#         self.projection = nn.Sequential(*layers)
#     """
#     def forward(self, x):
#         return self.projection(x)
#     """
#     def forward(self, x):
#         for i, layer in enumerate(self.projection):
#             x = layer(x)
#             if isinstance(layer, nn.Linear) or isinstance(layer, nn.LeakyReLU):
#                 print(f"Layer {i}: min={x.min().item()}, max={x.max().item()}")
#         return x
def stepn(n):
    import pdb
    for _ in range(int(n)):
        pdb.Pdb().onecmd('n')