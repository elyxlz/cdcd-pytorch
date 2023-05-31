from typing import Union, Optional, Tuple, List, Callable, Dict
from torchtyping import TensorType

import torch
from torch import nn, Tensor
import math
from einops import rearrange, repeat
from functools import partial

from .attend import Attend
from .utils import default, exists, to_2tuple


""" Components """


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class FixedEmbedding(nn.Module):
    def __init__(
        self,
        max_length: int,
        features: int,
    ):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, length, device = *x.shape[0:2], x.device
        assert_message = "Input sequence length must be <= max_length"
        assert length <= self.max_length, assert_message
        position = torch.arange(length, device=device)
        fixed_embedding = self.embedding(position)
        fixed_embedding = repeat(fixed_embedding, "n d -> b n d", b=batch_size)
        return fixed_embedding



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: TensorType["n"],         # a 1-D Tensor of N indices, one per batch element.
        dim: int,                   # the dimension of the output.
        max_period: int = 10000,    # controls the minimum frequency of the embeddings.
    ) -> TensorType["n", "d"]:
        """
        Create sinusoidal timestep embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(
        self,
        t: TensorType["n"],
    ) -> TensorType["n", "d"]:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class RotaryEmbedder(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: bool = None,
            out_features: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            norm_layer: Optional[nn.Module] = None,
            bias: bool = True,
            drop: float = 0.,
            use_conv: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(
        self,
        x: TensorType["b", "c", "h", "w"]
    ) -> TensorType["b", "c", "h", "w"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads = 8,
        dim_head = 64,
        qkv_bias = False,
        dropout = 0.,
        flash = True
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.attend = Attend(
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        rotary_emb = None
    ):
        n, device, h, has_context = x.shape[-2], x.device, self.num_heads, exists(context)
        context = default(context, x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



""" Apex """

class TransformerBlock(nn.Module):
    """
    A Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning and cross attention.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.context_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, context, rotary_emb=None):
        modulation_parts = self.adaLN_modulation(c).chunk(9, dim=1)
        shift_msa, scale_msa, gate_msa = modulation_parts[0:3]
        shift_xa, scale_xa, gate_xa = modulation_parts[3:6]
        shift_mlp, scale_mlp, gate_mlp = modulation_parts[6:9]
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rotary_emb=rotary_emb)
        x = x + gate_xa.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_xa, scale_xa), context=self.context_norm(context))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x
    

class FinalLayer(nn.Module):
    """
    The final layer of the Transformer.
    """
    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: TensorType["batch", "tokens", "hidden_size"],
        c: TensorType["batch", "tokens", "hidden_size"],
    ) -> TensorType["batch", "tokens", "hidden_size"]:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x
    


class Transformer(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        embedding_features: Optional[int] = None,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size)
        
        if exists(embedding_features) and embedding_features != hidden_size:
            self.embedding_projection = nn.Linear(embedding_features, hidden_size)
        else:
            self.embedding_projection = nn.Identity()
            
        self.initialize_weights()
    
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        
        
    def forward(
        self,
        x: TensorType["batch", "tokens", "hidden_size"],                                    # token embeddings
        features: TensorType["batch", "hidden_size"],                                       # conditioning features
        embedding: Optional[TensorType["batch", "channels", "embedding_features"]] = None,  # conditioning embeddings
        rotary_emb: Optional[TensorType["tokens", "head_dim"]] = None,                      # rotary embeddings
        **kwargs
    ) -> TensorType["batch", "tokens", "hidden_size"]:                                      # output embeddings
        
        num_tokens, device = x.shape[1], x.device

        if embedding is not None:
            embedding = self.embedding_projection(embedding)
            
        for block in self.blocks:
            x = block(x, c=features, context=embedding, rotary_emb=rotary_emb, **kwargs)
            
        out = self.final_layer(x[:, :num_tokens], features)
        return out