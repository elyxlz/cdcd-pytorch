from typing import Callable, Optional, Tuple
from torchtyping import TensorType

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .modules import RotaryEmbedder, TimestepEmbedder, FixedEmbedding
from .utils import rand_bool


class CDCD(nn.Module):
    def __init__(
        self,
        net_t: Callable,
        scheduler: nn.Module,
        hidden_size: int,
        score_hidden_size: int,
        embedding_max_length: int,
        embedding_features: int,
        **kwargs,
    ):
        super().__init__()
        
        self.scheduler = scheduler
        msg = 'prediction type must be epsilon'
        assert self.scheduler.config.prediction_type == "epsilon", msg
        
        # transformer / conformer backbone
        self.net = net_t(
            hidden_size=hidden_size,
            use_time_conditioning=False,
            use_embedding_cfg=False,
            cat_embeddings=True,
            embedding_features=embedding_features,
            **kwargs,
        )
        
        # fixed embedding for cfg
        self.fixed_embedding = FixedEmbedding(
            max_length=embedding_max_length,
            features=embedding_features,
        )
        
        # timestep embedding
        self.timestep_embedder = TimestepEmbedder(
            hidden_size=hidden_size,
            frequency_embedding_size=256,
        )

        # embeddings for every token
        self.token_embedder = nn.Embedding(1024, score_hidden_size)
        self.score_to_input = nn.Linear(score_hidden_size, hidden_size, bias=False)

        # rotary embedding with dim of attention heads
        self.rotary_embedder = RotaryEmbedder(64) 
        
        # logit simplex projection
        self.linear = nn.Linear(hidden_size, 1024, bias=True)
        
        self.initialize_parameters()
        
        
    def initialize_parameters(
        self,
    ):
        # initialize token_embedder
        nn.init.xavier_uniform_(self.token_embedder.weight)
        
        # initialize linear
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
    
        # initialize score_to_input
        nn.init.xavier_uniform_(self.score_to_input.weight)
        
        # initialize timestep embedder
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        

    def forward(
        self,
        tokens: TensorType["batch", "num_tokens"],
        embedding: TensorType["batch", "channels", "hidden_size"],
        embedding_mask_proba: float = 0.0,
        timesteps: Optional[TensorType["batch"]] = None,
        return_loss_per_timestep: bool = False,
        **kwargs,
    ) -> Tuple[TensorType["batch", "seq_len", 1024], TensorType["batch", "seq_len", "embedding"]]:
        b, device = tokens.shape[0], tokens.device
        
        # prepare scheduler
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(num_train_timesteps, device=device)
        
        # sample random timestep
        if timesteps is None:
            timesteps = torch.randint(0, num_train_timesteps, (b,), device=device).long()
        
        # embed tokens
        x0 = self.token_embedder(tokens)
        
        # normalize with L2 to avoid exploding params
        x0_norm = F.normalize(x0, p=2, dim=-1)
        
        # add noise to normalized embedding
        noise = torch.randn_like(x0_norm)
        x = self.scheduler.add_noise(x0_norm, noise, timesteps)
        x_expanded = self.score_to_input(x)
        
        # add rotary embeddings
        rotary_emb = self.rotary_embedder(x.shape[-2])
        
        # add time embeddings to be used in AdaLN
        features = self.timestep_embedder(timesteps)

        # randomly mask embedding for cfg
        if embedding_mask_proba > 0.0:
            embedding_mask = self.fixed_embedding(embedding)
            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba, device=device
            )
            embedding = torch.where(batch_mask, embedding_mask, embedding)
        
        # forward pass
        out_embeddings = self.net(
            x=x_expanded,
            embedding=embedding,
            features=features,
            rotary_emb=rotary_emb,
            **kwargs
        )
        
        # get logits
        logits = self.linear(out_embeddings)
        
        loss_full = F.cross_entropy(logits.view(-1, 1024), tokens.view(-1), reduction="none")
        loss = loss_full.mean()
        
        if return_loss_per_timestep:
            # create dict with averaged loss per timestep
            losses = loss_full.view(b, -1).mean(dim=-1)
            unique_timesteps, indices = torch.unique(timesteps, return_inverse=True)
            avg_losses = torch.zeros_like(unique_timesteps, dtype=losses.dtype)
            avg_losses.index_add_(0, indices, losses)
            counts = torch.bincount(indices)
            avg_losses /= counts
            
            loss_dict = {unique_timesteps[i].item(): avg_losses[i].item() for i in range(len(unique_timesteps))}
            return loss, loss_dict
        
        return loss
        
    @torch.no_grad()
    def sample(
        self,
        num_steps: int,
        x_noisy: TensorType["batch", "seq_len", "embedding"],       # already normalized
        embedding: TensorType["batch", "channels", "hidden_size"],
        embedding_scale: float = 1.0,
        show_progress: bool = False,
    ):
        b, device = x_noisy.shape[0], x_noisy.device
                
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        progress_bar = tqdm(timesteps, disable=not show_progress)
        for t in progress_bar:
            t_batched = torch.full(size=(b,), fill_value=t, device=device)
            
            # rotary embeddings
            rotary_emb = self.rotary_embedder(x_noisy.shape[-2])
            
            # time embeddings
            features = self.timestep_embedder(t_batched)
            
            # fixed embedding for cfg or uncond
            embedding_mask = self.fixed_embedding(embedding)
            if embedding_scale != 1.0:
                # batch cond and uncond to avoid two forward passes
                x_noisy_combined = torch.cat((x_noisy, x_noisy), dim=0)
                embedding_combined = torch.cat((embedding, embedding_mask), dim=0)
                features_combined = torch.cat((features, features), dim=0)
                
                x_noisy_combined_expanded = self.score_to_input(x_noisy_combined)
                out = self.net(
                    x_noisy_combined_expanded,
                    embedding=embedding_combined,
                    features=features_combined,
                    rotary_emb=rotary_emb,
                )

                logits_combined = self.linear(out)
                simplex_combined = F.softmax(logits_combined, dim=-1)
                simplex = simplex_combined[:b]
                
                x0_pred_combined = simplex_combined @ F.normalize(self.token_embedder.weight, p=2, dim=-1)
                                
                score_combined = (x0_pred_combined - x_noisy_combined)# / (t ** 2)
                score_cond, score_uncond = torch.split(score_combined, b, dim=0)
                
                # do cfg and update x_noisy
                score = score_uncond + (score_cond - score_uncond) * embedding_scale
                x0_pred = x_noisy + score
                
            else:
                if embedding_scale == 0.0:
                    embedding = embedding_mask
                    
                x_noisy_expanded = self.score_to_input(x_noisy)
                out = self.net(
                    x_noisy_expanded,
                    embedding=embedding,
                    features=features,
                    rotary_emb=rotary_emb,
                )
                
                logits = self.linear(out)
                simplex = F.softmax(logits, dim=-1)
                
                x0_pred = simplex @ F.normalize(self.token_embedder.weight, p=2, dim=-1)
                score = (x0_pred - x_noisy)# * (self.scheduler.betas[t-1] / self.scheduler.betas.sum())# / (t)# ** 2)
                
            #x_noisy = self.scheduler.step(score, t, x_noisy).prev_sample
            x_noisy = self.scheduler.add_noise(x0_pred, score, t)
            
            progress_bar.set_description(f"Sampling (noise={t})")
            
        # greedy decode
        tokens = torch.argmax(simplex, dim=-1)
        return tokens