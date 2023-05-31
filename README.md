# cdcd-pytorch
A partial implementation of Continuous Diffusion for Categorical Data by Deepmind, in pytorch.

# Usage
    
```python
from cdcd_pytorch import CDCD, DDPMScheduler

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.01,
    beta_schedule="linear",
    clip_sample=False,
    prediction_type="epsilon",
)

model = CDCD(
    scheduler=scheduler,
    hidden_size=768,
    num_heads=8,
    depth=12,
    score_hidden_size=256, # size of embeddings
    embedding_max_length=1,
    embedding_features=512,        
)

# get your tokens and some conditioning signal
x = torch.randint(0, 50000, (8, 1000))
embedding = torch.randn(8, 1, 512)

# do this many times
loss = model(x, embedding=embedding)


# once you're done training, you can sample from the model with classifier-free-guidance
noise = torch.randn(1, 1024, 256)
token_pred = model.sample(noise, num_steps=50, embedding=embedding, embedding_scale=2.5)
```


# TODO

- [ ] Add self conditioning
- [ ] Add input masking
- [ ] Experiment with a two stage training, first stage to train the embeddings with cross entropy and a second stage with frozen embeddings and a score matching loss