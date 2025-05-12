import math
from typing import List

import torch
from torch.nn.functional import log_softmax
from torch.distributions.categorical import Categorical
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def build_relaxed_single_token_prior(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
):
    """
    A single-token prior based of a simple mixture of Gaussians.
    """
    with torch.no_grad():
        bos = torch.tensor([[tokenizer.eos_token_id]], device=device)
        logits = model(bos).logits[:, -1]                               # (1,V)
        prior_probs = logits.softmax(-1).squeeze(0)                     # (V,)
    E: torch.Tensor = model.transformer.wte.weight.detach()             # (V,d)
    V, d = E.shape

    def logp(z: torch.Tensor, σ: float) -> torch.Tensor:
        const = -0.5 * d * math.log(2 * math.pi * σ * σ)
        inv_var = 1.0 / (σ * σ)
        
        diff = z.unsqueeze(1) - E.unsqueeze(0)                          # (N,V,d)
        mahal = diff.square().sum(-1)                                   # (N,V)
        log_gauss = const - 0.5 * inv_var * mahal                       # (N,V)
        log_weighted = log_gauss + prior_probs.log().unsqueeze(0)
        return torch.logsumexp(log_weighted, dim=-1)                    # (N,)

    def grad(z: torch.Tensor, σ: float) -> torch.Tensor:
        return torch.func.grad(lambda x: logp(x, σ).sum())(z)

    def sample(num: int, σ: float) -> torch.Tensor:
        cat = Categorical(prior_probs)
        tokens = cat.sample((num,))                                     # (num,)
        base = E[tokens]                                                # (num,d)
        noise = torch.randn_like(base, dtype=torch.float32) * σ
        return base + noise

    return logp, grad, sample

def build_suffix_likelihood(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    suffix_ids: List[int],
    device: torch.device,
):
    """
    Calculates the log-likelihood of a suffix given an embedding vector.
    """

    # suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    ids_tensor = torch.tensor(suffix_ids, device=device)
    bos_tensor = torch.tensor([tokenizer.bos_token_id], device=device)
    with torch.no_grad():
        suffix_embeds = model.transformer.wte(ids_tensor)  # (L,d)
        bos_embeds = model.transformer.wte(bos_tensor)    # (1,d)
    L = len(suffix_ids)

    def logp(z: torch.Tensor) -> torch.Tensor:
        N, d = z.shape
        z_seq = z.unsqueeze(1)                            # (N,1,d)
        suffix_expand = suffix_embeds.unsqueeze(0).expand(N, -1, -1)  # (N,L,d)
        bos_expand = bos_embeds.unsqueeze(0).expand(N, -1, -1)        # (N,1,d)
        inputs = torch.cat([bos_expand, z_seq, suffix_expand], dim=1)  # (N,2+L,d)
        logits = model(inputs_embeds=inputs).logits                   # (N,2+L,V)

        log_p = torch.zeros(N, device=device)
        for pos, tok_id in enumerate(suffix_ids):
            step_logits = logits[:, pos+1, :]
            log_p += log_softmax(step_logits, dim=-1)[:, tok_id]
        return log_p                                      # (N,)

    def grad(z: torch.Tensor) -> torch.Tensor:
        return torch.func.grad(lambda x: logp(x).sum())(z)

    return logp, grad