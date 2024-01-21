import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from tensordict import TensorDict
import torchrl

from abc import ABC
from typing import Optional, Tuple

from src.batched_env.env_wrapper import BatchedEnvWrapper


class Actor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, obs_batch: TensorDict, actions: Optional[Tensor] = None) -> Tuple[Distribution, Optional[Tensor]]:
        pi_logits = self.model(obs_batch)
        mask = nn.Flatten(-5, -1)(obs_batch['valid_actions_mask'])
        dist = torchrl.modules.MaskedCategorical(logits=pi_logits, mask=mask)
        logp_a = dist.log_prob(actions) if actions is not None else None
        return dist, logp_a


class ActorCritic(nn.Module):
    def __init__(self, actor: Actor, critic: nn.Module):
        super().__init__()
        self.pi = actor
        self.v = critic

    @torch.no_grad()
    def step(self, obs_batch: TensorDict):
        dist, _ = self.pi(obs_batch)
        v = torch.squeeze(self.v(obs_batch), dim=-1)
        a = dist.sample()
        log_probs = dist.log_prob(a)
        return a, v, log_probs

    @torch.no_grad()
    def act(self, obs_batch: TensorDict):
        dist, _ = self.pi(obs_batch)
        a = dist.sample()
        return a


class ActorCriticFabric(ABC):
    def __call__(self, env: BatchedEnvWrapper) -> ActorCritic:
        pass
