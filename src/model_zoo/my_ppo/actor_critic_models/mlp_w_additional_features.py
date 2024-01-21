from typing import Sequence

from src.model_zoo.my_ppo.actor_critic_models.base import ActorCriticFabric, ActorCritic, Actor
from src.batched_env.env_wrapper import BatchedEnvWrapper

import torch
from torch import nn, Tensor
from tensordict import TensorDict


class FeatureExtractor(nn.Module):
    def __init__(self, cap):
        super(FeatureExtractor, self).__init__()
        self.cap = cap

    def forward(self, x):
        x_quot = x // self.cap
        x_rem = x % self.cap
        return torch.cat([x, x_quot, x_rem], dim=-1).float()


class MlpWithFeaturesModel(nn.Module):
    def __init__(self, input_example: TensorDict, output_size: int, caps: Sequence[int]):
        super().__init__()
        hid_by_layer = 256
        out_by_layer = 256
        self.by_layer_models = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(-3, -1),
                FeatureExtractor(caps[i]),
                nn.Linear(input_example['state'][str(i)][0].numel() * 3, hid_by_layer),
                nn.LeakyReLU(),
                nn.Linear(hid_by_layer, out_by_layer),
                nn.LeakyReLU(),
            )
            for i in range(len(input_example['state'].keys()))
        ])
        hid_joint = 512
        self.joint_body = nn.Sequential(
            nn.Linear(out_by_layer * len(input_example['state'].keys()), hid_joint),
            nn.LeakyReLU(),
            nn.Linear(hid_joint, output_size)
        )

    def forward(self, observation: TensorDict) -> Tensor:
        by_layer_out = [
            model(observation['state'][str(i)])
            for i, model in enumerate(self.by_layer_models)
        ]
        joint_in = torch.cat(by_layer_out, dim=-1)
        return self.joint_body(joint_in)


class MLPwFActorCriticFabric(ActorCriticFabric):
    @staticmethod
    def _create_critic(env: BatchedEnvWrapper) -> nn.Module:
        return MlpWithFeaturesModel(
            env.get_obs(),
            1,
            [layer.cap for layer in env.supplier_network.base_layers]
        )

    @staticmethod
    def _create_actor(env: BatchedEnvWrapper) -> nn.Module:
        return MlpWithFeaturesModel(
            env.get_obs(),
            env.supplier_network.number_of_actions(),
            [layer.cap for layer in env.supplier_network.base_layers]
        )

    def __call__(self, env: BatchedEnvWrapper) -> ActorCritic:
        critic = self._create_critic(env)
        actor = Actor(self._create_actor(env))
        return ActorCritic(actor, critic)
