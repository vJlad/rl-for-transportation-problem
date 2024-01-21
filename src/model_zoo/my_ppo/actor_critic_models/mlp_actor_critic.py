from torch.nn.modules.module import T

from src.model_zoo.my_ppo.actor_critic_models.base import ActorCriticFabric, ActorCritic, Actor
from src.batched_env.env_wrapper import BatchedEnvWrapper

import torch
from torch import nn, Tensor
from tensordict import TensorDict


class MlpModel(nn.Module):
    def __init__(self, input_example: TensorDict, output_size: int):
        super().__init__()
        hid_by_layer = 64
        out_by_layer = 128
        self.by_layer_models = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(-3, -1),
                nn.Linear(input_example['state'][str(i)][0].numel(), hid_by_layer),
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
            model(observation['state'][str(i)].float())
            for i, model in enumerate(self.by_layer_models)
        ]
        joint_in = torch.cat(by_layer_out, dim=-1)
        return self.joint_body(joint_in)


class MLPActorCriticFabric(ActorCriticFabric):
    @staticmethod
    def _create_critic(env: BatchedEnvWrapper) -> nn.Module:
        return MlpModel(env.get_obs(), 1)

    @staticmethod
    def _create_actor(env: BatchedEnvWrapper) -> nn.Module:
        return MlpModel(env.get_obs(), env.supplier_network.number_of_actions())

    def __call__(self, env: BatchedEnvWrapper) -> ActorCritic:
        critic = self._create_critic(env)
        actor = Actor(self._create_actor(env))
        return ActorCritic(actor, critic)
