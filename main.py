from tensordict import TensorDict
import torch
import torchrl

from tqdm.auto import tqdm

from src.test_config import create_batched_test_one_stage_supplier_network
from src.batched_base_config import create_batched_base_supplier_network
from src.batched_env.env_wrapper import MaxDStrategy, DefaultStrategy, NearestSufficientChangeDStrategy
from src.model_zoo.my_ppo.ppo import ppo
from src.model_zoo.my_ppo.actor_critic_models.mlp_actor_critic import MLPActorCriticFabric

import logging

logging.getLogger().setLevel(logging.DEBUG)
logging.debug("Import done!")

ac = ppo(
    lambda : create_batched_base_supplier_network(500, MaxDStrategy(), 'cuda'),
    # create_batched_test_one_stage_supplier_network,
    actor_critic=MLPActorCriticFabric(),
    steps_per_epoch=16000,
    gamma=1.0,
    clip_ratio=0.1,
    target_kl=0.01,
    pi_lr=1e-5,
    vf_lr=5e-3,
    train_pi_iters=20,
    train_v_iters=10,
    epochs=500,
    lam=0.1,
    device='cuda',
)
