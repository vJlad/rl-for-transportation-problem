from dataclasses import dataclass, field
from typing import Union, Optional, Sequence

from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
import torch
from torch_discounted_cumsum import discounted_cumsum_right

from src.model_zoo.my_ppo.actor_critic_models.base import ActorCritic
from src.batched_env.env_wrapper import BatchedEnvWrapper
from src.model_zoo.my_ppo.metrics import EpochMetricsAggregator


@dataclass
class ReplayBuffer:
    batched_env: BatchedEnvWrapper
    size: int
    gamma: float
    lam: float
    epoch_aggregator: EpochMetricsAggregator
    max_episode_len: int = 1000
    device: Union[torch.device, str] = 'cpu'
    ptr: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self.obs_buf: TensorDictBase = TensorDict({}, batch_size=[self.size], device=self.device)
        for i in tqdm(range(0, self.size, self.batched_env.batch_size)):
            self.obs_buf[i: i + self.batched_env.batch_size] = self.batched_env.reset()
        self.act_buf: Tensor = torch.empty(self.size, dtype=torch.long, device=self.device)
        self.rew_buf: Tensor = torch.empty(self.size, dtype=torch.float32, device=self.device)
        self.ret_buf: Tensor = torch.empty(self.size, dtype=torch.float32, device=self.device)
        self.val_buf: Tensor = torch.empty(self.size, dtype=torch.float32, device=self.device)
        self.adv_buf: Tensor = torch.empty(self.size, dtype=torch.float32, device=self.device)
        self.logp_buf: Tensor = torch.empty(self.size, dtype=torch.float32, device=self.device)

    def get(self):
        assert self.ptr == self.size
        # the next three lines implement the advantage normalization trick
        adv_mean = torch.mean(self.adv_buf)
        adv_std = torch.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

    def allocate_ids(self, batch_size: int) -> Tensor:
        start_id = self.ptr
        end_id = self.ptr + batch_size
        assert end_id <= self.size
        self.ptr = end_id
        return torch.arange(start_id, end_id, device=self.device, dtype=torch.long)

    @staticmethod
    def _len_is_the_same(*tensors: Sequence[Tensor]) -> bool:
        return len(set(map(len, tensors))) == 1

    def store_batch(
            self,
            observations: TensorDictBase,
            actions: Tensor,
            rewards: Tensor,
            value_funcs: Tensor,
            log_probs: Tensor
    ):
        # assert self._len_is_the_same(observations, actions, rewards, value_funcs, log_probs), \
        #     ('Something wrong with shapes!\n'
        #      f'Found: {list(map(len,[observations, actions, rewards, value_funcs, log_probs]))=}')
        new_ids = self.allocate_ids(len(rewards))
        assert all(torch.less(new_ids, self.size))
        self.obs_buf[new_ids] = observations
        self.act_buf[new_ids] = actions
        self.rew_buf[new_ids] = rewards
        self.val_buf[new_ids] = value_funcs
        self.logp_buf[new_ids] = log_probs
        return new_ids

    def finalize_trajectory(self, path_slice: Tensor, last_val: Tensor, is_full_path: bool = False):
        rews = torch.cat([self.rew_buf[path_slice], last_val])
        if is_full_path:
            self.epoch_aggregator.store(EpRet=torch.sum(rews), EpLen=len(path_slice))
        vals = torch.cat([self.val_buf[path_slice], last_val])

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discounted_cumsum_right(deltas[None, ...], self.gamma * self.lam)[0]

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discounted_cumsum_right(rews[None, ...], self.gamma)[0, :-1]

    def sample_new_trajectories(
            self,
            actor_critic: ActorCritic,
            tqdm_bar: Optional[tqdm_asyncio]
    ) -> None:
        if tqdm_bar is not None:
            tqdm_bar.reset()

        assert self.size % self.batched_env.batch_size == 0, "Fractional batch last step is not implemented!"
        self.ptr = 0
        cur_state = self.batched_env.reset()
        trajectory_ids = [[] for _ in range(self.batched_env.batch_size)]
        while self.ptr < self.size:
            # do one step
            actions, val_funcs, log_probs = actor_critic.step(cur_state)
            next_states, rewards, dones = self.batched_env.step(actions)

            self.epoch_aggregator.store(VVals=val_funcs)

            # store step
            new_ids = self.store_batch(cur_state, actions, rewards, val_funcs, log_probs)

            # update trajectory indices and optionally finalize some of them
            for i, (traj, ind, is_done, vf) in enumerate(zip(trajectory_ids, new_ids, dones, val_funcs)):
                traj.append(ind)
                if is_done or len(traj) >= self.max_episode_len or self.ptr >= self.size:
                    tensor_ids = torch.stack(traj)
                    if is_done:
                        last_v = torch.tensor([0], dtype=vf.dtype, device=vf.device)
                        log_ep_ret = True
                    else:
                        # for early stopping predict value function with critic
                        last_v = torch.stack([vf])
                        log_ep_ret = False
                        dones[i] = True
                    self.finalize_trajectory(tensor_ids, last_v, log_ep_ret)
                    traj.clear()
            cur_state = next_states
            cur_state = self.batched_env.reset(dones)
            if tqdm_bar is not None:
                tqdm_bar.update(len(actions))

        assert all(len(tr) == 0 for tr in trajectory_ids), 'Something wrong with trajectory finalizing!'
