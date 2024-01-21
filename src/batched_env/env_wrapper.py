from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Any, Optional

from src.batched_env.supplier_network import BatchedSupplierNetwork
from src.batched_env.layer import BatchedLayer

from tensordict import TensorDict
import torch
from torch import Tensor


@dataclass
class BatchedDStrategy(ABC):
    def __call__(self, env: BatchedSupplierNetwork, action: int) -> int:
        raise NotImplementedError()


class DefaultStrategy(BatchedDStrategy):
    def __call__(self, env: BatchedSupplierNetwork, action: Tensor) -> Tensor:
        return torch.ones_like(env.base_layers[0].b[..., 0, 0, 0])


class MaxDStrategy(BatchedDStrategy):  # TODO
    def __call__(self, env: BatchedSupplierNetwork, action: Tensor) -> Tensor:
        instance_ids = env.decode_action(action)
        maxd_by_layers = torch.stack([
            self._extract_max_d_from_layer(layer, *instance_ids[2 * i: 2 * i + 3])
            for i, layer in enumerate(env.base_layers)
        ], dim=0).min(dim=0).values
        id_range = torch.arange(env.base_layers[0].batch_size, device=action.device)
        cur_out = env.base_layers[-1].b[id_range, :, :, instance_ids[-1]].sum(dim=(-1, -2))
        maxd_by_demand = env.demand[instance_ids[-1]] - cur_out
        return torch.where(torch.lt(maxd_by_layers, maxd_by_demand), maxd_by_layers, maxd_by_demand)

    def _extract_max_d_from_layer(self, layer: BatchedLayer, in_id: Tensor, hid_id: Tensor, out_id: Tensor):
        id_range = torch.arange(layer.batch_size, device=in_id.device)
        cur_in = layer.b[id_range, in_id, :, :].sum(dim=(-1, -2))
        max_d_by_in = layer.f_in[in_id] - cur_in
        all_hid = (layer.b[id_range, :, hid_id, :] + layer.cap - 1) // layer.cap
        mx_hid = layer.f_hid[hid_id]
        not_usd_hid_by_others = mx_hid - all_hid.sum(dim=(-1, -2)) + all_hid[id_range, in_id, out_id]
        max_d_by_hid = not_usd_hid_by_others * layer.cap - layer.b[id_range, in_id, hid_id, out_id]
        return torch.where(torch.lt(max_d_by_in, max_d_by_hid), max_d_by_in, max_d_by_hid)


class NearestSufficientChangeDStrategy(BatchedDStrategy):
    def __call__(self, env: BatchedSupplierNetwork, action: Tensor) -> Tensor:
        instance_ids = env.decode_action(action)
        d_by_layers = torch.stack([
            self._extract_d_from_layer(layer, *instance_ids[2 * i: 2 * i + 3])
            for i, layer in enumerate(env.base_layers)
        ]).min(dim=0).values
        id_range = torch.arange(env.base_layers[0].batch_size, device=action.device)
        d_by_demand = (env.demand[instance_ids[-1]] -
                       env.base_layers[-1].b[id_range, :, :, instance_ids[-1]].sum(dim=(-1, -2)))
        return torch.where(torch.lt(d_by_layers, d_by_demand), d_by_layers, d_by_demand)

    def _extract_d_from_layer(self, layer, in_id, hid_id, out_id):
        id_range = torch.arange(layer.batch_size, device=layer.b.device)
        d_by_in = layer.f_in[in_id] - layer.b[id_range, in_id, :, :].sum(dim=(-1, -2))
        d_by_hid = layer.cap - (layer.b[id_range, in_id, hid_id, out_id]) % layer.cap
        return torch.where(torch.lt(d_by_in, d_by_hid), d_by_in, d_by_hid)


@dataclass
class BatchedEnvWrapper:
    supplier_network: BatchedSupplierNetwork
    d_strategy: BatchedDStrategy = DefaultStrategy()

    def __post_init__(self):
        self.batch_size = self.supplier_network.base_layers[0].batch_size

    def get_obs(self):
        return TensorDict({
            'state': TensorDict({
                str(i): sub_state.float()
                for i, sub_state in enumerate(self.supplier_network.get_state())
            }, batch_size=self.batch_size),
            'valid_actions_mask': self.supplier_network.get_valid_actions(return_as_mask=True)
        }, batch_size=self.batch_size)

    def reset(self, ids_or_mask: Optional[Tensor] = None) -> TensorDict:
        self.supplier_network.reset(ids_or_mask)
        return self.get_obs()

    # def _load_state(self, obs: TensorDict):
    #     self.supplier_network.load_state(tuple(
    #         obs['state'][str(i)].long() for i in range(len(self.supplier_network.base_layers))
    #     ))

    def step(self, action) -> Tuple[TensorDict, Tensor, Tensor]:
        d = self.d_strategy(self.supplier_network, action)
        assert torch.all(torch.gt(d, 0))
        reward = self.supplier_network.step(action, d)
        done = self.supplier_network.is_done()
        return self.get_obs(), reward.float(), done
