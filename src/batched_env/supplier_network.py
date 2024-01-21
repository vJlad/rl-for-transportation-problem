import logging
from typing import Union, List, Sequence, Optional

from src.batched_env.layer import BatchedLayer

import numpy as np
import torch
from torch import Tensor


class BatchedSupplierNetwork:
    def __init__(self, base_layers: Sequence[BatchedLayer], demand: Tensor):
        assert all(le.base_shape[-1] == ri.base_shape[0] for le, ri in zip(base_layers[:-1], base_layers[1:]))
        assert len(set(layer.batch_size for layer in base_layers)) == 1
        self.demand = demand.clone()
        self.base_layers = base_layers
        self.instances_size = tuple(
            sum((list(layer.base_shape[:-1]) for layer in base_layers), start=[])
            + [base_layers[-1].base_shape[-1]]
        )

    def observation_size(self):
        return sum(layer.observation_size() for layer in self.base_layers)

    def number_of_actions(self) -> int:
        return np.prod(self.instances_size)

    def calc_cost(self) -> Tensor:
        return sum(
            (layer.calc_cost() for layer in self.base_layers),
            start=torch.zeros_like(self.base_layers[0].b[..., 0, 0, 0])
        )

    def is_done(self) -> Tensor:
        return torch.eq(self.base_layers[-1].b.sum(dim=(-3, -2)), self.demand).all(dim=-1)

    def encode_action(self, instance_ids: Sequence[Tensor]) -> Tensor:
        action_ids = torch.zeros_like(self.base_layers[0].b[..., 0, 0, 0], dtype=torch.long)
        for cur_ids, cur_sz in zip(instance_ids, self.instances_size):
            action_ids = action_ids * cur_sz + cur_ids
        return action_ids

    def decode_action(self, action_id: Tensor) -> Sequence[Tensor]:
        instance_ids = []
        for sz in self.instances_size[::-1]:
            instance_ids.append(action_id % sz)
            action_id = action_id // sz
        return instance_ids[::-1]

    def step(self, action_id: Tensor, d: Tensor) -> Tensor:
        instance_ids = self.decode_action(action_id)
        init_cost = self.calc_cost()
        for i, layer in enumerate(self.base_layers):
            layer.step(*instance_ids[2 * i: 2 * i + 3], d)
        return init_cost - self.calc_cost()

    def get_state(self) -> Sequence[Tensor]:
        return tuple(layer.get_state() for layer in self.base_layers)

    def load_state(self, state: Sequence[Tensor]) -> None:
        for layer, sub_state in zip(self.base_layers, state):
            layer.load_state(sub_state)

    def get_valid_actions(
            self, d: Optional[Tensor] = None, return_as_mask: bool = False
    ) -> Union[List[Tensor], Tensor]:
        if d is None:
            d = torch.ones_like(self.base_layers[0].b[..., 0, 0, 0])

        mask = None
        inner_axis = []
        for i in range(len(self.base_layers)):
            cur_mask = self.base_layers[i].get_possible_actions(d)
            if mask is None:
                mask = cur_mask
            else:
                cur_mask = cur_mask.reshape([
                    cur_mask.shape[0],
                    *inner_axis,
                    *inner_axis,
                    *cur_mask.shape[1:]
                ])
                mask = torch.logical_and(mask[..., None, None], cur_mask)
            inner_axis.append(1)
        demand_mask = torch.le(self.base_layers[-1].b.sum(dim=(-3, -2)) + d[:, None], self.demand)
        demand_mask = demand_mask.reshape([demand_mask.shape[0], *inner_axis, *inner_axis, demand_mask.shape[1]])
        mask = torch.logical_and(mask, demand_mask)
        if return_as_mask:
            return mask
        raise NotImplementedError()

    def reset(self, ids_or_mask: Optional[Tensor] = None) -> Sequence[Tensor]:
        for layer in self.base_layers:
            layer.reset(ids_or_mask)
        return self.get_state()
