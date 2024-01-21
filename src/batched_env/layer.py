from typing import Union, Tuple, List, Sequence, Optional

import torch
from torch import Tensor
import numpy as np


class BatchedLayer:
    """
    Keeps N independent instances of transportation layer.
    Each layer defined as a flow by every edge (IN x HID x OUT).
    The cost of a flow b on the edge is defined as
        c = Cflow * b + Cedge * ceil(b / cap)

    where cap is the capacity of every edge.
    The flow B for each layer is defined as a 3-dimensional array of shape
    |B| = (|IN| x |HID| x |OUT|)
    Each layer has its own restrictions:
    1. \sum_{h,o} F_{i,h,o} \leq Fin_i, forall i \in IN
    2. \sum_{i,o} ceil(F_{i,h,o} / cap) \leq Fhid_h, forall h \in HID

    The action for an every layer is defined as
    tuple(i, h, o, d), where
    """

    def __init__(
            self,
            batch_size: int,
            cap: int,
            c_flow: Tensor,
            c_edge: Tensor,
            f_in: Tensor,
            f_hid: Tensor,
    ):
        """ (TODO)
        Args:
            batch_size: number of batched layers
            cap: capacity of a single vehicle
            c_flow: cost of a unit flow on every edge (in,hid,out), shape='[IN x HID x OUT]'
            c_edge: cost of a unit vehicle-flow on every edge (in,hid,out), shape='[IN x HID x OUT]'
            f_in: maximal flow through every input vertex, shape='[IN]'
            f_hid: maximal vehicle-flow through every hidden vertex, shape='[HID]'
        """
        assert batch_size > 0
        assert c_flow.shape == c_edge.shape
        # TODO check all shapes!
        assert len(set(map(lambda x: x.dtype, [c_flow, c_edge, f_in, f_hid]))) == 1
        assert cap > 0
        assert all(map(lambda x: torch.min(x) >= 0, [f_in, f_hid]))

        self.batch_size: int = batch_size
        self.c_flow: Tensor = c_flow.clone()
        self.c_edge: Tensor = c_edge.clone()
        self.f_in: Tensor = f_in.clone()
        self.f_hid: Tensor = f_hid.clone()

        self.cap: int = cap
        self.base_shape: Sequence[int] = c_flow.shape
        self.b: Tensor = torch.zeros_like(c_flow, dtype=torch.long)[None].tile(batch_size, 1, 1, 1)

    def number_of_actions(self) -> int:
        return np.prod(self.base_shape)

    def _get_b_cap(self, cap: int) -> Tensor:  # [B, IN, HID, OUT]
        return (self.b + cap - 1) // cap

    def get_possible_actions(self, d: Optional[Tensor] = None) -> Tensor:
        if d is None:
            d = torch.ones_like(self.b[..., 0, 0, 0])
        in_next: Tensor = self.b.sum(dim=(-1, -2)) + d[..., None]  # [B, IN]
        mask: Tensor = torch.le(in_next, self.f_in)[..., None, None]

        cur_b_cap: Tensor = self._get_b_cap(self.cap)  # [B, IN, HID, OUT]
        hid_cur: Tensor = cur_b_cap.sum(dim=(-3, -1))  # [B, IN]
        mask = torch.logical_and(mask, torch.le(hid_cur, self.f_hid)[..., None, :, None])

        self.b += d[..., None, None, None]
        hid_inc: Tensor = self._get_b_cap(self.cap) - cur_b_cap  # [B, IN]
        self.b -= d[..., None, None, None]
        mask = torch.logical_and(
            mask,
            torch.le(hid_cur[..., None, :, None] + hid_inc, self.f_hid[..., None])
        )
        return mask

    def calc_cost(self) -> Tensor:
        return torch.sum(self.c_flow * self.b + self.c_edge * self._get_b_cap(self.cap), dim=(-3, -2, -1))

    def get_state(self) -> Tensor:
        return self.b.clone()

    def step(self, in_id: Tensor, hid_id: Tensor, out_id: Tensor, d: Tensor) -> None:
        self.b[torch.arange(self.batch_size, device=self.c_flow.device), in_id, hid_id, out_id] += d

    def observation_size(self) -> int:
        return np.prod(self.base_shape)

    def load_state(self, state):
        self.b = state.copy()

    def state_is_ok(self) -> Tensor:
        in_next = self.b.sum(dim=(-1, -2))  # [B, IN]
        res = torch.all(torch.le(in_next, self.f_in), dim=-1)  # [B]

        hid_cur = self._get_b_cap(self.cap).sum(dim=(-3, -1))  # [B, HID]
        res = torch.logical_and(res, torch.any(torch.le(hid_cur, self.f_hid), dim=-1))

        return res

    def reset(self, ids_or_mask: Optional[Tensor] = None):
        if ids_or_mask is None:
            self.b[...] = 0
        else:
            self.b[ids_or_mask, ...] = 0
