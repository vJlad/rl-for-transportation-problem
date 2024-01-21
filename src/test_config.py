from src.batched_env.env_wrapper import BatchedEnvWrapper, MaxDStrategy
from src.batched_env.layer import BatchedLayer
from src.batched_env.supplier_network import BatchedSupplierNetwork
from src.batched_env.layer import Layer
from src.batched_env.supplier_network import SupplierNetwork
from src.batched_env.gym_env_wrapper import GymEnvWrapper

import numpy as np
import torch


def create_test_one_stage_supplier_network():
    cap = 5
    in_size, hid_size, out_size = 2, 3, 4
    c_flow = np.ones((in_size, hid_size, out_size), dtype=np.int64)
    c_edge = np.ones((in_size, hid_size, out_size), dtype=np.int64) * 2
    f_in = np.array([6, 7], dtype=np.int64)
    f_hid = np.array([1, 2, 30], dtype=np.int64)
    f_out = np.array([4, 3, 2, 1], dtype=np.int64)
    layers_config = [
        dict(
            cap=cap,
            c_flow=c_flow,
            c_edge=c_edge,
            f_in=f_in,
            f_hid=f_hid,
        )
    ]
    demand = f_out
    layers = [Layer(**config) for config in layers_config]
    return GymEnvWrapper(SupplierNetwork(layers, demand))


def create_batched_test_one_stage_supplier_network():
    cap = 2
    in_size, hid_size, out_size = 2, 3, 4
    c_flow = torch.ones((in_size, hid_size, out_size), dtype=torch.long)
    c_edge = torch.ones((in_size, hid_size, out_size), dtype=torch.long) * 2
    f_in = torch.tensor([6, 7], dtype=torch.long)
    f_hid = torch.tensor([1, 2, 30], dtype=torch.long)
    f_out = torch.tensor([4, 3, 2, 1], dtype=torch.long)
    layers_config = [
        dict(
            batch_size=5,
            cap=cap,
            c_flow=c_flow,
            c_edge=c_edge,
            f_in=f_in,
            f_hid=f_hid,
        )
    ]
    demand = f_out
    layers = [BatchedLayer(**config) for config in layers_config]
    return BatchedEnvWrapper(BatchedSupplierNetwork(layers, demand), MaxDStrategy())
