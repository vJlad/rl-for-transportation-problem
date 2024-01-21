from src.batched_env.env_wrapper import BatchedEnvWrapper, MaxDStrategy
from src.batched_env.layer import BatchedLayer
from src.batched_env.supplier_network import BatchedSupplierNetwork

import torch


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
