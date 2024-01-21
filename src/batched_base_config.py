from src.batched_env.env_wrapper import BatchedEnvWrapper, MaxDStrategy, BatchedDStrategy
from src.batched_env.layer import BatchedLayer
from src.batched_env.supplier_network import BatchedSupplierNetwork

import torch


def create_batched_base_supplier_network(batch_size: int, d_strategy: BatchedDStrategy, device = 'cpu'):
    r_size, i1_size, j_size, i2_size, k_size = 4, 3, 3, 3, 6
    Ci1 = torch.tensor([16, 18, 17], dtype=torch.long, device=device)
    Fi1 = torch.tensor([10, 15, 13], dtype=torch.long, device=device)
    Ci1r = torch.tensor([[5, 7, 4, 6], [2, 4, 6, 3], [3, 5, 2, 7]], dtype=torch.long, device=device)
    Fr = torch.tensor([243, 281, 268, 179], dtype=torch.long, device=device)
    Crj = torch.tensor([[8, 5, 6], [6, 5, 6], [5, 5, 4], [4, 7, 7]], dtype=torch.long, device=device)
    Cj = torch.tensor([2, 1, 3], dtype=torch.long, device=device)
    Fj = torch.tensor([700, 700, 700], dtype=torch.long, device=device)
    Ci2 = torch.tensor([12, 14, 13], dtype=torch.long, device=device)
    Fi2 = torch.tensor([30, 25, 40], dtype=torch.long, device=device)
    Ci2j = torch.tensor([[3, 3, 4], [5, 2, 4], [7, 4, 6]], dtype=torch.long, device=device)
    Cjk = torch.tensor([[4, 4, 4, 7, 6, 6], [5, 6, 7, 7, 6, 8], [7, 4, 7, 7, 4, 6]], dtype=torch.long, device=device)
    Fk = torch.tensor([59, 141, 44, 43, 123, 286], dtype=torch.long, device=device)
    cap1 = 50
    cap2 = 20

    layers_config = [
        dict(
            cap=cap1,
            c_flow=torch.tile(Crj[:, None, :], (1, i1_size, 1)),
            c_edge=torch.tile((Ci1 + Ci1r.T)[..., None], (1, 1, j_size)),
            f_in=Fr,
            f_hid=Fi1,
        ),
        dict(
            cap=cap2,
            c_flow=torch.tile((Cj[:, None] + Cjk)[:, None, :], (1, i2_size, 1)),
            c_edge=torch.tile((Ci2 + Ci2j.T)[..., None], (1, 1, k_size)),
            f_in=Fj,
            f_hid=Fi2,
        ),
    ]
    demand = Fk
    layers = [BatchedLayer(batch_size=batch_size, **config) for config in layers_config]
    return BatchedEnvWrapper(BatchedSupplierNetwork(layers, demand), d_strategy)
