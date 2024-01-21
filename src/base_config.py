from src.batched_env.layer import Layer
from src.batched_env.supplier_network import SupplierNetwork
from src.batched_env.gym_env_wrapper import GymEnvWrapper, DStrategy

import numpy as np


def create_base_supplier_network(d_strategy: DStrategy):
    r_size, i1_size, j_size, i2_size, k_size = 4, 3, 3, 3, 6
    Ci1 = np.array([16, 18, 17], dtype=np.int64)
    Fi1 = np.array([10, 15, 13], dtype=np.int64)
    Ci1r = np.array([[5, 7, 4, 6], [2, 4, 6, 3], [3, 5, 2, 7]], dtype=np.int64)
    Fr = np.array([243, 281, 268, 179], dtype=np.int64)
    Crj = np.array([[8, 5, 6], [6, 5, 6], [5, 5, 4], [4, 7, 7]], dtype=np.int64)
    Cj = np.array([2, 1, 3], dtype=np.int64)
    Fj = np.array([700, 700, 700], dtype=np.int64)
    Ci2 = np.array([12, 14, 13], dtype=np.int64)
    Fi2 = np.array([30, 25, 40], dtype=np.int64)
    Ci2j = np.array([[3, 3, 4], [5, 2, 4], [7, 4, 6]], dtype=np.int64)
    Cjk = np.array([[4, 4, 4, 7, 6, 6], [5, 6, 7, 7, 6, 8], [7, 4, 7, 7, 4, 6]], dtype=np.int64)
    Fk = np.array([59, 141, 44, 43, 123, 286], dtype=np.int64)
    cap1 = 50
    cap2 = 20

    layers_config = [
        dict(
            cap=cap1,
            c_flow=np.tile(Crj[:, None, :], (1, i1_size, 1)),
            c_edge=np.tile((Ci1 + Ci1r.T)[..., None], (1, 1, j_size)),
            f_in=Fr,
            f_hid=Fi1,
        ),
        dict(
            cap=cap2,
            c_flow=np.tile((Cj[:, None] + Cjk)[:, None, :], (1, i2_size, 1)),
            c_edge=np.tile((Ci2 + Ci2j.T)[..., None], (1, 1, k_size)),
            f_in=Fj,
            f_hid=Fi2,
        ),
    ]
    demand = Fk
    layers = [Layer(**config) for config in layers_config]
    return GymEnvWrapper(SupplierNetwork(layers, demand), d_strategy)
