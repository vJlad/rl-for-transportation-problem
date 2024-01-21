from typing import Union

from torchmetrics.aggregation import MaxMetric, MinMetric, MeanMetric, BaseAggregator
from torchmetrics import MetricCollection, Metric
import torch


class StdMetric(BaseAggregator):
    def __init__(self, nan_strategy: Union[str, float] = "warn", **kwargs):
        super().__init__(
            "sum",
            torch.tensor(0.0),
            nan_strategy,
            state_name="mean_value",
            **kwargs
        )
        self.add_state("sum", default=torch.tensor(0, dtype=torch.float, device=self.device), dist_reduce_fx="sum")
        self.add_state("sum_sq", default=torch.tensor(0, dtype=torch.float, device=self.device), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.tensor(0, dtype=torch.float, device=self.device), dist_reduce_fx="sum")

    def update(self, val: Union[float, torch.Tensor]):
        if not isinstance(val, torch.Tensor):
            val = torch.as_tensor(val, dtype=torch.float, device=self.device)
        self.sum += val.sum()
        self.sum_sq += (val ** 2).sum()
        self.cnt += val.numel()

    def compute(self) -> torch.Tensor:
        return torch.sqrt(torch.clip(self.sum_sq / self.cnt - (self.sum / self.cnt) ** 2, min=0))


class EpochMetricsAggregator:
    def __init__(self, key2metric: dict[str, Metric]):
        self.key2metric = key2metric
        self.other_values = {}

    def store(self, **kwargs):
        for key, val in kwargs.items():
            if key in self.key2metric:
                self.key2metric[key].update(val)
            else:
                assert key not in self.other_values, f"You already set {key=} this iteration. Maybe you forgot to call 'compute_and_clear' method"
                self.other_values[key] = val

    def close_epoch(self):
        ans = self.other_values
        self.other_values = {}
        for key, metric in self.key2metric.items():
            result = metric.compute()
            if isinstance(result, dict):
                for metric_name, val in result.items():
                    name = f'{key}_{metric_name}'
                    assert name not in ans, f"Something went wrong with metric {name=}. This metric is logged al least twice!"
                    ans[name] = val
            else:
                ans[key] = result
            metric.reset()
        return ans


FullAggregator = MetricCollection({
    'Min': MinMetric(),
    'Max': MaxMetric(),
    'Mean': MeanMetric(),
    'Std': StdMetric(),
})
