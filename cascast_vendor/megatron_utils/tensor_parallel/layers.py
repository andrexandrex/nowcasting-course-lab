# Stubs for single-GPU inference — behave like nn.Linear.
import torch.nn as nn


class ColumnParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, gather_output=True,
                 async_tensor_model_parallel_allreduce=False,
                 use_cpu_initialization=False, **kwargs):
        super().__init__(in_features, out_features)

    def forward(self, x):
        return super().forward(x), None


class RowParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, input_is_parallel=False,
                 use_cpu_initialization=False, **kwargs):
        super().__init__(in_features, out_features)

    def forward(self, x):
        return super().forward(x), None
