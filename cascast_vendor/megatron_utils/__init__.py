# Stub megatron_utils for single-GPU / CPU inference.
# All rank/world-size calls return values appropriate for a single process.

class _MPU:
    def get_data_parallel_rank(self): return 0
    def get_data_parallel_world_size(self): return 1
    def get_data_parallel_src_rank(self): return 0
    def get_data_parallel_group(self): return None
    def get_tensor_model_parallel_rank(self): return 0
    def get_tensor_model_parallel_world_size(self): return 1
    def get_ensemble_parallel_group(self): return None
    def initialize_model_parallel(self, *args, **kwargs): pass

mpu = _MPU()
