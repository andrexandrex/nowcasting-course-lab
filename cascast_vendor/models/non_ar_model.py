# Stripped subset of the original CasCast models/non_ar_model.py.
# Only __init__ is needed for inference; all training/evaluation methods
# and megatron/wandb imports are removed.
import time
from models.model import basemodel


class non_ar_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()
        self.max_rain = 60.0
