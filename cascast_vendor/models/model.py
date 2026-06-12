# Stripped subset of the original CasCast models/model.py.
# Removes megatron_utils, wandb, Ceph, DDP, optimizer/lr-scheduler building,
# and all training/evaluation methods.  Only __init__ (model construction)
# and load_checkpoint (inference weight loading) are kept.
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import utils.misc as utils
from utils.misc import is_dist_avail_and_initialized


class basemodel(nn.Module):
    def __init__(self, logger, **params) -> None:
        super().__init__()
        self.model = {}
        self.sub_model_name = []
        self.params = params
        self.logger = logger

        self.data_type = params.get("data_type", "fp32")
        if self.data_type == "bf16":
            self.data_type = torch.bfloat16
        elif self.data_type == "fp16":
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32

        self.metrics_type = params.get("metrics_type", "None")
        self.use_ceph = False
        self.extra_params = params.get("extra_params", {})
        self.enabled_amp = self.extra_params.get("enabled_amp", False)
        # Use the device-agnostic API when available (torch >= 2.3) to avoid
        # a FutureWarning about the deprecated positional GradScaler form.
        try:
            self.gscaler = torch.amp.GradScaler('cuda', enabled=self.enabled_amp)
        except TypeError:
            self.gscaler = amp.GradScaler(enabled=self.enabled_amp)
        self.metric_best = None
        self.begin_epoch = 0
        self.begin_step = 0

        self.visualizer_params = params.get("visualizer", {})
        self.visualizer_type = self.visualizer_params.get("visualizer_type", None)
        self.visualizer = None

        # Build sub-models from config.
        sub_model = params.get('sub_model', {})
        for key in sub_model:
            if key == 'EarthFormer_xy':
                from networks.earthformer_xy import EarthFormer_xy
                self.model[key] = EarthFormer_xy(**sub_model['EarthFormer_xy'])
            elif key == 'autoencoder_kl':
                from networks.autoencoder_kl import autoencoder_kl
                self.model[key] = autoencoder_kl(config=sub_model['autoencoder_kl'])
            elif key == 'casformer':
                from networks.casformer import CasFormer
                self.model[key] = CasFormer(**sub_model['casformer'])
            else:
                raise NotImplementedError(f'Sub-model not supported in vendor: {key}')
            self.sub_model_name.append(key)

        # Metrics (inference sets metrics_type="None").
        if self.metrics_type == 'None':
            self.eval_metrics = None
        elif self.metrics_type == 'SEVIRSkillScore':
            from utils.metrics import SEVIRSkillScore
            skill_params = params.get("skill_score_params", {})
            seq_len = params.get("sevir_seq_len", 12)
            self.eval_metrics = SEVIRSkillScore(
                layout='NTCHW', seq_len=seq_len,
                dist_eval=is_dist_avail_and_initialized(), **skill_params)
        else:
            self.eval_metrics = None

        for key in self.model:
            self.model[key].eval()

        # Auto-load checkpoint if specified in config (inference path removes this).
        checkpoint_path = self.extra_params.get("checkpoint_path", None)
        if checkpoint_path is not None:
            self.load_checkpoint(
                checkpoint_path,
                load_model=True, load_optimizer=False,
                load_scheduler=False, load_epoch=False, load_metric_best=False,
            )

    def _apply_freeze_params(self, freeze_params):
        pass

    def load_checkpoint(self, checkpoint_path, load_model=True, load_optimizer=False,
                        load_scheduler=False, load_epoch=True, load_metric_best=True,
                        **kwargs):
        if not os.path.exists(checkpoint_path):
            if self.logger:
                self.logger.info(f"checkpoint not found: {checkpoint_path}")
            return
        checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if load_model and 'model' in checkpoint_dict:
            checkpoint_model = checkpoint_dict['model']
            submodels = list(self.model.keys())
            for key in checkpoint_model:
                if key not in submodels:
                    if self.logger:
                        self.logger.warning(f"skip loading sub-model: {key}")
                    continue
                new_state_dict = OrderedDict()
                for k, v in checkpoint_model[key].items():
                    name = k[7:] if k[:6] == "module" else k
                    new_state_dict[name] = v
                self.model[key].load_state_dict(new_state_dict, strict=False)
        if load_epoch and 'epoch' in checkpoint_dict:
            self.begin_epoch = checkpoint_dict['epoch']
        if load_metric_best and 'metric_best' in checkpoint_dict:
            self.metric_best = checkpoint_dict['metric_best']
        if self.logger:
            epoch = checkpoint_dict.get('epoch', '?')
            metric = checkpoint_dict.get('metric_best', '?')
            self.logger.info(f"loaded checkpoint  epoch={epoch}  metric_best={metric}")

    def to(self, device):
        for key in self.model:
            self.model[key].to(device)
        return self
