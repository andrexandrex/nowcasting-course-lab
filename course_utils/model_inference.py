"""Optional advanced EF/AE/CasCast inference helpers.

These functions are intentionally imported lazily. The basic course notebooks do
not need PyTorch, CUDA, timm, or the CasCast repository.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from course_utils.data import INPUT_FRAMES, MAX_RAIN, PRED_FRAMES, clean_rain_array, get_paths, save_prediction


class ModelDemoUnavailable(RuntimeError):
    """Raised when the optional GPU/CasCast demo cannot run in this environment."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise ModelDemoUnavailable("Falta PyYAML. Instala `pyyaml` para usar el demo avanzado.") from exc
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_model_demo_config(root: Path | None = None, config_path: Path | None = None) -> dict[str, Any]:
    paths = get_paths(root)
    config_path = config_path or (paths.root / "config" / "model_demo_paths.yaml")
    cfg = _load_yaml(config_path)

    cascast_repo = (paths.root / cfg["cascast_repo"]).resolve()
    checkpoints = {
        key: (paths.root / value).resolve()
        for key, value in cfg.get("checkpoints", {}).items()
    }
    cfg["cascast_repo"] = cascast_repo
    cfg["checkpoints"] = checkpoints
    return cfg


def resolve_device(device: str = "auto"):
    """Resolve auto/cpu/cuda into a torch.device with clear errors."""
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ModelDemoUnavailable("Se pidio --device cuda, pero CUDA no esta disponible.")
    if device not in ("cpu", "cuda"):
        raise ModelDemoUnavailable(f"Dispositivo no soportado: {device}. Usa auto, cpu o cuda.")
    return torch.device(device)


def check_model_demo_ready(config: dict[str, Any] | None = None, device: str | None = None) -> tuple[bool, str]:
    """Return a readiness flag and a Spanish explanation."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "PyTorch no esta instalado en este ambiente."

    import torch

    config = config or load_model_demo_config()
    requested_device = device or str(config.get("device", "auto"))
    try:
        resolved_device = resolve_device(requested_device)
    except ModelDemoUnavailable as exc:
        return False, str(exc)

    cascast_repo = Path(config["cascast_repo"])
    if not cascast_repo.exists():
        return False, f"No encontre el repo CasCast en {cascast_repo}."

    missing = [str(path) for path in config["checkpoints"].values() if not Path(path).exists()]
    if missing:
        return False, "Faltan checkpoints: " + ", ".join(missing)

    if resolved_device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        return True, f"Listo para ejecutar inferencia en CUDA: {gpu_name}."
    return (
        True,
        "Listo para ejecutar inferencia en CPU. EarthFormer deberia ser razonable; "
        "CasCast puede ser muy lento porque la difusion repite muchos pasos.",
    )


def _prepare_cascast_imports(cascast_repo: Path) -> None:
    _patch_huggingface_hub_for_vendored_diffusers()
    cascast_repo = cascast_repo.resolve()
    if str(cascast_repo) not in sys.path:
        sys.path.insert(0, str(cascast_repo))


def _patch_huggingface_hub_for_vendored_diffusers() -> None:
    """Patch newer huggingface_hub for CasCast's older vendored diffusers.

    The CasCast repo includes an older copy of diffusers that imports
    `hf_cache_home` from `huggingface_hub.constants`. Newer huggingface_hub
    versions removed that name, while keeping `HUGGINGFACE_HUB_CACHE`.
    Adding the alias before importing CasCast keeps the course environment
    compatible without forcing students to downgrade huggingface_hub.
    """
    try:
        import huggingface_hub
        import huggingface_hub.constants as hf_constants
    except Exception:
        return

    if not hasattr(hf_constants, "hf_cache_home") and hasattr(hf_constants, "HUGGINGFACE_HUB_CACHE"):
        hf_constants.hf_cache_home = hf_constants.HUGGINGFACE_HUB_CACHE

    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def get_token():
                return None

            @staticmethod
            def save_token(token):
                return None

            @staticmethod
            def delete_token():
                return None

        huggingface_hub.HfFolder = HfFolder

    if not hasattr(huggingface_hub, "cached_download"):
        def cached_download(*args, **kwargs):
            # Older diffusers used cached_download for optional Hub/dynamic-module
            # flows. Local scheduler inference should not call it, but when it is
            # called with a Hub filename we delegate to hf_hub_download.
            if "repo_id" in kwargs and "filename" in kwargs:
                return huggingface_hub.hf_hub_download(
                    repo_id=kwargs["repo_id"],
                    filename=kwargs["filename"],
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
            if args and isinstance(args[0], (str, os.PathLike)) and os.path.exists(args[0]):
                return str(args[0])
            raise RuntimeError(
                "CasCast intento usar cached_download de Hugging Face. "
                "El demo del curso solo soporta inferencia local con checkpoints ya descargados."
            )


def _build_model_from_config(config_file: Path, checkpoint_paths: dict[str, Path], model_type: str, device: str):
    import torch

    _prepare_cascast_imports(config_file.parents[2])
    from utils.builder import ConfigBuilder

    cfg = _load_yaml(config_file)
    cfg["logger"] = logging.getLogger(f"nowcasting_course.{model_type}")
    cfg["model"]["params"]["use_ceph"] = False
    cfg["model"]["params"]["metrics_type"] = "None"
    cfg["model"]["params"]["visualizer"] = {"visualizer_type": None}

    if model_type == "earthformer":
        cfg["model"]["params"]["extra_params"].pop("checkpoint_path", None)
    elif model_type == "cascast":
        extra = cfg["model"]["params"]["extra_params"]
        extra["predictor_checkpoint_path"] = str(checkpoint_paths["earthformer"])
        extra["autoencoder_checkpoint_path"] = str(checkpoint_paths["autoencoder"])
    else:
        raise ValueError(f"model_type desconocido: {model_type}")

    builder = ConfigBuilder(**cfg)
    model = builder.get_model()
    model.to(torch.device(device))
    model.eval()
    return model


def _load_checkpoint(model, checkpoint_path: Path) -> None:
    model.load_checkpoint(
        str(checkpoint_path),
        load_model=True,
        load_optimizer=False,
        load_scheduler=False,
        load_epoch=False,
        load_metric_best=False,
    )


def run_earthformer_demo(
    sequences: np.ndarray,
    sample_names: list[str] | None = None,
    root: Path | None = None,
    device: str | None = None,
    cpu_threads: int | None = None,
    save_outputs: bool = True,
) -> dict[str, np.ndarray]:
    """Run only EarthFormer for a small batch and optionally save outputs."""
    import torch

    paths = get_paths(root)
    config = load_model_demo_config(paths.root)
    requested_device = device or str(config.get("device", "auto"))
    ok, message = check_model_demo_ready(config, requested_device)
    if not ok:
        raise ModelDemoUnavailable(message)

    torch_device = resolve_device(requested_device)
    if cpu_threads is not None and torch_device.type == "cpu":
        torch.set_num_threads(int(cpu_threads))

    cascast_repo = Path(config["cascast_repo"])
    _prepare_cascast_imports(cascast_repo)

    sequences = clean_rain_array(np.asarray(sequences))
    if sequences.ndim == 3:
        sequences = sequences[None, ...]
    if sequences.shape[1:] != (INPUT_FRAMES + PRED_FRAMES, 128, 128):
        raise ValueError(f"sequences debe tener forma (B,25,128,128); llego {sequences.shape}.")

    sample_names = sample_names or [f"case_{i:03d}.npy" for i in range(sequences.shape[0])]
    checkpoint_paths = config["checkpoints"]
    ef_cfg = cascast_repo / "configs" / "sevir_used" / "EarthFormer_ideam.yaml"

    earthformer = _build_model_from_config(ef_cfg, checkpoint_paths, "earthformer", str(torch_device))
    _load_checkpoint(earthformer, checkpoint_paths["earthformer"])

    with torch.no_grad():
        x = torch.from_numpy(sequences[:, :INPUT_FRAMES, None]).float().to(torch_device)
        x01 = torch.clamp(x, 0.0, MAX_RAIN) / MAX_RAIN
        ef_key = list(earthformer.model.keys())[0]
        ef01 = torch.clamp(earthformer.model[ef_key](x01), 0.0, 1.0)
        earthformer_mm = (ef01[:, :, 0] * MAX_RAIN).detach().cpu().numpy().astype(np.float32)

    if save_outputs:
        for i, sample_name in enumerate(sample_names):
            save_prediction(earthformer_mm[i], sample_name, "earthformer", paths)

    return {"earthformer": earthformer_mm}


def _encode_raw(autoencoder_model, x):
    """Encode normalized frames to raw, unscaled autoencoder latents."""
    if hasattr(autoencoder_model, "module"):
        autoencoder_model = autoencoder_model.module
    return autoencoder_model.net.encode(x).sample()


def run_optional_model_demo(
    sequences: np.ndarray,
    sample_names: list[str] | None = None,
    root: Path | None = None,
    device: str | None = None,
    cpu_threads: int | None = None,
    ddim_steps: int | None = None,
    ensemble_member: int = 1,
    cfg_weight: float = 1.0,
    scale_factor: float | None = None,
    save_outputs: bool = True,
) -> dict[str, np.ndarray]:
    """Run EarthFormer and CasCast for a small batch.

    Parameters
    ----------
    sequences:
        Array with shape (B,25,128,128) or (25,128,128), in mm/h.
    sample_names:
        Names used when saving outputs. Required only when save_outputs=True.
    scale_factor:
        Latent scale factor for the CasCast diffusion. When ``None`` the model
        estimates it from the batch via ``init_scale_factor`` (legacy behavior).
        Pass the value the checkpoint was trained with (e.g. 0.702658) for
        reproducible results that do not depend on the batch.

    Returns
    -------
    dict with `earthformer` and `cascast`, each shaped (B,12,128,128).
    """
    import torch
    from einops import rearrange

    paths = get_paths(root)
    config = load_model_demo_config(paths.root)
    requested_device = device or str(config.get("device", "auto"))
    ok, message = check_model_demo_ready(config, requested_device)
    if not ok:
        raise ModelDemoUnavailable(message)

    torch_device = resolve_device(requested_device)
    if cpu_threads is not None and torch_device.type == "cpu":
        torch.set_num_threads(int(cpu_threads))

    cascast_repo = Path(config["cascast_repo"])
    _prepare_cascast_imports(cascast_repo)

    sequences = clean_rain_array(np.asarray(sequences))
    if sequences.ndim == 3:
        sequences = sequences[None, ...]
    if sequences.shape[1:] != (INPUT_FRAMES + PRED_FRAMES, 128, 128):
        raise ValueError(f"sequences debe tener forma (B,25,128,128); llego {sequences.shape}.")

    sample_names = sample_names or [f"case_{i:03d}.npy" for i in range(sequences.shape[0])]
    checkpoint_paths = config["checkpoints"]

    ef_cfg = cascast_repo / "configs" / "sevir_used" / "EarthFormer_ideam.yaml"
    cc_cfg = cascast_repo / "configs" / "sevir_used" / "cascast_diffusion_ideam.yaml"

    earthformer = _build_model_from_config(ef_cfg, checkpoint_paths, "earthformer", str(torch_device))
    _load_checkpoint(earthformer, checkpoint_paths["earthformer"])

    cascast = _build_model_from_config(cc_cfg, checkpoint_paths, "cascast", str(torch_device))
    _load_checkpoint(cascast, checkpoint_paths["diffusion"])
    cascast.sample_noise_scheduler.set_timesteps(int(ddim_steps or config.get("ddim_steps", 20)))
    cascast.ens_member = ensemble_member
    cascast.cfg_weight = cfg_weight
    if scale_factor is not None:
        cascast.scale_factor = torch.tensor(float(scale_factor), device=torch_device)

    with torch.no_grad():
        x = torch.from_numpy(sequences[:, :INPUT_FRAMES, None]).float().to(torch_device)
        y = torch.from_numpy(sequences[:, INPUT_FRAMES:, None]).float().to(torch_device)
        x01 = torch.clamp(x, 0.0, MAX_RAIN) / MAX_RAIN
        y01 = torch.clamp(y, 0.0, MAX_RAIN) / MAX_RAIN

        ef_key = list(earthformer.model.keys())[0]
        ef01 = torch.clamp(earthformer.model[ef_key](x01), 0.0, 1.0)
        earthformer_mm = (ef01[:, :, 0] * MAX_RAIN).detach().cpu().numpy().astype(np.float32)

        autoencoder = cascast.model["autoencoder_kl"]
        full01 = torch.cat([x01, y01], dim=1)
        b, t_total, c, h, w = full01.shape
        z_full = _encode_raw(autoencoder, full01.reshape(-1, c, h, w).contiguous())
        z_full = rearrange(z_full, "(b t) c h w -> b t c h w", b=b, t=t_total)
        z_past = z_full[:, :INPUT_FRAMES]
        z_future = z_full[:, INPUT_FRAMES:]

        _, t_future, _, _, _ = ef01.shape
        z_cond = _encode_raw(autoencoder, ef01.reshape(-1, c, h, w).contiguous())
        z_cond = rearrange(z_cond, "(b t) c h w -> b t c h w", b=b, t=t_future)
        z_cond = cascast._apply_past_context(z_cond, z_past)

        if scale_factor is None and float(cascast.scale_factor.item()) == 1.0:
            cascast.init_scale_factor(z_future)

        z_samples = cascast.denoise(
            template_data=z_future * cascast.scale_factor,
            cond_data=z_cond * cascast.scale_factor,
            bs=b,
            vis=True,
            cfg=cfg_weight,
            ensemble_member=ensemble_member,
        )
        if z_samples.ndim == 6:
            z_samples = z_samples.mean(dim=1)

        pred01 = cascast.decode_stage(rearrange(z_samples, "b t c h w -> (b t) c h w").contiguous())
        pred01 = torch.clamp(pred01, 0.0, 1.0)
        pred01 = rearrange(pred01, "(b t) c h w -> b t c h w", b=b, t=PRED_FRAMES)
        cascast_mm = (pred01[:, :, 0] * MAX_RAIN).detach().cpu().numpy().astype(np.float32)

    if save_outputs:
        for i, sample_name in enumerate(sample_names):
            save_prediction(earthformer_mm[i], sample_name, "earthformer", paths)
            save_prediction(cascast_mm[i], sample_name, "cascast", paths)

    return {"earthformer": earthformer_mm, "cascast": cascast_mm}
