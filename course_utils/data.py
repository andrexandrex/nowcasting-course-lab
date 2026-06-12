"""Data loading and prediction path helpers for the course lab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SAMPLE_FILES = [
    "barranca_seq_20240426_2000_patch_04_rain_rate.npy",
    "barranca_seq_20240508_0600_patch_05_rain_rate.npy",
    "barranca_seq_20240512_0800_patch_03_rain_rate.npy",
    "guaviare_seq_20240329_0700_patch_05_rain_rate.npy",
    "guaviare_seq_20240329_0800_patch_03_rain_rate.npy",
]

SAMPLE_FILES_piura = [
    "piura_furuno_rr_seq_20250326_0100_len25_stride12.npy",
    "piura_furuno_rr_seq_20250327_0600_len25_stride12.npy",
    "piura_furuno_rr_seq_20250329_0900_len25_stride12.npy",
]

INPUT_FRAMES = 13
PRED_FRAMES = 12
EXPECTED_SAMPLE_SHAPE = (25, 128, 128)
LEAD_MINUTES = np.arange(5, 65, 5)
MAX_RAIN = 60.0


@dataclass(frozen=True)
class CoursePaths:
    root: Path
    sample_dir: Path
    prediction_dir: Path
    persistence_dir: Path
    earthformer_dir: Path
    cascast_dir: Path
    model_dir: Path
    checkpoints_dir: Path


def find_project_root(start: Path | None = None) -> Path:
    """Find nowcasting_course_lab from a notebook, script, or course root."""
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "environment.yml").exists() and (candidate / "course_utils").exists():
            return candidate
    raise FileNotFoundError(
        "No pude encontrar la carpeta nowcasting_course_lab. "
        "Ejecuta el notebook desde la carpeta del curso o desde notebooks/."
    )


def get_paths(root: Path | None = None) -> CoursePaths:
    root = find_project_root(root)
    prediction_dir = root / "outputs" / "predictions"
    return CoursePaths(
        root=root,
        sample_dir=root / "data" / "samples",
        prediction_dir=prediction_dir,
        persistence_dir=prediction_dir / "persistence",
        earthformer_dir=prediction_dir / "earthformer",
        cascast_dir=prediction_dir / "cascast",
        model_dir=prediction_dir / "model",
        checkpoints_dir=root / "checkpoints",
    )


def clean_rain_array(array: np.ndarray) -> np.ndarray:
    """Convert invalid/background values to 0 and return float32."""
    return np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def load_sample(sample_name: str, paths: CoursePaths | None = None) -> np.ndarray:
    paths = paths or get_paths()
    path = paths.sample_dir / sample_name
    if not path.exists():
        raise FileNotFoundError(
            f"No encontre {path}. Ejecuta: python scripts/download_assets.py"
        )
    sequence = clean_rain_array(np.load(path))
    if sequence.shape != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(f"{sample_name}: forma {sequence.shape}; esperaba {EXPECTED_SAMPLE_SHAPE}.")
    return sequence


def split_sequence(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sequence.shape[0] < INPUT_FRAMES + PRED_FRAMES:
        raise ValueError(f"La secuencia necesita al menos 25 cuadros; tiene {sequence.shape[0]}.")
    inputs = sequence[:INPUT_FRAMES]
    target = sequence[INPUT_FRAMES : INPUT_FRAMES + PRED_FRAMES]
    return inputs, target


def make_persistence_prediction(
    inputs: np.ndarray,
    pred_frames: int = PRED_FRAMES,
    save_to: "Path | None" = None,
) -> np.ndarray:
    last_observed_frame = clean_rain_array(inputs)[INPUT_FRAMES - 1]
    pred = np.repeat(last_observed_frame[None, :, :], pred_frames, axis=0).astype(np.float32)
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_to, pred)
    return pred


def prediction_candidates(sample_name: str, paths: CoursePaths | None = None) -> list[tuple[str, Path]]:
    paths = paths or get_paths()
    return [
        ("cascast", paths.cascast_dir / sample_name),
        ("earthformer", paths.earthformer_dir / sample_name),
        ("modelo", paths.model_dir / sample_name),
        ("persistencia", paths.persistence_dir / sample_name),
    ]


def load_prediction(
    sample_name: str,
    inputs: np.ndarray,
    paths: CoursePaths | None = None,
    prefer: tuple[str, ...] = ("cascast", "earthformer", "modelo", "persistencia"),
) -> tuple[np.ndarray, str]:
    paths = paths or get_paths()
    candidates = {label: path for label, path in prediction_candidates(sample_name, paths)}
    for label in prefer:
        path = candidates.get(label)
        if path is not None and path.exists():
            pred = clean_rain_array(np.load(path))
            if pred.shape != (PRED_FRAMES, inputs.shape[-2], inputs.shape[-1]):
                raise ValueError(f"{path}: forma de prediccion inesperada {pred.shape}.")
            return pred, label
    return make_persistence_prediction(inputs), "persistencia creada en memoria"


def save_prediction(prediction: np.ndarray, sample_name: str, model_name: str, paths: CoursePaths | None = None) -> Path:
    paths = paths or get_paths()
    out_dir = paths.prediction_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / sample_name
    np.save(out_path, clean_rain_array(prediction))
    return out_path

