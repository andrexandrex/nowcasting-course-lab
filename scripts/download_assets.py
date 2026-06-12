#!/usr/bin/env python
"""Descarga datos y checkpoints del laboratorio de nowcasting desde Hugging Face.

Datos (dataset andrexandrex322/ideam-nowcasting-samples):
  samples/        -> data/samples/        (5 secuencias del curso, lo minimo)
  ideam_data/     -> data/ideam_data/     (5 casos + training_dataset/ completo)
  piura_data/     -> data/piura_data/     (casos Piura, tarea 02)
  sophy_data/     -> data/sophy_data/     (casos Sophy, tarea 02)

Checkpoints (modelo andrexandrex322/ideam-nowcasting-earthformer-cascast/ef_ideam_final/):
  ef_ckpt.pth     -> checkpoints/ef_ideam_final/ef_ckpt.pth      (EarthFormer determinista)
  ae_ckpt.pth     -> checkpoints/ef_ideam_final/ae_ckpt.pth      (Autoencoder para CasCast)
  diff_ckpt.pth   -> checkpoints/ef_ideam_final/diff_ckpt.pth    (CasCast difusión)

Ejemplos:
  python scripts/download_assets.py                  # solo las 5 muestras del curso
  python scripts/download_assets.py --checkpoints    # muestras + checkpoints (~4.7 GB)
  python scripts/download_assets.py --piura --sophy  # datos de la tarea 02
  python scripts/download_assets.py --all            # todo (datos + checkpoints, pesado)
Sugerencia: lanza la descarga de checkpoints en una terminal mientras instalas las
librerias de inferencia en otra; el .pth de difusion pesa varios GB.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_utils.data import EXPECTED_SAMPLE_SHAPE, SAMPLE_FILES


SAMPLE_REPO_ID = "andrexandrex322/ideam-nowcasting-samples"
MODEL_REPO_ID = "andrexandrex322/ideam-nowcasting-earthformer-cascast"

# Carpeta del dataset en HF -> carpeta local dentro de data/
DATA_FOLDERS = {
    "samples": "samples",
    "ideam": "ideam_data",
    "piura": "piura_data",
    "sophy": "sophy_data",
}

# Subcarpeta donde están los checkpoints en el repositorio HF
CHECKPOINT_SUBDIR = "ef_ideam_final"

# Mapeo: nombre en HF -> nombre local (ya están correctos, no necesitan renombrado)
CHECKPOINT_MAP = {
    "ef_ckpt.pth": "ef_ckpt.pth",      # EarthFormer determinista
    "ae_ckpt.pth": "ae_ckpt.pth",      # Autoencoder para CasCast
    "diff_ckpt.pth": "diff_ckpt.pth",  # CasCast difusión
}

# Nombres alternativos que algunos scripts podrían esperar (crear enlaces simbólicos)
ALTERNATIVE_NAMES = {
    "checkpoint_epoch_ef.pth": "ef_ckpt.pth",
    "checkpoint_latest_ae.pth": "ae_ckpt.pth",
    "checkpoint_latest_diff.pth": "diff_ckpt.pth",
}


def validate_sample(path: Path) -> None:
    array = np.load(path)
    if array.shape != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(
            f"{path.name}: forma inesperada {array.shape}; se esperaba {EXPECTED_SAMPLE_SHAPE}."
        )


def download_data_folder(root: Path, hf_folder: str) -> Path:
    """Descarga una carpeta del dataset a data/<hf_folder>/ preservando su estructura."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {hf_folder}/ ...")
    snapshot_download(
        repo_id=SAMPLE_REPO_ID,
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=[f"{hf_folder}/**"],
    )
    dest = data_dir / hf_folder
    n = sum(1 for _ in dest.rglob("*.npy"))
    print(f"Listo: {n} archivos .npy en {dest}")
    return dest


def download_samples(root: Path) -> None:
    """Descarga las 5 muestras del curso a data/samples/ y las valida."""
    dest = download_data_folder(root, DATA_FOLDERS["samples"])
    for filename in SAMPLE_FILES:
        validate_sample(dest / filename)
    print(f"Validadas {len(SAMPLE_FILES)} muestras del curso en {dest}")


def create_symlinks(output_dir: Path) -> None:
    """Crea enlaces simbólicos para nombres alternativos que otros scripts esperan."""
    for alt_name, target_name in ALTERNATIVE_NAMES.items():
        target_path = output_dir / target_name
        alt_path = output_dir / alt_name
        
        if not target_path.exists():
            print(f"  Advertencia: {target_name} no existe, no se crea enlace {alt_name}")
            continue
        
        if alt_path.exists():
            # Si ya existe y es un enlace o archivo, lo removemos
            alt_path.unlink()
        
        alt_path.symlink_to(target_name)
        print(f"  Enlace creado: {alt_name} -> {target_name}")


def download_checkpoints(root: Path) -> None:
    """Descarga los checkpoints y crea enlaces para nombres alternativos."""
    output_dir = root / "checkpoints" / CHECKPOINT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Descargando checkpoints (el de difusion pesa varios GB)...")
    for hf_name, local_name in CHECKPOINT_MAP.items():
        destination = output_dir / local_name
        if destination.exists():
            # Verificar que el archivo no esté vacío
            if destination.stat().st_size > 0:
                print(f"  ya existe, se omite: {local_name} ({destination.stat().st_size / 1e9:.2f} GB)")
                continue
            else:
                print(f"  archivo vacío encontrado, re-descargando: {local_name}")
                destination.unlink()
        
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO_ID, 
            filename=hf_name,
            subfolder=CHECKPOINT_SUBDIR  # ¡Importante! Buscar en la subcarpeta correcta
        )
        shutil.copy2(downloaded, destination)
        size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"  {hf_name} -> {destination} ({size_mb:.1f} MB)")

    # Crear enlaces simbólicos para nombres alternativos
    print("Creando enlaces para nombres alternativos...")
    create_symlinks(output_dir)
    print(f"Listo: checkpoints en {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga datos y, opcionalmente, checkpoints para el laboratorio."
    )
    parser.add_argument("--ideam", action="store_true", help="Descarga ideam_data/ (incluye training_dataset/).")
    parser.add_argument("--piura", action="store_true", help="Descarga piura_data/ (tarea 02).")
    parser.add_argument("--sophy", action="store_true", help="Descarga sophy_data/ (tarea 02).")
    parser.add_argument("--checkpoints", action="store_true", help="Descarga los checkpoints .pth a checkpoints/ef_ideam_final/.")
    parser.add_argument("--all", action="store_true", help="Descarga todos los datos y los checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT

    print(f"Directorio raíz: {root}")
    print("-" * 50)

    # Las 5 muestras del curso siempre se descargan (son el minimo para 01 y 03).
    download_samples(root)

    if args.all or args.ideam:
        download_data_folder(root, DATA_FOLDERS["ideam"])
    if args.all or args.piura:
        download_data_folder(root, DATA_FOLDERS["piura"])
    if args.all or args.sophy:
        download_data_folder(root, DATA_FOLDERS["sophy"])
    if args.all or args.checkpoints:
        download_checkpoints(root)

    print("\n" + "=" * 50)
    print("✅ Descarga completa.")
    print("=" * 50)


if __name__ == "__main__":
    main()