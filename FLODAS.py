#!/usr/bin/env python3
"""
FLODAS UNet full-scene inference on Planet SuperDove 8-band GeoTIFFs.

Expected folder layout:
    FLODAS.py
    model_unet.py
    config.yaml
    *.pth  (trained model weights normally FLODAS_UNET_CC-BY-NC-4.0 from Zenodo record. One .pth file unless otherwise specified in config.yaml)

Basic usage:
    python FLODAS.py path/to/scene.tif

Outputs are written in same folder as the input raster.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

# Helps avoid OpenMP duplicate runtime errors on some Windows/PyTorch setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from rasterio.windows import Window
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union
from skimage.measure import label
from skimage.morphology import remove_small_holes

from model_unet import UNet


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"

# -----------------------------------------------------------------------------
# Fixed preprocessing used during model training.
# Do not change these unless the model was trained with different normalization.
# Band order: 443, 490, 531, 565, 610, 665, 705, 865 nm.
# -----------------------------------------------------------------------------
MEAN_VALS = np.array(
    [562.88, 576.28, 570.76, 558.09, 528.92, 530.12, 538.71, 708.19],
    dtype=np.float32,
)
STD_VALS = np.array(
    [780.07, 778.45, 791.35, 794.22, 827.83, 839.33, 872.28, 1062.16],
    dtype=np.float32,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "weights_filename": "",
    },
    "tiling": {
        "tile_size": 256,
        "stride": 128,
        "nodata_tile_frac": 0.9,
    },
    "postprocessing": {
        "threshold": 0.90,
        "min_object_size": 50,
        "edge_buffer_px": 50,
        "apply_gaussian": True,
        "gauss_sigma": 1.0,
        "apply_morph_smoothing": True,
        "morph_size": 3,
        "fill_holes": True,
        "min_hole_px": 768,
    },
    "isobands": {
        "enabled": True,
        "min_prob": 0.10,
        "step": 0.10,
    },
    "outputs": {
        "write_probability": True,
        "write_binary_mask": True,
        "write_vectors": True,
        "write_excluded": True,
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FLODAS full-scene UNet inference on an 8-band Planet SuperDove GeoTIFF."
    )
    parser.add_argument(
        "raster",
        type=Path,
        help="Input 8-band GeoTIFF. Outputs are written to this raster's folder.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        type=Path,
        help="Optional override. Defaults to config.yaml in the same folder as FLODAS.py.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional output tag. Defaults to current timestamp, e.g. 20260518-1215.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when CUDA is available.",
    )
    return parser.parse_args()


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a nested dictionary."""
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Place config.yaml in the same folder as FLODAS.py, or pass --config path/to/config.yaml."
        )

    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "YAML config files require PyYAML. Install it with: pip install pyyaml"
                ) from exc
            user_config = yaml.safe_load(f) or {}
        elif suffix == ".json":
            user_config = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")

    return deep_update(DEFAULT_CONFIG, user_config)


def resolve_weights_path(cfg: dict[str, Any]) -> Path:
    """Find model weights in the same folder as FLODAS.py."""
    weights_filename = str(cfg.get("model", {}).get("weights_filename", "")).strip()

    if weights_filename:
        weights_path = SCRIPT_DIR / weights_filename
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Configured model weights not found: {weights_path}\n"
                "The weights file must be in the same folder as FLODAS.py."
            )
        return weights_path

    candidates = sorted(SCRIPT_DIR.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No .pth model weights found in {SCRIPT_DIR}.\n"
            "Place the trained weights file in the same folder as FLODAS.py."
        )
    if len(candidates) > 1:
        names = "\n".join(f"  - {p.name}" for p in candidates)
        raise RuntimeError(
            "More than one .pth file was found in the FLODAS folder:\n"
            f"{names}\n"
            "Set model.weights_filename in config.yaml to choose the correct file."
        )
    return candidates[0]

def clean_model_stem(weights_path: Path) -> str:
    stem = weights_path.stem
    return stem.replace("unet_", "")


def load_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    model = UNet(n_channels=8, n_classes=1)

    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        checkpoint = torch.load(weights_path, map_location=device)

    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def validate_raster(src: rasterio.DatasetReader) -> None:
    if src.count != 8:
        raise ValueError(
            f"Expected an 8-band Planet SuperDove raster, but input has {src.count} bands."
        )
    if src.crs is None:
        print("[WARN] Input raster has no CRS. Raster outputs will still be written, but vectors may be incomplete.")


def create_weight_mask(tile_size: int) -> np.ndarray:
    center = tile_size // 2
    y, x = np.ogrid[:tile_size, :tile_size]
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    max_dist = np.sqrt(2 * (center**2))
    weights = 1 - (dist / max_dist)
    return np.clip(weights, 0, 1).astype(np.float32)


def build_valid_mask_all_bands(src: rasterio.DatasetReader) -> np.ndarray:
    data = src.read(masked=True)
    mask = np.ma.getmaskarray(data)
    return ~np.any(mask, axis=0)


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return tile start positions, including the final edge tile."""
    if length < tile_size:
        raise ValueError(
            f"Raster dimension {length} is smaller than tile_size {tile_size}. "
            "Use a smaller tile_size or pad the input raster."
        )

    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def distance_border_exclusion(valid_mask: np.ndarray, edge_buffer_px: int) -> np.ndarray:
    if edge_buffer_px <= 0:
        return np.zeros_like(valid_mask, dtype=bool)

    dist_in = distance_transform_edt(valid_mask)
    dist_out = distance_transform_edt(~valid_mask)
    inside = valid_mask & (dist_in < edge_buffer_px)
    outside = (~valid_mask) & (dist_out < edge_buffer_px)
    return inside | outside

def fuse_soft_probs(
    src: rasterio.DatasetReader,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int,
    stride: int,
    nodata_tile_frac: float,
) -> np.ndarray:
    height, width = src.height, src.width
    pred_sum = np.zeros((height, width), dtype=np.float32)
    pred_count = np.zeros((height, width), dtype=np.float32)
    weight_mask = create_weight_mask(tile_size)

    y_starts = tile_starts(height, tile_size, stride)
    x_starts = tile_starts(width, tile_size, stride)

    print(f"[INFO] Running tiled inference: {len(y_starts) * len(x_starts)} tiles")

    mean = MEAN_VALS[:, None, None]
    std = STD_VALS[:, None, None]

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                window = Window(x, y, tile_size, tile_size)
                tile = src.read(window=window, masked=True).astype(np.float32)

                tile_mask = np.ma.getmaskarray(tile)
                tile_valid = ~np.any(tile_mask, axis=0)
                if (1.0 - tile_valid.mean()) > nodata_tile_frac:
                    continue

                tile_filled = tile.filled(0)
                tile_norm = (tile_filled - mean) / std

                tensor = torch.from_numpy(tile_norm).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)

                pred_sum[y : y + tile_size, x : x + tile_size] += probs * weight_mask
                pred_count[y : y + tile_size, x : x + tile_size] += weight_mask

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_prob = np.true_divide(pred_sum, pred_count)
        avg_prob[pred_count == 0] = 0

    return avg_prob


def apply_postprocessing(
    avg_prob: np.ndarray,
    valid_mask: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    post = cfg["postprocessing"]

    threshold = float(post["threshold"])
    min_object_size = int(post["min_object_size"])
    edge_buffer_px = int(post["edge_buffer_px"])

    prob_for_threshold = avg_prob.copy()

    if bool(post["apply_gaussian"]):
        sigma = float(post["gauss_sigma"])
        prob_for_threshold = gaussian_filter(prob_for_threshold, sigma=sigma)
        print(f"[OK] Applied Gaussian smoothing before thresholding: sigma={sigma}")

    binary = (prob_for_threshold >= threshold).astype(np.uint8)
    print(f"[INFO] Threshold = {threshold}")

    excluded_small = np.zeros_like(binary, dtype=bool)
    if min_object_size > 0:
        labeled = label(binary, connectivity=1)
        areas = np.bincount(labeled.ravel())
        label_ids = np.arange(len(areas))
        small_labels = label_ids[(areas < min_object_size) & (label_ids != 0)]
        excluded_small = np.isin(labeled, small_labels)
        binary[excluded_small] = 0
        print(f"[OK] Removed objects smaller than {min_object_size} px")

    excluded_border = distance_border_exclusion(valid_mask, edge_buffer_px)
    binary[excluded_border] = 0
    if edge_buffer_px > 0:
        print(f"[OK] Applied edge buffer exclusion: {edge_buffer_px} px")

    if bool(post["apply_morph_smoothing"]):
        morph_size = int(post["morph_size"])
        if morph_size > 1:
            structure = np.ones((morph_size, morph_size), dtype=np.uint8)
            binary = binary_closing(binary, structure=structure).astype(np.uint8)
            print(f"[OK] Applied morphological closing: {morph_size}x{morph_size}")

    if bool(post["fill_holes"]):
        min_hole_px = int(post["min_hole_px"])
        if min_hole_px > 0:
            before = int(binary.sum())
            binary = remove_small_holes(
                binary.astype(bool), area_threshold=min_hole_px
            ).astype(np.uint8)
            after = int(binary.sum())
            print(f"[OK] Filled holes smaller than {min_hole_px} px; added {after - before} foreground px")

    binary[~valid_mask] = 0
    excluded_total = excluded_small | excluded_border
    return binary, excluded_total

def export_excluded(mask_bool: np.ndarray, reason: str, transform, crs) -> gpd.GeoDataFrame | None:
    mask_u8 = mask_bool.astype(np.uint8)
    shape_iter = shapes(mask_u8, mask=mask_u8 == 1, transform=transform)
    polygons = [shape(geom) for geom, value in shape_iter if value == 1 and shape(geom).is_valid]

    if not polygons:
        return None

    return gpd.GeoDataFrame(
        {"class": [0], "reason": [reason], "geometry": [MultiPolygon(polygons)]},
        crs=crs,
    )


def vectorize_mask(binary_mask: np.ndarray, transform, crs) -> gpd.GeoDataFrame | None:
    mask_u8 = binary_mask.astype(np.uint8)
    shape_iter = shapes(mask_u8, mask=mask_u8 == 1, transform=transform)
    polygons = [shape(geom) for geom, value in shape_iter if value == 1 and shape(geom).is_valid]

    if not polygons:
        return None

    return gpd.GeoDataFrame(
        {"class": [1], "geometry": [MultiPolygon(polygons)]},
        crs=crs,
    )


def band_thresholds(min_prob: float, step: float) -> list[float]:
    thresholds = [1.0]
    value = 1.0
    while value - step >= min_prob - 1e-9:
        value = max(min_prob, value - step)
        thresholds.append(value)
    return thresholds


def vectorize_isobands(
    prob: np.ndarray,
    valid_mask: np.ndarray,
    transform,
    crs,
    min_prob: float,
    step: float,
) -> gpd.GeoDataFrame:
    thresholds = band_thresholds(min_prob, step)
    cumulative = [(prob >= threshold) & valid_mask for threshold in thresholds]

    rows = []
    for i in range(1, len(thresholds)):
        high = thresholds[i - 1]
        low = thresholds[i]
        ring = cumulative[i] & (~cumulative[i - 1])
        ring_u8 = ring.astype(np.uint8)

        shape_iter = shapes(ring_u8, mask=ring_u8 == 1, transform=transform)
        geometries = [shape(geom) for geom, value in shape_iter if value == 1]
        geometries = [geom for geom in geometries if geom.is_valid and not geom.is_empty]

        if not geometries:
            continue

        merged = unary_union(geometries)
        if isinstance(merged, Polygon):
            merged = MultiPolygon([merged])

        rows.append(
            {
                "p_min": int(round(low * 100)),
                "p_max": int(round(high * 100)),
                "geometry": merged,
            }
        )

    if not rows:
        return gpd.GeoDataFrame(columns=["p_min", "p_max", "geometry"], crs=crs)

    return gpd.GeoDataFrame(rows, crs=crs).sort_values(
        by=["p_min", "p_max"], ascending=[False, False], ignore_index=True
    )


def save_probability_raster(
    output_path: Path,
    profile: dict[str, Any],
    avg_prob: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    prob_profile = profile.copy()
    prob_profile.update({"count": 1, "dtype": rasterio.float32, "nodata": np.nan})

    avg_prob_out = avg_prob.copy()
    avg_prob_out[~valid_mask] = np.nan

    with rasterio.open(output_path, "w", **prob_profile) as dst:
        dst.write(avg_prob_out.astype(np.float32), 1)
        dst.write_mask(valid_mask.astype(np.uint8) * 255)

    print(f"[OK] Soft probability raster saved: {output_path}")


def save_binary_raster(output_path: Path, profile: dict[str, Any], binary: np.ndarray) -> None:
    mask_profile = profile.copy()
    mask_profile.update({"count": 1, "dtype": rasterio.uint8, "nodata": 0})

    with rasterio.open(output_path, "w", **mask_profile) as dst:
        dst.write(binary.astype(np.uint8), 1)

    print(f"[OK] Final binary mask saved: {output_path}")

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    raster_path = args.raster
    if not raster_path.exists():
        raise FileNotFoundError(f"Input raster not found: {raster_path}")

    weights_path = resolve_weights_path(cfg)
    output_dir = raster_path.parent
    run_tag = args.run_tag or time.strftime("%Y%m%d-%H%M")
    model_stem = clean_model_stem(weights_path)
    raster_stem = raster_path.stem

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[INFO] FLODAS folder: {SCRIPT_DIR}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Weights: {weights_path}")
    print(f"[INFO] Output folder: {output_dir}")
    print(f"[INFO] Device: {device}")

    model = load_model(weights_path, device)

    tiling = cfg["tiling"]
    tile_size = int(tiling["tile_size"])
    stride = int(tiling["stride"])
    nodata_tile_frac = float(tiling["nodata_tile_frac"])

    with rasterio.open(raster_path) as src:
        validate_raster(src)
        profile = src.profile
        transform = src.transform
        crs = src.crs

        valid_mask = build_valid_mask_all_bands(src)
        avg_prob = fuse_soft_probs(
            src=src,
            model=model,
            device=device,
            tile_size=tile_size,
            stride=stride,
            nodata_tile_frac=nodata_tile_frac,
        )

    outputs = cfg["outputs"]
    post = cfg["postprocessing"]
    threshold = float(post["threshold"])
    threshold_suffix = f"_thr{int(round(threshold * 100))}"
    prefix = f"{raster_stem}_pred-{model_stem}_{run_tag}"

    if bool(outputs["write_probability"]):
        prob_tif = output_dir / f"{prefix}_prob.tif"
        save_probability_raster(prob_tif, profile, avg_prob, valid_mask)

    binary, excluded_total = apply_postprocessing(avg_prob, valid_mask, cfg)

    if bool(outputs["write_binary_mask"]):
        mask_tif = output_dir / f"{prefix}{threshold_suffix}.tif"
        save_binary_raster(mask_tif, profile, binary)

    if bool(outputs["write_vectors"]):
        gdf_pos = vectorize_mask(binary, transform, crs)
        if gdf_pos is not None and not gdf_pos.empty:
            shp_path = output_dir / f"{prefix}{threshold_suffix}.shp"
            gdf_pos.to_file(shp_path)
            print(f"[OK] Vector polygons saved: {shp_path}")
        else:
            print("[INFO] No positive features to vectorize.")

    if bool(outputs["write_excluded"]):
        gdf_excluded = export_excluded(excluded_total, "excluded_area", transform, crs)
        if gdf_excluded is not None and not gdf_excluded.empty:
            excluded_path = output_dir / f"{raster_stem}_excluded-{model_stem}_{run_tag}{threshold_suffix}.shp"
            gdf_excluded.to_file(excluded_path)
            print(f"[OK] Excluded regions saved: {excluded_path}")
        else:
            print("[INFO] No excluded regions to save.")

    isobands = cfg["isobands"]
    if bool(isobands["enabled"]):
        print("[INFO] Building probability isobands from unsmoothed probabilities.")
        gdf_bands = vectorize_isobands(
            prob=avg_prob,
            valid_mask=valid_mask,
            transform=transform,
            crs=crs,
            min_prob=float(isobands["min_prob"]),
            step=float(isobands["step"]),
        )

        if not gdf_bands.empty:
            min_prob_pct = int(round(float(isobands["min_prob"]) * 100))
            iso_path = output_dir / f"{prefix}_isobands_{min_prob_pct}-100.shp"
            gdf_bands.to_file(iso_path)
            print(f"[OK] Isobands saved: {iso_path}")
        else:
            print("[INFO] No isobands to save.")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
