from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def read_sites(csv_path: str | Path, input_config: dict) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = {
        "x": input_config.get("x_column", "x"),
        "y": input_config.get("y_column", "y"),
        "z": input_config.get("z_column", "z"),
        "color": input_config.get("color_column", "color"),
    }
    missing = [column for column in cols.values() if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.copy()
    for key in ("x", "y", "z"):
        df[key] = pd.to_numeric(df[cols[key]], errors="coerce")
    bad = df[["x", "y", "z"]].isna().any(axis=1)
    if bad.any():
        rows = (df.index[bad] + 2).tolist()
        raise ValueError(f"Non-numeric or missing XYZ coordinates on CSV rows: {rows}")

    df["color"] = df[cols["color"]].astype(str)
    invalid_colors = []
    for idx, value in df["color"].items():
        # Matplotlib validates the full color syntax later; reject obvious empties.
        if not value.strip() or value.lower() == "nan":
            invalid_colors.append(int(idx) + 2)
    if invalid_colors:
        raise ValueError(f"Missing colors on CSV rows: {invalid_colors}")

    id_col = input_config.get("id_column", "site_id")
    label_col = input_config.get("label_column", "label")
    df["site_id"] = df[id_col].astype(str) if id_col in df else [f"site_{i+1:03d}" for i in range(len(df))]
    df["label"] = df[label_col].astype(str) if label_col in df else df["site_id"]

    # Keep an optional source anatomical area column. The dorsal/ventral
    # classification is based on this field when present, as requested.
    area_col = input_config.get("area_column", "area")
    if area_col and area_col in df.columns:
        df["area"] = df[area_col].astype(str).str.strip()

    # Optional manually curated dorsal/ventral selection. When supplied, this
    # is treated as the anatomical target for the final Julich-surface snap,
    # rather than reclassifying the contact from its potentially imprecise MNI
    # coordinate alone.
    dv_col = input_config.get("dv_group_column", "insula_dv_group")
    if dv_col and dv_col in df.columns:
        manual_dv = df[dv_col].astype(str).str.strip().str.lower()
        dv_aliases = {
            "dorsale": "Dorsale", "dorsal": "Dorsale", "d": "Dorsale",
            "ventrale": "Ventrale", "ventral": "Ventrale", "v": "Ventrale",
        }
        df["manual_dv_group"] = manual_dv.map(dv_aliases).fillna("")
    if df["site_id"].duplicated().any():
        duplicates = df.loc[df["site_id"].duplicated(), "site_id"].tolist()
        raise ValueError(f"site_id values must be unique. Duplicates: {duplicates}")

    df["hemisphere"] = df["x"].map(lambda x: "left" if x < 0 else "right" if x > 0 else "midline")
    if (df["hemisphere"] == "midline").any():
        ids = df.loc[df["hemisphere"] == "midline", "site_id"].tolist()
        raise ValueError(
            "Sites with x=0 cannot be assigned automatically to one hemisphere. "
            f"Please inspect: {ids}"
        )

    # Keep a conservative range check to catch voxel coordinates accidentally
    # supplied as millimetres. This does not validate the MNI152 variant.
    outside = (df[["x", "y", "z"]].abs() > 110).any(axis=1)
    if outside.any():
        ids = df.loc[outside, "site_id"].tolist()
        raise ValueError(
            "Some coordinates are outside the conservative MNI millimetre range "
            f"(+/-110 mm): {ids}. Check units and coordinate convention."
        )

    # Validate user-provided colors with Matplotlib without importing plotting code.
    try:
        from matplotlib.colors import is_color_like
        invalid = [c for c in df["color"].unique() if not is_color_like(c)]
    except Exception:
        invalid = []
    if invalid:
        raise ValueError(f"Invalid Matplotlib color values: {invalid}")

    # Optional outcome/status and scalar columns. Keep t_value intact so that
    # the plotting stage can produce a continuous colour-map + colourbar.
    keep = ["site_id", "label", "x", "y", "z", "color", "hemisphere"]

    status_col = input_config.get("status_column", "status")
    if status_col and status_col in df.columns:
        df["status"] = df[status_col].astype(str).str.strip().str.lower()
        df["status"] = df["status"].replace({
            "both_preger_loss": "both_prefer_loss",
            "both_preger_gain": "both_prefer_gain",
        })
        df.loc[df["status"].isin(["", "nan", "none"]), "status"] = "unknown"
        keep.append("status")

    tvalue_col = input_config.get("tvalue_column", "t_value")
    if tvalue_col and tvalue_col in df.columns:
        df["t_value"] = pd.to_numeric(df[tvalue_col], errors="coerce")
        if df["t_value"].isna().any():
            bad_rows = (df.index[df["t_value"].isna()] + 2).tolist()
            raise ValueError(f"Non-numeric or missing t_value values on CSV rows: {bad_rows}")
        keep.append("t_value")

    if "area" in df.columns:
        keep.append("area")
    if "manual_dv_group" in df.columns:
        keep.append("manual_dv_group")
    return df[keep]
