from __future__ import annotations

import argparse
from pathlib import Path
import json
import warnings
from types import SimpleNamespace

import numpy as np

from .config import load_config
from .io import read_sites
from .atlas import fetch_julich_insula_atlas
from .projection import project_probability_atlas_to_fslr, project_sites_to_fslr
from .assignment import assign_sites_to_julich_regions
from .plotting import make_surface_figures, _dv_group


def run(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = config_path.parent

    input_path = Path(config["input"]["csv"])
    if not input_path.is_absolute():
        input_path = root / input_path
    output_dir = Path(config.get("plot", {}).get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sites = read_sites(input_path, config["input"])
    print(f"Loaded {len(sites)} electrode sites from {input_path}")
    print("Coordinate assumption: MNI152 millimetres, RAS+ (x<0 left; x>0 right).")

    atlas_cfg = config.get("atlas", {})
    atlas = fetch_julich_insula_atlas(
        parcellation=atlas_cfg.get("parcellation", "julich 3.1"),
        space=atlas_cfg.get("space", "mni152"),
        region_codes=atlas_cfg.get("region_codes"),
        probability_threshold=float(atlas_cfg.get("probability_threshold", 0.05)),
    )

    surface_cfg = config.get("surface", {})
    density = surface_cfg.get("density", "32k")
    surface_probability_threshold = float(
        atlas_cfg.get("surface_probability_threshold", atlas_cfg.get("probability_threshold", 0.08))
    )
    # The probability-first projection is expensive on the first run (one
    # volume-to-surface transform per region/hemisphere). Cache it separately
    # from the old nearest-neighbour categorical atlas so plot-only reruns are fast.
    surface_cache = output_dir / f"julich_probability_wta_fslr_{density}.npz"
    cache_meta = {
        "projection_method": "linear_probability_wta",
        "parcellation": str(atlas_cfg.get("parcellation", "julich 3.1")),
        "space": str(atlas_cfg.get("space", "mni152")),
        "density": str(density),
        "probability_threshold": surface_probability_threshold,
        "region_names": {str(k): str(v) for k, v in atlas.region_names.items()},
    }
    atlas_surface = None
    if surface_cache.exists():
        try:
            with np.load(surface_cache, allow_pickle=False) as cached:
                cached_meta = json.loads(str(cached["metadata"].item()))
                if cached_meta == cache_meta:
                    atlas_surface = SimpleNamespace(
                        left=cached["left"].copy(), right=cached["right"].copy()
                    )
                    print(f"Loaded cached continuous fsLR atlas: {surface_cache}")
        except Exception as exc:
            print(f"WARNING: could not use surface-atlas cache ({exc}); recomputing.")
    if atlas_surface is None:
        print(f"Projecting continuous Julich probabilities to fsLR {density}...")
        atlas_surface = project_probability_atlas_to_fslr(
            atlas, density=density, probability_threshold=surface_probability_threshold
        )
        np.savez_compressed(
            surface_cache, left=atlas_surface.left.astype(np.int32),
            right=atlas_surface.right.astype(np.int32), metadata=json.dumps(cache_meta),
        )

    radius = float(surface_cfg.get("marker_radius_mm", 5.0))
    fallback_radii = tuple(
        float(value) for value in surface_cfg.get("fallback_marker_radii_mm", [7.5])
    )
    print(
        f"Projecting electrode centres to fsLR (marker radii "
        f"{[radius, *fallback_radii]} mm)..."
    )
    _, mapped_sites = project_sites_to_fslr(
        atlas.image,
        sites,
        density=density,
        radius_mm=radius,
        fallback_radii_mm=fallback_radii,
        atlas_surface=atlas_surface,
        snap_to_insula=bool(surface_cfg.get("snap_to_insula", True)),
        max_snap_distance_mm=float(surface_cfg.get("max_snap_distance_mm", 6.0)),
        last_resort_radii_mm=tuple(float(v) for v in surface_cfg.get("last_resort_marker_radii_mm", [10.0, 15.0, 20.0, 25.0, 30.0])),
        region_names=atlas.region_names,
    )

    # Enrich site output with probabilistic atlas assignment at the original MNI point.
    mapped_sites = assign_sites_to_julich_regions(mapped_sites, atlas)
    area_col = str(config.get("plot", {}).get("area_column", "area"))
    if area_col in mapped_sites.columns:
        raw_area = mapped_sites[area_col].astype(str).str.strip()
        has_code = raw_area.map(lambda value: bool(__import__("re").search(
            r"(?<![a-z0-9])(ig\d+|ia\d+|id\d+)(?!\d)", value.lower())))
        mapped_sites["insula_area_for_dv"] = raw_area.where(has_code, mapped_sites["atlas_region_top"])
    else:
        mapped_sites["insula_area_for_dv"] = mapped_sites["atlas_region_top"]
    derived_dv = mapped_sites["insula_area_for_dv"].map(_dv_group)
    if "manual_dv_group" in mapped_sites:
        manual = mapped_sites["manual_dv_group"].astype(str).str.strip()
        valid_manual = manual.isin(["Dorsale", "Ventrale"])
        mapped_sites["insula_dv_group"] = derived_dv.where(~valid_manual, manual)
    else:
        mapped_sites["insula_dv_group"] = derived_dv
    mapped_csv = output_dir / "sites_surface_coordinates.csv"
    mapped_sites.to_csv(mapped_csv, index=False)

    atlas_npz = output_dir / f"julich_insula_labels_fslr_{density}.npz"
    np.savez_compressed(
        atlas_npz,
        left=atlas_surface.left.astype(np.int16),
        right=atlas_surface.right.astype(np.int16),
        label_names=json.dumps(atlas.region_names),
        label_colors=json.dumps(atlas.region_colors),
    )

    plot_cfg = config.get("plot", {})
    figure_paths = make_surface_figures(
        atlas_surface=atlas_surface,
        mapped_sites=mapped_sites,
        atlas_result=atlas,
        density=density,
        output_dir=output_dir,
        plot_config=plot_cfg,
    )

    print("\nFinished. Files created:")
    for path in figure_paths:
        print(f"  {path}")
    print(f"  {mapped_csv}")
    print(f"  {atlas_npz}")
    print("\nProjection methods:")
    print(mapped_sites["projection_method"].value_counts(dropna=False).to_string())
    no_projection = mapped_sites.loc[mapped_sites["surface_vertex"] < 0, "site_id"].tolist()
    if no_projection:
        print("\nWARNING: the following sites did not project to an fsLR vertex:", ", ".join(no_projection))
        print(
            "These contacts still have no defensible fsLR surface coordinate and cannot be drawn. "
            "Check coordinate convention and MNI152 template variant. All contacts that do "
            "have a surface candidate are now included, even outside the thresholded Julich mask."
        )
    outside_julich = mapped_sites.loc[
        (mapped_sites["surface_vertex"] >= 0) & (~mapped_sites["surface_in_julich_mask"]),
        "site_id",
    ].tolist()
    if outside_julich:
        print(
            "\nINFO: these contacts reached a cortical fsLR vertex outside the thresholded Julich insula mask; "
            "they are INCLUDED as hollow markers to distinguish uncertain/out-of-mask projections: "
            + ", ".join(outside_julich)
        )
    print("\nDorsal/ventral counts from the selected area labels:")
    if "insula_dv_group" in mapped_sites.columns:
        for side in ("left", "right"):
            sub = mapped_sites[mapped_sites["hemisphere"] == side]
            n_d = int((sub["insula_dv_group"] == "Dorsale").sum())
            n_v = int((sub["insula_dv_group"] == "Ventrale").sum())
            n_u = int((sub["insula_dv_group"] == "Non classée").sum())
            print(f"  {side}: Dorsale={n_d}, Ventrale={n_v}, non classée={n_u}")
    n_displayed = int((mapped_sites["surface_vertex"].astype(int) >= 0).sum())
    n_inside_mask = int(mapped_sites["surface_in_julich_mask"].astype(bool).sum())
    print(f"\nVisibility: {n_displayed}/{len(mapped_sites)} contacts have a drawable fsLR surface vertex; "
          f"{n_inside_mask} are on the Julich mask and {n_displayed - n_inside_mask} are outside it (hollow markers).")
    print(
        f"\nAudit files: {output_dir / 'site_counts_dorsal_ventral.csv'}, "
        f"{output_dir / 'overlapping_sites_report.csv'} and {output_dir / 'site_visibility_report.csv'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Plot MNI152 insular sites on a Julich-labelled fsLR inflated surface.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration (default: config.yaml)")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
