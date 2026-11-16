from __future__ import annotations

from dataclasses import dataclass
import re

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine
from scipy.spatial import cKDTree


@dataclass
class SurfaceProjection:
    left: np.ndarray
    right: np.ndarray


def _map_mni_to_fslr(image, density: str = "32k", method: str = "nearest") -> SurfaceProjection:
    try:
        from neuromaps.transforms import mni152_to_fslr
    except ImportError as exc:
        raise RuntimeError("neuromaps is required for MNI152 -> fsLR registration.") from exc
    projected = mni152_to_fslr(image, fslr_density=density, method=method)
    left_img, right_img = projected
    return SurfaceProjection(
        left=np.asarray(left_img.agg_data()).squeeze(),
        right=np.asarray(right_img.agg_data()).squeeze(),
    )


def project_atlas_to_fslr(atlas_img, density: str = "32k") -> SurfaceProjection:
    """Project a categorical MNI152 parcellation to fsLR using nearest labels."""
    return _map_mni_to_fslr(atlas_img, density=density, method="nearest")


def project_probability_atlas_to_fslr(
    atlas_result, density: str = "32k", probability_threshold: float = 0.08,
) -> SurfaceProjection:
    """Project each Julich probability map continuously to fsLR, then label vertices.

    Unlike projecting a hard, voxelwise winner-takes-all NIfTI with nearest-neighbour
    interpolation, this samples the underlying regional probability maps with linear
    interpolation and only then chooses the strongest region at each surface vertex.
    This reduces voxel-grid stair-stepping at parcel borders. `probability_threshold`
    controls how much of the low-probability fringe is retained.
    """
    labels = {"left": None, "right": None}
    best = {"left": None, "right": None}
    missing = []

    for label_id, region_name in sorted(atlas_result.region_names.items(), key=lambda item: int(item[0])):
        match = re.search(r"\b(ig\d+|ia\d+|id\d+)\s*\((left|right)\)", str(region_name), flags=re.I)
        if not match:
            missing.append(str(region_name))
            continue
        code, side = match.group(1).capitalize(), match.group(2).lower()
        pimg = atlas_result.probability_images.get(f"{code}_{side}")
        if pimg is None:
            missing.append(str(region_name))
            continue
        projected = _map_mni_to_fslr(pimg, density=density, method="linear")
        values = np.asarray(getattr(projected, side), dtype=np.float32).squeeze()
        if labels[side] is None:
            labels[side] = np.zeros(values.shape, dtype=np.int32)
            best[side] = np.zeros(values.shape, dtype=np.float32)
        if values.shape != labels[side].shape:
            raise RuntimeError(f"Inconsistent fsLR probability-map shape for {region_name}: {values.shape}")
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        choose = (values > best[side]) & (values >= float(probability_threshold))
        labels[side][choose] = int(label_id)
        best[side][choose] = values[choose]

    for side in ("left", "right"):
        if labels[side] is None:
            raise RuntimeError(f"No Julich probability maps were projected to fsLR {side}.")
    if missing:
        print("WARNING: some Julich maps could not be projected: " + ", ".join(missing))
    print(
        "Surface atlas: continuous probability interpolation; "
        f"probability threshold={float(probability_threshold):.3f}."
    )
    return SurfaceProjection(left=labels["left"], right=labels["right"])


def _make_marker_volume(reference_img, sites: pd.DataFrame, radius_mm: float) -> nib.Nifti1Image:
    """Create a labelled sphere around each contact in the reference MNI grid."""
    shape = reference_img.shape[:3]
    inv_affine = np.linalg.inv(reference_img.affine)
    affine = reference_img.affine
    marker_data = np.zeros(shape, dtype=np.uint16)
    best_distance = np.full(shape, np.inf, dtype=np.float32)

    voxel_sizes = nib.affines.voxel_sizes(affine)
    max_vox = np.ceil(radius_mm / np.maximum(voxel_sizes, 1e-6)).astype(int) + 1

    for marker_id, row in enumerate(sites.itertuples(index=False), start=1):
        xyz = np.array([row.x, row.y, row.z], dtype=float)
        ijk = apply_affine(inv_affine, xyz)
        center = np.rint(ijk).astype(int)
        lo = np.maximum(center - max_vox, 0)
        hi = np.minimum(center + max_vox + 1, np.asarray(shape))
        if np.any(lo >= hi):
            raise ValueError(
                f"Site {row.site_id} lies outside the atlas reference volume: {xyz.tolist()}"
            )

        grid = np.indices(tuple(hi - lo), dtype=np.float32)
        grid[0] += lo[0]
        grid[1] += lo[1]
        grid[2] += lo[2]
        vox = np.moveaxis(grid, 0, -1).reshape(-1, 3)
        world = apply_affine(affine, vox)
        dist = np.linalg.norm(world - xyz[None, :], axis=1).reshape(tuple(hi - lo))
        local = dist <= radius_mm

        current_best = best_distance[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        current_labels = marker_data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        update = local & (dist < current_best)
        current_best[update] = dist[update]
        current_labels[update] = marker_id

    header = reference_img.header.copy()
    header.set_data_dtype(np.uint16)
    marker_img = nib.Nifti1Image(marker_data, affine, header=header)
    marker_img.set_qform(affine, code=1)
    marker_img.set_sform(affine, code=1)
    return marker_img


def _load_inflated_vertices(density: str):
    from neuromaps.datasets import fetch_atlas

    surfaces = fetch_atlas("fsLR", density)
    inflated = surfaces["inflated"]
    left_path = inflated.L if hasattr(inflated, "L") else inflated[0]
    right_path = inflated.R if hasattr(inflated, "R") else inflated[1]
    left = nib.load(left_path)
    right = nib.load(right_path)
    return {
        "left": np.asarray(left.agg_data("NIFTI_INTENT_POINTSET"), dtype=float),
        "right": np.asarray(right.agg_data("NIFTI_INTENT_POINTSET"), dtype=float),
    }


def _medoid_vertex(vertex_ids: np.ndarray, vertices: np.ndarray) -> int:
    """Return the projected vertex closest to the candidate-set centroid."""
    coords = vertices[vertex_ids]
    center = coords.mean(axis=0)
    return int(vertex_ids[np.argmin(np.sum((coords - center) ** 2, axis=1))])


def project_sites_to_fslr(
    reference_img,
    sites: pd.DataFrame,
    density: str = "32k",
    radius_mm: float = 5.0,
    fallback_radii_mm: tuple[float, ...] | list[float] = (7.5,),
    atlas_surface: SurfaceProjection | None = None,
    snap_to_insula: bool = True,
    max_snap_distance_mm: float = 6.0,
    last_resort_radii_mm: tuple[float, ...] | list[float] = (10.0, 15.0, 20.0, 25.0, 30.0),
    region_names: dict[int, str] | None = None,
) -> tuple[SurfaceProjection | None, pd.DataFrame]:
    """
    Project contact neighbourhoods to fsLR.

    Each site is first projected with radius_mm. Sites with no surface vertex are
    retried at the explicitly listed fallback radii. If a projected point lands
    just outside the Julich insula mask, it can optionally be snapped to the
    nearest labelled insula vertex, provided the distance on the inflated mesh
    is <= max_snap_distance_mm. Original MNI coordinates and all projection
    diagnostics are retained in the returned dataframe.
    """
    mapped = sites.copy().reset_index(drop=True)
    n_sites = len(mapped)
    mapped["marker_id"] = np.arange(1, n_sites + 1, dtype=int)
    mapped["surface_hemi"] = mapped["hemisphere"]
    mapped["surface_vertex"] = -1
    mapped["surface_x"] = np.nan
    mapped["surface_y"] = np.nan
    mapped["surface_z"] = np.nan
    mapped["projected_vertex_count"] = 0
    mapped["projection_radius_mm"] = np.nan
    mapped["surface_in_julich_mask"] = False
    mapped["surface_snap_distance_mm"] = np.nan
    mapped["projection_method"] = "not_projected"

    vertices_by_hemi = _load_inflated_vertices(density)
    marker_by_hemi = {"left": None, "right": None}

    if atlas_surface is not None:
        atlas_values = {
            "left": np.rint(atlas_surface.left).astype(np.int64),
            "right": np.rint(atlas_surface.right).astype(np.int64),
        }
        insula_trees = {}
        insula_vertex_ids = {}
        for side in ("left", "right"):
            ids = np.flatnonzero(atlas_values[side] > 0)
            insula_vertex_ids[side] = ids
            insula_trees[side] = (
                cKDTree(vertices_by_hemi[side][ids]) if len(ids) else None
            )
    else:
        atlas_values = None
        insula_trees = {"left": None, "right": None}
        insula_vertex_ids = {"left": np.array([], dtype=int), "right": np.array([], dtype=int)}

    radii = [float(radius_mm)] + [float(r) for r in fallback_radii_mm]
    radii = list(dict.fromkeys(r for r in radii if r > 0))
    pending = list(range(n_sites))
    last_raw_candidate = {}

    for attempt_radius in radii:
        if not pending:
            break
        pending_df = mapped.iloc[pending][["site_id", "label", "x", "y", "z", "color", "hemisphere"]].copy()
        pending_df = pending_df.reset_index(drop=True)
        marker_img = _make_marker_volume(reference_img, pending_df, radius_mm=attempt_radius)
        marker_surface = _map_mni_to_fslr(marker_img, density=density, method="nearest")
        surface_values = {
            "left": np.rint(marker_surface.left).astype(np.int64),
            "right": np.rint(marker_surface.right).astype(np.int64),
        }

        still_pending = []
        for local_idx, original_idx in enumerate(pending):
            side = str(mapped.at[original_idx, "hemisphere"])
            marker_id = local_idx + 1
            candidate_ids = np.flatnonzero(surface_values[side] == marker_id)
            if len(candidate_ids) == 0:
                still_pending.append(original_idx)
                continue

            previous_raw = last_raw_candidate.get(original_idx)
            if previous_raw is None or attempt_radius < previous_raw[1]:
                last_raw_candidate[original_idx] = (candidate_ids, attempt_radius)
            mapped.at[original_idx, "projected_vertex_count"] = int(len(candidate_ids))
            mapped.at[original_idx, "projection_radius_mm"] = attempt_radius

            chosen_ids = candidate_ids
            in_mask = False
            if atlas_values is not None:
                in_mask_ids = candidate_ids[atlas_values[side][candidate_ids] > 0]
                if len(in_mask_ids):
                    chosen_ids = in_mask_ids
                    in_mask = True
                else:
                    # Do not immediately accept a neighbouring non-insular vertex:
                    # a larger sphere may overlap the actual insula ribbon.
                    still_pending.append(original_idx)
                    continue

            vertex_id = _medoid_vertex(chosen_ids, vertices_by_hemi[side])
            xyz_surface = vertices_by_hemi[side][vertex_id]
            mapped.at[original_idx, "surface_vertex"] = vertex_id
            mapped.at[original_idx, "surface_x"] = float(xyz_surface[0])
            mapped.at[original_idx, "surface_y"] = float(xyz_surface[1])
            mapped.at[original_idx, "surface_z"] = float(xyz_surface[2])
            mapped.at[original_idx, "surface_in_julich_mask"] = bool(in_mask)
            mapped.at[original_idx, "projection_method"] = (
                f"mni_sphere_{attempt_radius:g}mm"
            )

        pending = still_pending

    # Final individual retries avoid label competition between neighbouring
    # contacts in a shared marker volume. They are used only for sites that had
    # no candidate in the batched transforms. Large radii are explicitly recorded.
    still_unprojected = list(pending)
    single_radii = [float(r) for r in last_resort_radii_mm if float(r) > 0]
    for original_idx in still_unprojected:
        # Retain the widest batched candidate as a fallback if an isolated retry
        # does not improve it.
        best_raw = last_raw_candidate.get(original_idx)
        side = str(mapped.at[original_idx, "hemisphere"])
        one_site = mapped.iloc[[original_idx]][
            ["site_id", "label", "x", "y", "z", "color", "hemisphere"]
        ].copy().reset_index(drop=True)
        for attempt_radius in single_radii:
            marker_img = _make_marker_volume(reference_img, one_site, radius_mm=attempt_radius)
            marker_surface = _map_mni_to_fslr(marker_img, density=density, method="nearest")
            surface_values = np.rint(getattr(marker_surface, side)).astype(np.int64)
            candidate_ids = np.flatnonzero(surface_values == 1)
            if len(candidate_ids) == 0:
                continue
            if best_raw is None or attempt_radius < best_raw[1]:
                best_raw = (candidate_ids, attempt_radius)
            in_mask_ids = (
                candidate_ids[atlas_values[side][candidate_ids] > 0]
                if atlas_values is not None else np.array([], dtype=int)
            )
            if len(in_mask_ids):
                vertex_id = _medoid_vertex(in_mask_ids, vertices_by_hemi[side])
                xyz_surface = vertices_by_hemi[side][vertex_id]
                mapped.at[original_idx, "surface_vertex"] = vertex_id
                mapped.at[original_idx, "surface_x"] = float(xyz_surface[0])
                mapped.at[original_idx, "surface_y"] = float(xyz_surface[1])
                mapped.at[original_idx, "surface_z"] = float(xyz_surface[2])
                mapped.at[original_idx, "surface_in_julich_mask"] = True
                mapped.at[original_idx, "projected_vertex_count"] = int(len(candidate_ids))
                mapped.at[original_idx, "projection_radius_mm"] = attempt_radius
                mapped.at[original_idx, "projection_method"] = f"single_site_mni_sphere_{attempt_radius:g}mm"
                break
        else:
            if best_raw is not None:
                candidate_ids, attempt_radius = best_raw
                vertex_id = _medoid_vertex(candidate_ids, vertices_by_hemi[side])
                xyz_surface = vertices_by_hemi[side][vertex_id]
                mapped.at[original_idx, "surface_vertex"] = vertex_id
                mapped.at[original_idx, "surface_x"] = float(xyz_surface[0])
                mapped.at[original_idx, "surface_y"] = float(xyz_surface[1])
                mapped.at[original_idx, "surface_z"] = float(xyz_surface[2])
                mapped.at[original_idx, "surface_in_julich_mask"] = False
                mapped.at[original_idx, "projected_vertex_count"] = int(len(candidate_ids))
                mapped.at[original_idx, "projection_radius_mm"] = attempt_radius
                mapped.at[original_idx, "projection_method"] = f"single_site_surface_candidate_outside_julich_{attempt_radius:g}mm"
                last_raw_candidate[original_idx] = (candidate_ids, attempt_radius)

    # Final anatomical assignment: snap every successfully projected contact to
    # the nearest Julich surface vertex in its manually curated dorsal/ventral
    # group. There is deliberately no 6-mm cutoff: the user has already curated
    # these contacts as insular. The snap distance is retained for QC.
    if atlas_surface is not None:
        atlas_values = {
            "left": np.rint(atlas_surface.left).astype(np.int64),
            "right": np.rint(atlas_surface.right).astype(np.int64),
        }
        dorsal_codes = {"id4", "id6", "id7"}
        ventral_codes = {"ia2", "ia3", "id8", "id9", "id10"}

        def _region_code_from_name(value):
            match = re.search(r"(?<![a-z0-9])(ig\d+|ia\d+|id\d+)(?!\d)", str(value).lower())
            return match.group(1) if match else ""

        group_ids = {"left": {"Dorsale": set(), "Ventrale": set()},
                     "right": {"Dorsale": set(), "Ventrale": set()}}
        if region_names:
            for label_id, name in region_names.items():
                match = re.search(r"\b(left|right)\b", str(name), flags=re.I)
                if not match:
                    continue
                side = match.group(1).lower()
                code = _region_code_from_name(name)
                if code in dorsal_codes:
                    group_ids[side]["Dorsale"].add(int(label_id))
                elif code in ventral_codes:
                    group_ids[side]["Ventrale"].add(int(label_id))

        group_vertices = {side: {} for side in ("left", "right")}
        group_trees = {side: {} for side in ("left", "right")}
        for side in ("left", "right"):
            labels_side = atlas_values[side]
            for group in ("Dorsale", "Ventrale"):
                ids = np.flatnonzero(np.isin(labels_side, list(group_ids[side][group])))
                group_vertices[side][group] = ids
                group_trees[side][group] = cKDTree(vertices_by_hemi[side][ids]) if len(ids) else None
            all_ids = np.flatnonzero(labels_side > 0)
            group_vertices[side]["__all__"] = all_ids
            group_trees[side]["__all__"] = cKDTree(vertices_by_hemi[side][all_ids]) if len(all_ids) else None

        snapped_ids, snapped_groups, snapped_regions = [], [], []
        for original_idx in range(n_sites):
            side = str(mapped.at[original_idx, "hemisphere"])
            vertex_id = int(mapped.at[original_idx, "surface_vertex"])
            if vertex_id < 0 or vertex_id >= len(vertices_by_hemi[side]):
                snapped_ids.append(-1); snapped_groups.append(""); snapped_regions.append("")
                continue

            manual_group = str(mapped.at[original_idx, "manual_dv_group"]) if "manual_dv_group" in mapped else ""
            target_group = manual_group if manual_group in {"Dorsale", "Ventrale"} else ""
            raw_vertex = vertex_id
            raw_xyz = vertices_by_hemi[side][raw_vertex]
            raw_label = int(atlas_values[side][raw_vertex])
            raw_name = str(region_names.get(raw_label, "")) if region_names else ""
            raw_code = _region_code_from_name(raw_name)
            if not target_group:
                if raw_code in dorsal_codes:
                    target_group = "Dorsale"
                elif raw_code in ventral_codes:
                    target_group = "Ventrale"

            tree = group_trees[side].get(target_group) if target_group else None
            target_ids = group_vertices[side].get(target_group, np.array([], dtype=int)) if target_group else np.array([], dtype=int)
            if tree is None or len(target_ids) == 0:
                tree = group_trees[side]["__all__"]
                target_ids = group_vertices[side]["__all__"]

            if tree is None or len(target_ids) == 0:
                snapped_ids.append(raw_vertex); snapped_groups.append(target_group); snapped_regions.append(raw_name)
                continue

            # Preserve the exact projected location if it already belongs to
            # the requested group; otherwise snap to the nearest vertex in that
            # group's Julich surface, even when the Euclidean surface distance
            # exceeds the historical 6-mm threshold.
            in_target = raw_label in group_ids[side].get(target_group, set()) if target_group else (raw_label > 0)
            if in_target:
                final_vertex = raw_vertex
                snap_distance = 0.0
            else:
                snap_distance, nearest_local = tree.query(raw_xyz, k=1)
                final_vertex = int(target_ids[int(nearest_local)])
                final_xyz = vertices_by_hemi[side][final_vertex]
                mapped.at[original_idx, "surface_snap_distance_mm"] = float(snap_distance)
                mapped.at[original_idx, "projection_method"] = (
                    f"nearest_julich_{target_group.lower()}_surface" if target_group
                    else "nearest_julich_insula_surface"
                )
                mapped.at[original_idx, "projection_radius_mm"] = mapped.at[original_idx, "projection_radius_mm"]
                mapped.at[original_idx, "projected_vertex_count"] = max(
                    int(mapped.at[original_idx, "projected_vertex_count"]), 1)
                mapped.at[original_idx, "surface_x"] = float(final_xyz[0])
                mapped.at[original_idx, "surface_y"] = float(final_xyz[1])
                mapped.at[original_idx, "surface_z"] = float(final_xyz[2])

            final_xyz = vertices_by_hemi[side][final_vertex]
            final_label = int(atlas_values[side][final_vertex])
            final_name = str(region_names.get(final_label, "")) if region_names else ""
            final_code = _region_code_from_name(final_name)
            final_group = ("Dorsale" if final_code in dorsal_codes else
                           "Ventrale" if final_code in ventral_codes else target_group)
            mapped.at[original_idx, "surface_vertex"] = final_vertex
            mapped.at[original_idx, "surface_x"] = float(final_xyz[0])
            mapped.at[original_idx, "surface_y"] = float(final_xyz[1])
            mapped.at[original_idx, "surface_z"] = float(final_xyz[2])
            mapped.at[original_idx, "surface_in_julich_mask"] = bool(final_label > 0)
            mapped.at[original_idx, "atlas_region_on_surface"] = final_name
            mapped.at[original_idx, "atlas_dv_group_on_surface"] = final_group
            mapped.at[original_idx, "surface_snap_distance_mm"] = float(snap_distance)
            snapped_ids.append(final_vertex); snapped_groups.append(final_group); snapped_regions.append(final_name)

        if "atlas_region_on_surface" not in mapped.columns:
            mapped["atlas_region_on_surface"] = snapped_regions
        if "atlas_dv_group_on_surface" not in mapped.columns:
            mapped["atlas_dv_group_on_surface"] = snapped_groups

    # Legacy pending records are kept only if all batch and per-site retries failed.
    no_projection = mapped.loc[mapped["surface_vertex"] < 0, "site_id"].tolist()
    if no_projection:
        print(
            "WARNING: no fsLR vertex was found for: "
            + ", ".join(no_projection)
            + ". Check MNI152 variant/coordinate convention and whether contacts "
              "are sufficiently close to the cortical ribbon."
        )

    outside_mask = mapped.loc[
        (mapped["surface_vertex"] >= 0) & (~mapped["surface_in_julich_mask"]),
        "site_id",
    ].tolist()
    if outside_mask:
        print(
            "INFO: these sites projected to fsLR but are outside the thresholded "
            "Julich insula mask; they remain available for plotting: "
            + ", ".join(outside_mask)
        )

    snapped_mask = mapped["projection_method"].astype(str).str.startswith("nearest_julich_")
    snapped_mask &= pd.to_numeric(mapped["surface_snap_distance_mm"], errors="coerce").fillna(0.0) > 0
    snapped = mapped.loc[snapped_mask, "site_id"].tolist()
    if snapped:
        print(
            "INFO: snapped to the nearest Julich surface vertex in the manually "
            "selected dorsal/ventral group (distance on inflated fsLR surface "
            "recorded in surface_snap_distance_mm): " + ", ".join(snapped)
        )

    return None, mapped
