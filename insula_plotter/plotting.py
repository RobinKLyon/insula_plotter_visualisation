from __future__ import annotations

from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.collections import PolyCollection, LineCollection
from .flatmap import build_insula_flatmap
from scipy.spatial import cKDTree

DORSAL_CODES = {"id4", "id6", "id7"}
VENTRAL_CODES = {"ia2", "ia3", "id8", "id9", "id10"}
STATUS_ORDER = ("gain_only", "loss_only", "both", "both_prefer_gain", "both_prefer_loss")
DEFAULT_STATUS_COLORS = {
    "gain_only": "#259B55",
    "loss_only": "#B9824A",
    "both": "#8C564B",
    "both_prefer_gain": "#C66A4E",
    "both_prefer_loss": "#65433D",
}
DEFAULT_DV_COLORS = {"Dorsale": "#D95F59", "Ventrale": "#4C83B6"}

# Site-bearing parcels use coherent within-family shades rather than unrelated hues.
DEFAULT_FAMILY_REGION_COLORS = {
    # Warm coral shades: dorsal insula
    "id4": "#B94F45", "id6": "#D46D5F", "id7": "#E99B8E",
    # Blue/teal shades: ventral insula
    "ia2": "#2E718F", "ia3": "#438BA5", "id8": "#62A6BA",
    "id9": "#8BC1CE", "id10": "#B4D9E0",
}

def _adjust_contrast(hex_color, contrast=1.0):
    """Adjust RGB contrast around mid-grey while preserving a smooth parcel palette."""
    rgb = np.asarray(to_rgba(hex_color)[:3], dtype=float)
    value = float(max(0.0, contrast))
    rgb = np.clip((rgb - 0.5) * value + 0.5, 0.0, 1.0)
    return (*rgb, 1.0)

def _family_region_palette(atlas_result, cfg):
    palette = dict(DEFAULT_FAMILY_REGION_COLORS)
    palette.update({str(k).lower(): str(v) for k, v in cfg.get("family_region_colors", {}).items()})
    _names, codes = _label_lookup(atlas_result)
    contrast = float(cfg.get("parcel_color_contrast", 1.0))
    return {lid: _adjust_contrast(palette[code], contrast)
            for lid, code in codes.items() if code in palette}


def _region_code(value) -> str:
    m = re.search(r"(?<![a-z0-9])(ig\d+|ia\d+|id\d+)(?!\d)", str(value).strip().lower())
    return m.group(1) if m else ""


def _dv_group(area) -> str:
    code = _region_code(area)
    if code in DORSAL_CODES:
        return "Dorsale"
    if code in VENTRAL_CODES:
        return "Ventrale"
    return "Non classée"


def _load_fslr_surface(density: str):
    from neuromaps.datasets import fetch_atlas
    atlas = fetch_atlas("fsLR", density)
    inflated = atlas["inflated"]
    meshes = {}
    for side, key, idx in (("left", "L", 0), ("right", "R", 1)):
        path = getattr(inflated, key) if hasattr(inflated, key) else inflated[idx]
        img = nib.load(path)
        meshes[side] = (
            np.asarray(img.agg_data("NIFTI_INTENT_POINTSET"), dtype=float),
            np.asarray(img.agg_data("NIFTI_INTENT_TRIANGLE"), dtype=np.int64),
        )
    return meshes


def _grow_vertex_mask(seed_mask, faces, rings=1):
    mask = np.asarray(seed_mask, dtype=bool).copy()
    for _ in range(max(0, int(rings))):
        hit = np.any(mask[faces], axis=1)
        if not hit.any():
            break
        mask[np.unique(faces[hit])] = True
    return mask


def _context_mask(seed_mask, vertices, faces, cfg):
    """Keep the insula plus nearby cortex on the same inflated surface.

    A Euclidean neighbourhood on the inflated mesh is used as a practical
    local context window, then a few mesh rings are added to avoid jagged edges.
    """
    radius = float(cfg.get("context_distance_mm", 24.0))
    seed_ids = np.flatnonzero(seed_mask)
    if len(seed_ids) == 0:
        return np.asarray(seed_mask, dtype=bool)
    tree = cKDTree(vertices[seed_ids])
    dist, _ = tree.query(vertices, k=1)
    mask = np.asarray(seed_mask, dtype=bool) | (dist <= radius)
    return _grow_vertex_mask(mask, faces, int(cfg.get("context_smoothing_rings", 2)))


def _face_labels(vertex_labels, faces):
    tri = vertex_labels[faces]
    out = np.zeros(len(faces), dtype=np.int64)
    for i, row in enumerate(tri):
        vals, counts = np.unique(row[row > 0], return_counts=True)
        if len(vals):
            out[i] = vals[np.argmax(counts)]
    return out

def _split_labelled_triangles(vertices, faces, vertex_labels):
    """Split each mesh triangle into label polygons using edge-midpoint borders.

    A face with two labels is split at the midpoints of its label-changing edges.
    A three-label junction uses the triangle centroid as the common junction.
    This prevents whole triangular facets from being coloured as if they were a
    parcel and makes the visible parcel border pass through triangle interiors.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    vertex_labels = np.asarray(vertex_labels, dtype=int)
    polygons, polygon_labels = [], []
    for face in faces:
        tri = vertices[face]
        labs = vertex_labels[face]
        unique = np.unique(labs)
        centroid = tri.mean(axis=0)
        midpoints = {(i, (i + 1) % 3): (tri[i] + tri[(i + 1) % 3]) * 0.5 for i in range(3)}
        for label in unique:
            points = []
            for i in range(3):
                prev_i = (i - 1) % 3
                next_i = (i + 1) % 3
                if labs[i] != label:
                    continue
                if labs[prev_i] != label:
                    points.append((tri[prev_i] + tri[i]) * 0.5)
                points.append(tri[i])
                if labs[next_i] != label:
                    points.append((tri[i] + tri[next_i]) * 0.5)
            if len(unique) == 3:
                points.append(centroid)
            if len(points) >= 3:
                polygons.append(np.asarray(points, dtype=float))
                polygon_labels.append(int(label))
    return polygons, np.asarray(polygon_labels, dtype=int)


def _camera_direction(elev, azim):
    el, az = np.deg2rad([elev, azim])
    v = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return v / np.linalg.norm(v)


def _shade_face_colors(vertices, faces, base_colors, light_direction, strength=0.55,
                       ambient=0.58):
    """Apply simple directional diffuse shading to 3D triangles.

    Matplotlib's Poly3DCollection does not provide robust mesh lighting across
    versions. This face-normal shading adds depth cues while preserving parcel hues.
    """
    colors = np.asarray(base_colors, dtype=float).copy()
    if len(faces) == 0:
        return colors
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12
    normals[good] /= lengths[good, None]
    light = np.asarray(light_direction, dtype=float)
    light /= max(np.linalg.norm(light), 1e-12)
    # Absolute dot product avoids dark patches from inconsistent mesh winding.
    diffuse = np.abs(normals @ light)
    illumination = float(ambient) + (1.0 - float(ambient)) * diffuse
    factor = (1.0 - float(strength)) + float(strength) * illumination
    colors[:, :3] *= factor[:, None]
    colors[:, :3] = np.clip(colors[:, :3], 0.0, 1.0)
    return colors


def _shade_polygon_colors(polygons, base_colors, light_direction, strength=0.55, ambient=0.58):
    """Apply one face-normal shading factor to polygons split inside source triangles."""
    colors = np.asarray(base_colors, dtype=float).copy()
    if not len(polygons):
        return colors
    normals = []
    for poly in polygons:
        if len(poly) < 3:
            normals.append([0.0, 0.0, 1.0])
        else:
            normals.append(np.cross(poly[1] - poly[0], poly[2] - poly[0]))
    normals = np.asarray(normals, dtype=float)
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12
    normals[good] /= lengths[good, None]
    light = np.asarray(light_direction, dtype=float)
    light /= max(np.linalg.norm(light), 1e-12)
    diffuse = np.abs(normals @ light)
    illumination = float(ambient) + (1.0 - float(ambient)) * diffuse
    factor = (1.0 - float(strength)) + float(strength) * illumination
    colors[:, :3] *= factor[:, None]
    colors[:, :3] = np.clip(colors[:, :3], 0.0, 1.0)
    return colors


def _label_lookup(atlas_result):
    by_id = {int(k): str(v) for k, v in atlas_result.region_names.items()}
    code_by_id = {k: _region_code(v) for k, v in by_id.items()}
    return by_id, code_by_id


def _normalize_status(s):
    s = str(s).strip().lower().replace(" ", "_").replace("-", "_")
    return {"both_preger_gain": "both_prefer_gain", "both_preger_loss": "both_prefer_loss"}.get(s, s)


def _category_palette(cfg):
    p = dict(DEFAULT_STATUS_COLORS)
    p.update({str(k).lower(): str(v) for k, v in cfg.get("status_colors", {}).items()})
    return p


def _region_palette(atlas_result, cfg):
    p = {int(k): str(v) for k, v in atlas_result.region_colors.items()}
    overrides = cfg.get("region_colors", {})
    _, code_by_id = _label_lookup(atlas_result)
    for lid, code in code_by_id.items():
        if code in overrides:
            p[lid] = overrides[code]
    return p


def _unique_edges(faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    return np.unique(np.sort(edges, axis=1), axis=0)


def _triangle_boundary_segments(vertices, faces, labels, edge_predicate):
    """Approximate categorical borders through triangle interiors, not mesh edges.

    For a triangle crossed by a boundary, the crossing point is placed at the
    midpoint of each label-changing edge. Two crossings form one straight segment;
    a three-way junction is connected through the triangle centroid.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    labels = np.asarray(labels)
    segments = []
    local_edges = ((0, 1), (1, 2), (2, 0))
    for face in faces:
        tri_labels = labels[face]
        tri_xyz = vertices[face]
        crossings = []
        for i, j in local_edges:
            if edge_predicate(tri_labels[i], tri_labels[j]):
                crossings.append((tri_xyz[i] + tri_xyz[j]) * 0.5)
        if len(crossings) == 2:
            segments.append([crossings[0], crossings[1]])
        elif len(crossings) == 1:
            center = tri_xyz.mean(axis=0)
            segments.append([crossings[0], center])
        elif len(crossings) == 3:
            center = tri_xyz.mean(axis=0)
            segments.extend([[point, center] for point in crossings])
    if not segments:
        return np.empty((0, 2, vertices.shape[1]), dtype=float)
    return np.asarray(segments, dtype=float)

def _parcel_boundaries(vertices, faces, vertex_labels, roi_vertex_mask=None):
    labels = np.asarray(vertex_labels, dtype=int)
    return _triangle_boundary_segments(
        vertices, faces, labels,
        lambda a, b: (a != b) and ((a > 0) or (b > 0)),
    )

def _dv_boundaries(vertices, faces, vertex_labels, code_by_id, roi_vertex_mask=None):
    codes = np.array([code_by_id.get(int(v), "") for v in vertex_labels], dtype=object)
    groups = np.array(["D" if c in DORSAL_CODES else "V" if c in VENTRAL_CODES else "" for c in codes], dtype=object)
    return _triangle_boundary_segments(
        vertices, faces, groups,
        lambda a, b: {str(a), str(b)} == {"D", "V"},
    )


def _region_handles(atlas_result, mode, cfg, visible_label_ids=None):
    """Legend for atlas parcels that actually contain one or more displayed sites.

    Every insular parcel remains coloured on the surface. The legend is deliberately
    sparse: only parcels containing contacts receive an individual label.
    """
    _names, codes = _label_lookup(atlas_result)
    family_colors = _family_region_palette(atlas_result, cfg)
    if visible_label_ids is None:
        visible_label_ids = set()
    visible_label_ids = {int(v) for v in visible_label_ids if int(v) in codes}

    if mode == "dv_groups":
        present_groups = {_dv_group(codes[lid]) for lid in visible_label_ids}
        handles = []
        for group in ("Dorsale", "Ventrale"):
            if group not in present_groups:
                continue
            color = cfg.get("dv_group_colors", {}).get(group, DEFAULT_DV_COLORS[group])
            handles.append(Patch(facecolor=color, edgecolor="#555555", linewidth=0.45,
                                 label=group))
        return handles

    handles = []
    for lid in sorted(visible_label_ids, key=lambda k: (codes.get(k, ""), k)):
        code = codes.get(lid, "")
        if not code:
            continue
        group = _dv_group(code)
        suffix = f" — {group.lower()}" if group != "Non classée" else ""
        if lid in family_colors:
            swatch = family_colors[lid]
        else:
            swatch = cfg.get("unclassified_site_region_color", "#AEB6BF")
        handles.append(Patch(
            facecolor=swatch, edgecolor="#555555", linewidth=0.45,
            label=f"{code.upper()}{suffix}"))
    return handles


def _status_handles(all_sites, drawn_sites, cfg):
    """Return a category legend only when the CSV really contains status."""
    if "status" not in all_sites.columns:
        return []
    statuses = all_sites["status"].map(_normalize_status)
    known_total = int(statuses.isin(STATUS_ORDER).sum())
    if known_total == 0:
        return []
    drawn = drawn_sites["status"].map(_normalize_status) if "status" in drawn_sites else pd.Series([], dtype=str)
    palette = _category_palette(cfg)
    handles = []
    for category in STATUS_ORDER:
        n_total = int(statuses.eq(category).sum())
        if n_total == 0:
            continue
        n_drawn = int(drawn.eq(category).sum())
        handles.append(Line2D([0], [0], marker="o", linestyle="", color="none",
                              markerfacecolor=palette[category], markeredgecolor="#202020",
                              markersize=5.2, label=f"{category} (n={n_drawn}/{n_total})"))
    return handles


def _input_color_handles(all_sites, drawn_sites, cfg):
    """Legend for the actual per-site CSV colours when there is no status column."""
    if "status" in all_sites.columns or "color" not in all_sites.columns:
        return []
    handles = []
    for color, group in all_sites.groupby("color", sort=False):
        n_total = len(group)
        n_drawn = int(drawn_sites["site_id"].isin(group["site_id"]).sum()) if len(drawn_sites) else 0
        handles.append(Line2D([0], [0], marker="o", linestyle="", color="none",
                              markerfacecolor=str(color), markeredgecolor="#202020",
                              markersize=5.2, label=f"{str(color).upper()} ({n_drawn}/{n_total})"))
    return handles


def _tvalue_norm(sites, cfg):
    if "t_value" not in sites.columns:
        return None
    values = pd.to_numeric(sites["t_value"], errors="coerce").dropna().to_numpy(float)
    if not len(values):
        return None
    limit = float(cfg.get("tvalue_vlim", np.max(np.abs(values))))
    if limit <= 0:
        limit = float(np.max(np.abs(values))) or 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _get_display_sites(mapped_sites, side, vertex_labels, cfg, meshes=None, combined=False):
    if combined:
        frame = mapped_sites[mapped_sites["hemisphere"].isin(["left", "right"])].copy()
    else:
        frame = mapped_sites[mapped_sites["hemisphere"] == side].copy()
    frame = frame[frame["surface_vertex"].astype(int) >= 0].copy()
    frame = frame.dropna(subset=["surface_x", "surface_y", "surface_z"])
    ids = frame["surface_vertex"].astype(int).to_numpy()
    valid_idx = (ids >= 0) & (ids < len(vertex_labels))
    frame = frame.iloc[np.flatnonzero(valid_idx)].copy()
    ids = ids[valid_idx]
    # Keep every contact with a valid fsLR surface vertex. Contacts outside the
    # thresholded Julich mask are retained as explicitly flagged, hollow markers
    # rather than silently removed. This makes the full dataset visible while
    # distinguishing atlas-supported locations from out-of-mask projections.
    frame["display_on_julich_mask"] = vertex_labels[ids] > 0
    # A high-radius or snapped point can technically touch the insula patch while
    # its exact cortical location remains uncertain. Keep it visible but hollow.
    radii = pd.to_numeric(frame.get("projection_radius_mm", pd.Series(np.nan, index=frame.index)), errors="coerce")
    methods = frame.get("projection_method", pd.Series("", index=frame.index)).astype(str).str.lower()
    snap_distance = pd.to_numeric(frame.get("surface_snap_distance_mm", pd.Series(np.nan, index=frame.index)), errors="coerce")
    max_filled_radius = float(cfg.get("filled_marker_max_projection_radius_mm", 7.5))
    manual_group_snap = methods.str.contains("nearest_julich_(?:dorsale|ventrale)_surface", regex=True)
    # A manual D/V-constrained snap is intentionally accepted as a valid insular
    # marker, even when the original MNI-to-surface discrepancy was large.
    frame["display_projection_uncertain"] = (
        (~frame["display_on_julich_mask"].astype(bool))
        | (((radii > max_filled_radius)
            | methods.str.contains("single_site", regex=True)
            | (snap_distance > float(cfg.get("max_snap_distance_mm", 6.0)))
           ) & ~manual_group_snap)
    ).to_numpy(bool)
    if len(frame) and meshes is not None:
        # Use the exact fsLR vertices from the surface currently being rendered.
        # In combined mode, homologous left/right vertex IDs are represented on
        # the selected target hemisphere, following the FS_LR strategy.
        target = cfg.get("combined_target_hemisphere", side) if combined else side
        tv = meshes[target][0]
        frame["surface_x"] = tv[ids, 0]
        frame["surface_y"] = tv[ids, 1]
        frame["surface_z"] = tv[ids, 2]
        if combined:
            frame["display_origin_hemisphere"] = frame["hemisphere"]
    return frame


def _spread_overlapping_sites_3d(sites, vertices, faces, cfg):
    """Separate colliding contact markers tangentially in display space only.

    Every member of a collision cluster is displaced (not just the later point),
    symmetrically around the shared/near-shared location. Original `surface_*`
    coordinates and vertex IDs remain untouched for anatomical reporting.
    """
    if len(sites) < 2:
        out = sites.copy()
        for axis in "xyz":
            out[f"display_{axis}"] = out[f"surface_{axis}"].to_numpy(float)
        out["display_offset_mm"] = 0.0
        return out

    out = sites.copy().reset_index(drop=True)
    base = out[["surface_x", "surface_y", "surface_z"]].to_numpy(float)
    display = base.copy()
    offsets = np.zeros(len(out), dtype=float)
    collision_distance = float(cfg.get("overlap_detect_distance_mm", 0.75))
    offset_radius = float(cfg.get("overlap_offset_mm", 1.45))
    pairs = cKDTree(base).query_pairs(r=max(collision_distance, 1e-6), output_type="set")
    if not pairs:
        for j, axis in enumerate("xyz"):
            out[f"display_{axis}"] = display[:, j]
        out["display_offset_mm"] = offsets
        return out

    parent = list(range(len(out)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for a, b in pairs:
        union(a, b)
    clusters = {}
    for idx in range(len(out)):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    surface_tree = cKDTree(np.asarray(vertices, dtype=float))
    site_ids = out.get("site_id", pd.Series(np.arange(len(out)))).astype(str).to_numpy()
    for members in clusters.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda idx: site_ids[idx])
        center = base[members].mean(axis=0)
        local_ids = surface_tree.query_ball_point(center, float(cfg.get("overlap_tangent_neighborhood_mm", 5.0)))
        local = np.asarray(vertices, dtype=float)[local_ids] if len(local_ids) >= 3 else np.asarray(vertices, dtype=float)
        local_center = local.mean(axis=0)
        try:
            _u, _s, vh = np.linalg.svd(local - local_center, full_matrices=False)
            tangent_1, tangent_2 = vh[0], vh[1]
        except np.linalg.LinAlgError:
            tangent_1, tangent_2 = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        # For two coincident contacts this yields opposite shifts; for larger
        # clusters it distributes all points around a small tangent-plane ring.
        radius = offset_radius * max(1.0, np.sqrt(len(members) / 2.0))
        for order, idx in enumerate(members):
            angle = 2.0 * np.pi * order / len(members)
            delta = radius * (np.cos(angle) * tangent_1 + np.sin(angle) * tangent_2)
            display[idx] = center + delta
            offsets[idx] = float(np.linalg.norm(delta))

    for j, axis in enumerate("xyz"):
        out[f"display_{axis}"] = display[:, j]
    out["display_offset_mm"] = offsets
    return out


def _spread_overlapping_sites_2d(points, cfg):
    """Display-only 2-D offsets for coincident sites on the flatmap."""
    points = np.asarray(points, dtype=float).copy()
    if len(points) < 2:
        return points
    threshold = float(cfg.get("flatmap_overlap_detect_distance", 1e-5))
    radius = float(cfg.get("flatmap_overlap_offset", 0.018))
    pairs = cKDTree(points).query_pairs(r=max(threshold, 1e-8), output_type="set")
    if not pairs:
        return points
    parent = list(range(len(points)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for i in range(len(points)):
        groups.setdefault(find(i), []).append(i)
    out = points.copy()
    for members in groups.values():
        if len(members) < 2:
            continue
        center = points[members].mean(axis=0)
        rr = radius * max(1.0, np.sqrt(len(members) / 2.0))
        for order, idx in enumerate(members):
            angle = 2 * np.pi * order / len(members)
            out[idx] = center + rr * np.array([np.cos(angle), np.sin(angle)])
    return out


def _draw_contacts(ax, sites, camera, cfg, site_color_mode="input_color", norm=None):
    """Draw every projected site with the same filled-marker style.

    Projection confidence / Julich-mask membership is still recorded in the
    CSV reports, but it no longer changes the marker face or outline.
    """
    if len(sites) == 0:
        return None
    display_cols = [f"display_{axis}" for axis in "xyz"]
    coord_cols = display_cols if all(col in sites.columns for col in display_cols) else ["surface_x", "surface_y", "surface_z"]
    xyz = sites[coord_cols].to_numpy(float)
    xyz = xyz + camera[None, :] * float(cfg.get("point_offset_mm", 0.0))
    ax.computed_zorder = False
    size = float(cfg.get("marker_size", 18))
    edge = cfg.get("marker_edge_color", "#202020")
    linewidth = float(cfg.get("marker_edge_width", 0.45))

    if site_color_mode == "t_value":
        if "t_value" not in sites.columns:
            raise ValueError("t_value colour mode requested, but CSV has no t_value column.")
        vals = pd.to_numeric(sites["t_value"], errors="coerce").to_numpy(float)
        cmap = cfg.get("tvalue_cmap", "coolwarm")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size,
                   c=vals, cmap=cmap, norm=norm, edgecolors=edge,
                   linewidths=linewidth, depthshade=False, alpha=1.0, zorder=30)
        return None

    statuses = sites.get("status", pd.Series(["unknown"] * len(sites), index=sites.index)).map(_normalize_status)
    palette = _category_palette(cfg)
    colors = np.asarray([palette.get(st, row_color) for st, row_color in zip(statuses, sites["color"].astype(str))], dtype=object)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size,
               c=colors.tolist(), edgecolors=edge, linewidths=linewidth,
               depthshade=False, alpha=1.0, zorder=30)
    return None


def _flatten_local_fold(vertices, faces, insula_mask, cfg):
    """Compress local surface depth along the insula patch's PCA normal for display.

    This is a display-only deformation: fsLR vertex identities and stored anatomical
    coordinates remain unchanged. Weighting fades smoothly into neighbouring cortex.
    """
    strength = float(np.clip(cfg.get("fold_flatten_strength", 0.0), 0.0, 0.95))
    if strength <= 0 or int(np.count_nonzero(insula_mask)) < 10:
        return np.asarray(vertices, dtype=float)
    xyz = np.asarray(vertices, dtype=float)
    roi = xyz[np.asarray(insula_mask, dtype=bool)]
    center = roi.mean(axis=0)
    _u, _s, vh = np.linalg.svd(roi - center, full_matrices=False)
    normal = vh[-1]
    normal /= max(np.linalg.norm(normal), 1e-12)
    # Smoothly fade the flattening from full strength on the insula to zero in context.
    tree = cKDTree(roi)
    distance, _ = tree.query(xyz, k=1)
    fade = float(cfg.get("fold_flatten_fade_mm", 22.0))
    if fade <= 0:
        weight = (distance == 0).astype(float)
    else:
        weight = np.exp(-0.5 * (distance / fade) ** 2)
    signed_depth = (xyz - center) @ normal
    out = xyz - (strength * weight * signed_depth)[:, None] * normal[None, :]
    return out

def _add_surface_panel(ax, side, meshes, atlas_surface, mapped_sites, atlas_result,
                       cfg, mode="dv_labels", combined=False,
                       site_color_mode="input_color", tvalue_norm=None):
    vertices, faces = meshes[side]
    labels_img = atlas_surface.left if side == "left" else atlas_surface.right
    vlabels = np.rint(labels_img).astype(int)
    parcel_mask = vlabels > 0
    if not parcel_mask.any():
        ax.set_axis_off()
        ax.set_title(f"{side.capitalize()} hemisphere — no Julich labels")
        return [], [], pd.DataFrame()

    # Calculate displayed contacts first: only parcels containing a displayed
    # site receive saturated family colours in the default region-label view.
    sites = _get_display_sites(mapped_sites, side, vlabels, cfg, meshes=meshes, combined=combined)
    site_vertex_ids = sites["surface_vertex"].astype(int).to_numpy() if len(sites) else np.array([], dtype=int)
    visible_label_ids = set(np.unique(vlabels[site_vertex_ids]).astype(int)) - {0}

    original_vertices = np.asarray(vertices, dtype=float)
    if float(cfg.get("fold_flatten_strength", 0.0)) > 0:
        vertices = _flatten_local_fold(original_vertices, faces, parcel_mask, cfg)

    # Keep the complete atlas insula plus local surrounding inflated cortex.
    # Also retain a cortical neighbourhood beneath any displayed out-of-mask
    # contact, so its hollow QC marker is never floating over a blank background.
    context_seed = parcel_mask.copy()
    if len(sites):
        context_site_ids = sites["surface_vertex"].astype(int).to_numpy()
        context_site_ids = context_site_ids[(context_site_ids >= 0) & (context_site_ids < len(context_seed))]
        context_seed[context_site_ids] = True
    roi_mask = _context_mask(context_seed, original_vertices, faces, cfg)
    face_mask = np.any(roi_mask[faces], axis=1)
    roi_faces = faces[face_mask]
    polygons3d, polygon_labels = _split_labelled_triangles(vertices, roi_faces, vlabels)
    names, code_by_id = _label_lookup(atlas_result)
    palette = _region_palette(atlas_result, cfg)
    family_palette = _family_region_palette(atlas_result, cfg)
    show_only_site_regions = bool(cfg.get("show_only_regions_with_sites", True))
    # Only the non-insular cortical shell is translucent. Julich parcels remain
    # independently opaque (parcel_alpha), so the labels and site markers stay crisp.
    context_alpha = float(np.clip(cfg.get("context_alpha", 0.12), 0.0, 1.0))
    context = to_rgba(cfg.get("context_color", "#E4E7EB"), context_alpha)
    neutral_parcel = to_rgba(cfg.get("unselected_region_color", "#CFD3D8"), 1.0)
    face_colors = np.tile(context, (len(polygons3d), 1))

    for i, label_id in enumerate(polygon_labels):
        if label_id <= 0:
            continue
        code = code_by_id.get(int(label_id), "")
        group = _dv_group(code)
        if mode in {"all_labels", "dv_labels", "region_labels"}:
            if mode != "all_labels" and show_only_site_regions and int(label_id) not in visible_label_ids:
                face_colors[i] = neutral_parcel
            elif mode == "all_labels":
                face_colors[i] = to_rgba(
                    palette.get(int(label_id), "#999999"),
                    float(cfg.get("parcel_alpha", 1.0)))
            elif int(label_id) in family_palette:
                face_colors[i] = to_rgba(family_palette[int(label_id)], float(cfg.get("parcel_alpha", 1.0)))
            else:
                # Site-bearing parcels outside the requested D/V grouping remain
                # visible but neutral, rather than introducing a third bright hue.
                face_colors[i] = to_rgba(
                    cfg.get("unclassified_site_region_color", "#AEB6BF"),
                    float(cfg.get("parcel_alpha", 1.0)))
        elif mode == "dv_groups":
            if group in {"Dorsale", "Ventrale"}:
                col = cfg.get("dv_group_colors", {}).get(group, DEFAULT_DV_COLORS[group])
                face_colors[i] = to_rgba(col, float(cfg.get("parcel_alpha", 1.0)))
            else:
                face_colors[i] = neutral_parcel

    # Give the surrounding cortex a stronger light/shadow gradient than the
    # atlas parcels. This preserves the atlas palette but makes the 3D relief clearer.
    elev = float(cfg.get("elev", 18))
    azim = float(cfg.get("left_azim", 180) if side == "left" else cfg.get("right_azim", 0))
    camera = _camera_direction(elev, azim)
    light_direction = camera * 0.72 + np.array([0.0, 0.0, 0.68])
    shaded = face_colors.copy()
    # Shade context and parcels separately to avoid washing out the parcel colours.
    if np.any(polygon_labels <= 0):
        idx = np.flatnonzero(polygon_labels <= 0)
        shaded[idx] = _shade_polygon_colors(
            [polygons3d[i] for i in idx], shaded[idx], light_direction,
            strength=float(cfg.get("context_shading_strength", 0.78)),
            ambient=float(cfg.get("context_ambient", 0.48)))
    if np.any(polygon_labels > 0):
        idx = np.flatnonzero(polygon_labels > 0)
        shaded[idx] = _shade_polygon_colors(
            [polygons3d[i] for i in idx], shaded[idx], light_direction,
            strength=float(cfg.get("parcel_shading_strength", 0.20)),
            ambient=float(cfg.get("parcel_ambient", 0.78)))

    mesh = Poly3DCollection(polygons3d, facecolors=shaded,
                            edgecolors="none", linewidths=0, antialiased=True,
                            zsort="average")
    mesh.set_zorder(1)
    ax.add_collection3d(mesh)

    # Continuous-looking boundaries pass through the interior of each source triangle.
    parcel_lines = _parcel_boundaries(vertices, roi_faces, vlabels, roi_mask)
    if len(parcel_lines):
        grey = Line3DCollection(parcel_lines, colors=cfg.get("parcel_boundary_color", "#747B84"),
                                linewidths=float(cfg.get("parcel_boundary_linewidth", 0.55)), alpha=0.9)
        grey.set_zorder(8)
        ax.add_collection3d(grey)

    dv_lines = _dv_boundaries(vertices, roi_faces, vlabels, code_by_id, roi_mask)
    if len(dv_lines):
        dv_lines = dv_lines + camera[None, None, :] * float(cfg.get("dv_boundary_offset_mm", 0.6))
        red = Line3DCollection(dv_lines, colors=cfg.get("dv_boundary_color", "#C62828"),
                               linewidths=float(cfg.get("dv_boundary_linewidth", 1.8)), alpha=0.98)
        red.set_zorder(15)
        ax.add_collection3d(red)

    if len(sites):
        draw_ids = sites["surface_vertex"].astype(int).to_numpy()
        sites = sites.copy()
        # Start each point exactly on the current (optionally mildly flattened)
        # surface. Then offset every member of a collision cluster tangentially
        # for legibility; the stored surface coordinates remain unchanged.
        sites["surface_x"] = vertices[draw_ids, 0]
        sites["surface_y"] = vertices[draw_ids, 1]
        sites["surface_z"] = vertices[draw_ids, 2]
        sites = _spread_overlapping_sites_3d(sites, vertices, faces, cfg)
    _draw_contacts(ax, sites, camera, cfg, site_color_mode=site_color_mode, norm=tvalue_norm)

    # For unilateral panels, count original anatomical labels, not only plotted ones.
    source_sites = mapped_sites if combined else mapped_sites[mapped_sites["hemisphere"] == side]
    dv = source_sites.get("insula_area_for_dv", source_sites.get("atlas_region_top", pd.Series([], dtype=str))).map(_dv_group)
    n_d = int((dv == "Dorsale").sum())
    n_v = int((dv == "Ventrale").sum())
    shown_group = sites.get("insula_dv_group", pd.Series([], dtype=str))
    drawn_d = int((shown_group == "Dorsale").sum())
    drawn_v = int((shown_group == "Ventrale").sum())
    shown_vertices = sites["surface_vertex"].astype(int).to_numpy() if len(sites) else np.array([], dtype=int)
    shown_vertices = shown_vertices[(shown_vertices >= 0) & (shown_vertices < len(vlabels))]
    visible_label_ids = set(np.unique(vlabels[shown_vertices]).astype(int)) - {0}
    def _count_text(total, drawn):
        return str(total) if total == drawn else f"{total} ({drawn} affichés)"
    if combined:
        title = (f"Couverture bilatérale sur fsLR {side} — "
                 f"Dorsale : {_count_text(n_d, drawn_d)} | Ventrale : {_count_text(n_v, drawn_v)}")
    else:
        title = (f"Insula {side} — Dorsale : {_count_text(n_d, drawn_d)} | "
                 f"Ventrale : {_count_text(n_v, drawn_v)}")
    ax.set_title(title, fontsize=float(cfg.get("side_title_fontsize", 10.5)), pad=5)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    try:
        projection = str(cfg.get("projection_type", "persp")).lower()
        if projection == "persp":
            ax.set_proj_type("persp", focal_length=float(cfg.get("focal_length", 0.92)))
        else:
            ax.set_proj_type("ortho")
    except Exception:
        pass

    # Zoom around the actual insular parcel, not the much larger surrounding
    # cortex context mask. The context mesh is still rendered, but its outer
    # parts are cropped by the axes limits. This keeps anatomical context while
    # making the insula substantially larger in the final figure.
    zoom_mask = parcel_mask if bool(cfg.get("zoom_to_insula_only", True)) else roi_mask
    zoom_vertices = vertices[np.flatnonzero(zoom_mask)]
    # Include every projected contact in the camera limits. Otherwise a valid
    # cortical candidate outside the thresholded Julich mask could be drawn but
    # cropped away by the insula-only zoom, which looks like a missing contact.
    if len(sites) and bool(cfg.get("zoom_include_projected_sites", True)):
        contact_xyz = sites[["surface_x", "surface_y", "surface_z"]].to_numpy(float)
        contact_xyz = contact_xyz[np.isfinite(contact_xyz).all(axis=1)]
        if len(contact_xyz):
            zoom_vertices = np.vstack([zoom_vertices, contact_xyz])
    mins, maxs = zoom_vertices.min(axis=0), zoom_vertices.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    pad_fraction = float(cfg.get("zoom_padding_fraction", 0.10))
    pad_mm = float(cfg.get("zoom_context_margin_mm", 5.0))
    pad = np.maximum(spans * pad_fraction, pad_mm)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    ax.set_box_aspect(spans + 2 * pad)

    return _region_handles(atlas_result, mode, cfg, visible_label_ids), _status_handles(source_sites, sites, cfg), sites


def _add_legends(fig, region_handles, site_handles, cfg, colorbar_mappable=None):
    if region_handles:
        ncol = min(int(cfg.get("region_legend_ncol", 8)), len(region_handles))
        fig.legend(handles=region_handles, loc="lower center", bbox_to_anchor=(0.5, 0.075),
                   ncol=ncol, frameon=False, fontsize=float(cfg.get("legend_fontsize", 7.5)),
                   title="Parcelles Julich contenant au moins un contact",
                   title_fontsize=float(cfg.get("legend_title_fontsize", 8)),
                   columnspacing=0.9, handletextpad=0.35)
    if site_handles:
        is_status_legend = any(
            any(h.get_label().startswith(key) for key in STATUS_ORDER)
            for h in site_handles
        )
        fig.legend(handles=site_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncol=min(6, len(site_handles)), frameon=False,
                   fontsize=float(cfg.get("legend_fontsize", 7.5)),
                   title="Catégories de sites" if is_status_legend else "Sites — couleurs du CSV",
                   title_fontsize=float(cfg.get("legend_title_fontsize", 8)),
                   columnspacing=0.9, handletextpad=0.35)
    if colorbar_mappable is not None:
        cax = fig.add_axes([0.925, 0.31, 0.018, 0.38])
        cb = fig.colorbar(colorbar_mappable, cax=cax)
        cb.set_label("t_value", fontsize=float(cfg.get("colorbar_label_fontsize", 9)))
        cb.ax.tick_params(labelsize=float(cfg.get("colorbar_tick_fontsize", 8)))


def _combined_sites_for_hemisphere(mapped_sites, target="right"):
    # Keep original hemisphere in `display_origin_hemisphere` for auditability.
    out = mapped_sites.copy()
    out["display_origin_hemisphere"] = out["hemisphere"]
    out["hemisphere"] = target
    return out


def _make_bilateral_panel_figure(output, meshes, atlas_surface, mapped_sites, atlas_result,
                                 cfg, mode, filename, combined=False,
                                 site_color_mode="input_color", tvalue_norm=None):
    region_handles, site_handles = [], []
    colorbar_mappable = None
    if combined:
        fig = plt.figure(figsize=tuple(cfg.get("combined_figsize", [12.0, 9.0])), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        target = cfg.get("combined_target_hemisphere", "right")
        combined_data = _combined_sites_for_hemisphere(mapped_sites, target=target)
        rh, _sh, shown = _add_surface_panel(
            ax, target, meshes, atlas_surface, combined_data, atlas_result, cfg,
            mode=mode, combined=True, site_color_mode=site_color_mode, tvalue_norm=tvalue_norm)
        region_handles = rh
        site_handles = [] if site_color_mode == "t_value" else _status_handles(mapped_sites, shown, cfg)
        if not site_handles and site_color_mode == "input_color":
            site_handles = _input_color_handles(mapped_sites, shown, cfg)
    else:
        fig = plt.figure(figsize=tuple(cfg.get("bilateral_figsize", [15.0, 8.5])), facecolor="white")
        shown = []
        region_handle_map = {}
        for i, side in enumerate(("left", "right"), start=1):
            ax = fig.add_subplot(1, 2, i, projection="3d")
            r, _s, shown_side = _add_surface_panel(
                ax, side, meshes, atlas_surface, mapped_sites, atlas_result, cfg,
                mode=mode, site_color_mode=site_color_mode, tvalue_norm=tvalue_norm)
            for handle in r:
                region_handle_map[handle.get_label()] = handle
            shown.append(shown_side)
        region_handles = list(region_handle_map.values())
        shown_all = pd.concat(shown, ignore_index=True) if shown else mapped_sites.iloc[0:0]
        site_handles = [] if site_color_mode == "t_value" else _status_handles(mapped_sites, shown_all, cfg)
        if not site_handles and site_color_mode == "input_color":
            site_handles = _input_color_handles(mapped_sites, shown_all, cfg)
    fig.suptitle(cfg.get("title", "Insular sites on fsLR inflated surface"),
                 fontsize=float(cfg.get("title_fontsize", 13)), y=0.98)
    if site_color_mode == "t_value" and tvalue_norm is not None:
        import matplotlib as mpl
        mappable = mpl.cm.ScalarMappable(norm=tvalue_norm, cmap=cfg.get("tvalue_cmap", "coolwarm"))
        colorbar_mappable = mappable
    fig.subplots_adjust(left=0.015, right=0.90 if colorbar_mappable is not None else 0.985,
                        top=0.90, bottom=0.23 if region_handles else 0.10, wspace=0.02)
    _add_legends(fig, region_handles, site_handles, cfg, colorbar_mappable=colorbar_mappable)
    fig.savefig(output / filename, dpi=int(cfg.get("dpi", 300)), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output / filename


def _write_overlap_report(mapped_sites, drawn_sites, output_dir):
    if len(drawn_sites):
        keys = ["hemisphere", "surface_vertex"]
        counts = drawn_sites.groupby(keys, dropna=False).agg(
            n_sites=("site_id", "size"), site_ids=("site_id", lambda x: ";".join(map(str, x))),
            statuses=("status", lambda x: ";".join(map(str, x))) if "status" in drawn_sites else ("site_id", "size")
        ).reset_index()
        counts = counts[counts["n_sites"] > 1]
    else:
        counts = pd.DataFrame(columns=["hemisphere", "surface_vertex", "n_sites", "site_ids", "statuses"])
    counts.to_csv(output_dir / "overlapping_sites_report.csv", index=False)


def _export_workbench(meshes, atlas_surface, mapped_sites, atlas_result, output_dir, cfg):
    if not bool(cfg.get("export_workbench", True)):
        return
    wb = output_dir / "workbench"
    wb.mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        vertices, faces = meshes[side]
        vlabels = np.rint(atlas_surface.left if side == "left" else atlas_surface.right).astype(int)
        roi = _context_mask(vlabels > 0, vertices, faces, cfg)
        face_keep = np.any(roi[faces], axis=1)
        f = faces[face_keep]
        used = np.unique(f)
        remap = np.full(len(vertices), -1, int); remap[used] = np.arange(len(used))
        meta = nib.gifti.GiftiMetaData.from_dict({
            "AnatomicalStructurePrimary": "CortexLeft" if side == "left" else "CortexRight",
            "GeometricType": "Anatomical", "Name": f"{side} insula context", "SurfaceType": "Inflated"})
        surf = nib.gifti.GiftiImage(meta=meta, darrays=[
            nib.gifti.GiftiDataArray(data=vertices[used].astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(data=remap[f].astype(np.int32), intent="NIFTI_INTENT_TRIANGLE")])
        nib.save(surf, wb / f"insula_{side}_context.surf.gii")
        table = nib.gifti.GiftiLabelTable()
        family_palette = _family_region_palette(atlas_result, cfg)
        side_sites = mapped_sites[mapped_sites["hemisphere"] == side] if "hemisphere" in mapped_sites else mapped_sites.iloc[0:0]
        side_site_vertices = side_sites.loc[side_sites["surface_vertex"].astype(int) >= 0, "surface_vertex"].astype(int).to_numpy() if len(side_sites) else np.array([], dtype=int)
        visible_ids = set(np.unique(vlabels[side_site_vertices]).astype(int)) - {0} if len(side_site_vertices) else set()
        names_for_labels = {int(k): str(v) for k, v in atlas_result.region_names.items()}
        label_names = {0: "Unlabelled context", **names_for_labels}
        for key, name in label_names.items():
            if key == 0 or key not in visible_ids:
                col = to_rgba(cfg.get("unselected_region_color", "#D7DCE1"))
            elif key in family_palette:
                col = family_palette[key]
            else:
                col = to_rgba(cfg.get("unclassified_site_region_color", "#AEB6BF"))
            lab = nib.gifti.GiftiLabel(key=int(key), red=col[0], green=col[1], blue=col[2], alpha=col[3]); lab.label = name; table.labels.append(lab)
        arr = nib.gifti.GiftiDataArray(data=vlabels[used].astype(np.int32), intent="NIFTI_INTENT_LABEL")
        nib.save(nib.gifti.GiftiImage(darrays=[arr], labeltable=table), wb / f"insula_{side}_julich.label.gii")

        # Export site categories for interactive inspection in wb_view.
        site_data = np.zeros(len(vertices), dtype=np.int32)
        site_table = nib.gifti.GiftiLabelTable()
        blank = nib.gifti.GiftiLabel(key=0, red=0.0, green=0.0, blue=0.0, alpha=0.0)
        blank.label = "No site"
        site_table.labels.append(blank)
        side_sites = _get_display_sites(mapped_sites, side, vlabels, cfg={})
        palette = _category_palette(cfg)
        for key, row in enumerate(side_sites.itertuples(index=False), start=1):
            vertex_id = int(row.surface_vertex)
            status = _normalize_status(getattr(row, "status", "unknown"))
            rgba = to_rgba(palette.get(status, getattr(row, "color", "#333333")))
            lab = nib.gifti.GiftiLabel(key=key, red=rgba[0], green=rgba[1], blue=rgba[2], alpha=rgba[3])
            lab.label = f"{row.site_id} [{status}]"
            site_table.labels.append(lab)
            site_data[vertex_id] = key
        site_arr = nib.gifti.GiftiDataArray(data=site_data[used].astype(np.int32), intent="NIFTI_INTENT_LABEL")
        nib.save(nib.gifti.GiftiImage(darrays=[site_arr], labeltable=site_table), wb / f"insula_{side}_sites.label.gii")
    (wb / "README_workbench.txt").write_text(
        "Open insula_<side>_context.surf.gii in wb_view, then load insula_<side>_julich.label.gii.\n"
        "The label file contains the complete Julich insula map; the Python figures highlight only the selected D/V parcels.\n"
        "Optionally load insula_<side>_sites.label.gii to inspect contact categories/sites.\n",
        encoding="utf-8")


def _flatmap_edges_with_counts(faces):
    all_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    return np.unique(np.sort(all_edges, axis=1), axis=0, return_counts=True)


def _flatmap_draw_panel(ax, side, flat, meshes, atlas_surface, mapped_sites,
                        atlas_result, cfg, combined=False,
                        site_color_mode="input_color", tvalue_norm=None,
                        parcel_mode="region_labels"):
    """Render a topologically unfolded 2-D map of the whole insular patch."""
    vertices, _full_faces = meshes[side]
    vlabels = np.rint(atlas_surface.left if side == "left" else atlas_surface.right).astype(int)
    xy, faces, _face_labels_unused = flat.xy, flat.faces, flat.face_labels
    polygons, polygon_labels = _split_labelled_triangles(xy, faces, vlabels)
    _names, code_by_id = _label_lookup(atlas_result)
    palette = _region_palette(atlas_result, cfg)
    family_palette = _family_region_palette(atlas_result, cfg)
    sites = _get_display_sites(mapped_sites, side, vlabels, cfg, meshes=meshes, combined=combined)
    site_vertex_ids = sites["surface_vertex"].astype(int).to_numpy() if len(sites) else np.array([], dtype=int)
    visible_label_ids = set(np.unique(vlabels[site_vertex_ids]).astype(int)) - {0}

    face_colors = []
    for lid in polygon_labels:
        if lid <= 0:
            face_colors.append(to_rgba(cfg.get("flatmap_context_color", "#D9DDE2"), 1.0))
            continue
        code = code_by_id.get(int(lid), "")
        group = _dv_group(code)
        if parcel_mode == "dv_groups":
            col = cfg.get("dv_group_colors", {}).get(group, DEFAULT_DV_COLORS.get(group, "#C8CDD3"))
        elif bool(cfg.get("show_only_regions_with_sites", True)) and int(lid) not in visible_label_ids:
            col = cfg.get("unselected_region_color", "#D4D8DE")
        else:
            col = family_palette.get(int(lid), cfg.get("unclassified_site_region_color", "#AEB6BF"))
        face_colors.append(to_rgba(col, 1.0))
    patch = PolyCollection(polygons, facecolors=face_colors, edgecolors="none",
                           linewidths=0, antialiased=True, zorder=1)
    ax.add_collection(patch)

    # Smooth-looking parcel boundaries are drawn through triangle interiors at
    # edge midpoints, rather than tracing every mesh edge (which looked jagged).
    parcel_segments = _triangle_boundary_segments(
        xy, faces, vlabels, lambda a, b: (a != b) and ((a > 0) or (b > 0)))
    if len(parcel_segments):
        ax.add_collection(LineCollection(
            parcel_segments, colors=cfg.get("parcel_boundary_color", "#747E89"),
            linewidths=float(cfg.get("flatmap_boundary_linewidth", 0.65)),
            alpha=0.95, zorder=4))

    # Red dorsal/ventral boundary as a continuous line through triangle interiors.
    groups = np.array(["D" if code_by_id.get(int(v), "") in DORSAL_CODES
                       else "V" if code_by_id.get(int(v), "") in VENTRAL_CODES
                       else "" for v in vlabels], dtype=object)
    dv_segments = _triangle_boundary_segments(
        xy, faces, groups, lambda a, b: {str(a), str(b)} == {"D", "V"})
    if len(dv_segments):
        ax.add_collection(LineCollection(
            dv_segments, colors=cfg.get("dv_boundary_color", "#C62828"),
            linewidths=float(cfg.get("flatmap_dv_linewidth", 1.8)),
            alpha=1.0, zorder=6))

    # Annotate only the dorsal/ventral regions, as requested; other parcels
    # remain coloured and can be identified through their site-linked legend.
    if bool(cfg.get("flatmap_show_dv_region_labels", True)):
        for lid, code in code_by_id.items():
            group = _dv_group(code)
            if group not in {"Dorsale", "Ventrale"}:
                continue
            ids = np.flatnonzero((vlabels == int(lid)) & np.isfinite(xy).all(axis=1))
            if len(ids) < int(cfg.get("flatmap_min_label_vertices", 3)):
                continue
            center = np.nanmean(xy[ids], axis=0)
            ax.text(center[0], center[1], f"{code.upper()}\n{group}",
                    ha="center", va="center",
                    fontsize=float(cfg.get("flatmap_region_fontsize", 6.5)),
                    color=cfg.get("flatmap_region_text_color", "#20252B"),
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.72), zorder=10)

    if len(sites):
        vertex_ids = sites["surface_vertex"].astype(int).to_numpy()
        in_bounds = (vertex_ids >= 0) & (vertex_ids < len(xy))
        safe_ids = np.clip(vertex_ids, 0, len(xy) - 1)
        flat_vertex_ids = safe_ids.copy()
        finite_flat = np.isfinite(xy[safe_ids]).all(axis=1)
        # Contacts outside the atlas patch can still be shown on the flatmap by
        # snapping their display marker to the nearest vertex of the unfolded
        # patch. This is a display-only approximation and is clearly drawn hollow.
        needs_flat_snap = in_bounds & ~finite_flat
        valid_flat_ids = np.flatnonzero(np.isfinite(xy).all(axis=1))
        flat_snap_distance = np.zeros(len(sites), dtype=float)
        if needs_flat_snap.any() and len(valid_flat_ids):
            flat_tree = cKDTree(vertices[valid_flat_ids])
            d, nearest = flat_tree.query(vertices[safe_ids[needs_flat_snap]], k=1)
            flat_snap_distance[needs_flat_snap] = d
            max_flat_snap = float(cfg.get("flatmap_site_snap_distance_mm", 20.0))
            accepted_local = np.asarray(d) <= max_flat_snap
            affected = np.flatnonzero(needs_flat_snap)
            flat_vertex_ids[affected[accepted_local]] = valid_flat_ids[np.asarray(nearest)[accepted_local]]
            finite_flat[affected[accepted_local]] = True
        keep_sites = in_bounds & finite_flat
        sites = sites.iloc[np.flatnonzero(keep_sites)].copy()
        flat_vertex_ids = flat_vertex_ids[keep_sites]
        sites["flatmap_display_vertex"] = flat_vertex_ids
        sites["flatmap_snap_distance_mm"] = flat_snap_distance[keep_sites]
        sites["display_on_julich_mask"] = (
            vlabels[sites["surface_vertex"].astype(int).to_numpy()] > 0
        )
        points = _spread_overlapping_sites_2d(xy[flat_vertex_ids], cfg)
        marker_size = float(cfg.get("flatmap_marker_size", 12))
        marker_edge = cfg.get("marker_edge_color", "#292929")
        marker_lw = float(cfg.get("flatmap_marker_edge_width", 0.35))
        if site_color_mode == "t_value":
            vals = pd.to_numeric(sites["t_value"], errors="coerce").to_numpy(float)
            cmap = cfg.get("tvalue_cmap", "coolwarm")
            ax.scatter(points[:, 0], points[:, 1], s=marker_size,
                       c=vals, cmap=cmap, norm=tvalue_norm,
                       edgecolors=marker_edge, linewidths=marker_lw, zorder=20)
        else:
            statuses = sites.get("status", pd.Series(["unknown"] * len(sites), index=sites.index)).map(_normalize_status)
            status_palette = _category_palette(cfg)
            colors = np.asarray([status_palette.get(st, col) for st, col in zip(statuses, sites["color"].astype(str))], dtype=object)
            ax.scatter(points[:, 0], points[:, 1], s=marker_size,
                       c=colors.tolist(), edgecolors=marker_edge,
                       linewidths=marker_lw, zorder=20)

    source_sites = mapped_sites if combined else mapped_sites[mapped_sites["hemisphere"] == side]
    shown_vertices = sites["flatmap_display_vertex"].astype(int).to_numpy() if len(sites) and "flatmap_display_vertex" in sites else (sites["surface_vertex"].astype(int).to_numpy() if len(sites) else np.array([], dtype=int))
    shown_vertices = shown_vertices[(shown_vertices >= 0) & (shown_vertices < len(vlabels))]
    visible_ids = set(np.unique(vlabels[shown_vertices]).astype(int)) - {0}
    region_handles = _region_handles(atlas_result, "region_labels", cfg, visible_ids)
    site_handles = [] if site_color_mode == "t_value" else _status_handles(source_sites, sites, cfg)
    if not site_handles and site_color_mode == "input_color":
        site_handles = _input_color_handles(source_sites, sites, cfg)

    area = source_sites.get("insula_area_for_dv", source_sites.get("atlas_region_top", pd.Series([], dtype=str)))
    dv_all = area.map(_dv_group)
    n_d, n_v = int((dv_all == "Dorsale").sum()), int((dv_all == "Ventrale").sum())
    shown_group = sites.get("insula_dv_group", pd.Series([], dtype=str))
    drawn_d, drawn_v = int((shown_group == "Dorsale").sum()), int((shown_group == "Ventrale").sum())
    title = (f"Couverture bilatérale aplatie — {side} | Dorsale {n_d} ({drawn_d} affichés) | "
             f"Ventrale {n_v} ({drawn_v} affichés)" if combined else
             f"Insula {side} aplatie | Dorsale {n_d} ({drawn_d} affichés) | Ventrale {n_v} ({drawn_v} affichés)")
    ax.set_title(title, fontsize=float(cfg.get("side_title_fontsize", 10.5)), pad=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.margins(float(cfg.get("flatmap_margin", 0.055)))
    ax.axis("off")
    if bool(cfg.get("flatmap_show_orientation", True)):
        ax.text(0.02, 0.02, "Postérieur  ←  Antérieur", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.5, color="#424B55")
        ax.text(0.98, 0.02, "Ventral  ↓  Dorsal", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color="#424B55")
    return region_handles, site_handles, sites


def _make_flatmap_figure(output, meshes, atlas_surface, flatmaps, mapped_sites,
                         atlas_result, cfg, filename, combined=False,
                         site_color_mode="input_color", tvalue_norm=None,
                         parcel_mode="region_labels"):
    import matplotlib as mpl
    if combined:
        side = str(cfg.get("combined_target_hemisphere", "right"))
        fig, ax = plt.subplots(1, 1, figsize=tuple(cfg.get("flatmap_combined_figsize", [11.0, 8.0])), facecolor="white")
        combined_sites = _combined_sites_for_hemisphere(mapped_sites, target=side)
        rh, sh, shown = _flatmap_draw_panel(
            ax, side, flatmaps[side], meshes, atlas_surface, combined_sites, atlas_result,
            cfg, combined=True, site_color_mode=site_color_mode, tvalue_norm=tvalue_norm,
            parcel_mode=parcel_mode)
        region_handles, site_handles = rh, sh
    else:
        fig, axes = plt.subplots(1, 2, figsize=tuple(cfg.get("flatmap_bilateral_figsize", [15.0, 7.5])), facecolor="white")
        region_map, shown_frames = {}, []
        for ax, side in zip(axes, ("left", "right")):
            rh, _sh, shown = _flatmap_draw_panel(
                ax, side, flatmaps[side], meshes, atlas_surface, mapped_sites, atlas_result,
                cfg, combined=False, site_color_mode=site_color_mode,
                tvalue_norm=tvalue_norm, parcel_mode=parcel_mode)
            for handle in rh:
                region_map[handle.get_label()] = handle
            shown_frames.append(shown)
        region_handles = list(region_map.values())
        shown_all = pd.concat(shown_frames, ignore_index=True) if shown_frames else mapped_sites.iloc[0:0]
        site_handles = [] if site_color_mode == "t_value" else _status_handles(mapped_sites, shown_all, cfg)
        if not site_handles and site_color_mode == "input_color":
            site_handles = _input_color_handles(mapped_sites, shown_all, cfg)

    flat_legend_sites = shown if combined else shown_all
    # Save the display-only correspondence used for contacts that lie just
    # outside the flatmap domain. Original surface coordinates remain untouched.
    if filename == "insula_flatmap_merged.png" and len(flat_legend_sites):
        report_cols = [c for c in ["site_id", "hemisphere", "surface_vertex",
                                  "flatmap_display_vertex", "flatmap_snap_distance_mm",
                                  "display_on_julich_mask"] if c in flat_legend_sites.columns]
        flat_legend_sites[report_cols].drop_duplicates("site_id").to_csv(
            output / "flatmap_site_projection_report.csv", index=False)

    title = cfg.get("flatmap_title", "Insular flatmap — Julich-Brain on fsLR")
    if parcel_mode == "dv_groups":
        title += " — dorsal / ventral"
    if site_color_mode == "t_value":
        title += " — t_value"
    fig.suptitle(title, fontsize=float(cfg.get("title_fontsize", 14)), y=0.985)
    if site_color_mode == "t_value" and tvalue_norm is not None:
        mappable = mpl.cm.ScalarMappable(norm=tvalue_norm, cmap=cfg.get("tvalue_cmap", "coolwarm"))
        cax = fig.add_axes([0.925, 0.31, 0.018, 0.40])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label("t_value", fontsize=float(cfg.get("colorbar_label_fontsize", 9)))
        cb.ax.tick_params(labelsize=float(cfg.get("colorbar_tick_fontsize", 8)))
    if region_handles:
        fig.legend(handles=region_handles, loc="lower center", bbox_to_anchor=(0.5, 0.075),
                   ncol=min(int(cfg.get("region_legend_ncol", 8)), len(region_handles)),
                   frameon=False, fontsize=float(cfg.get("legend_fontsize", 7.5)),
                   title="Régions Julich contenant des sites", title_fontsize=8)
    if site_handles:
        fig.legend(handles=site_handles, loc="lower center", bbox_to_anchor=(0.5, 0.018),
                   ncol=min(6, len(site_handles)), frameon=False,
                   fontsize=float(cfg.get("legend_fontsize", 7.5)), title="Sites")
    fig.subplots_adjust(left=0.025, right=0.89 if site_color_mode == "t_value" else 0.985,
                        top=0.91, bottom=0.19 if region_handles else 0.10, wspace=0.04)
    path = output / filename
    fig.savefig(path, dpi=int(cfg.get("dpi", 300)), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _make_flatmaps(meshes, atlas_surface, mapped_sites, atlas_result, output_dir, cfg):
    output_dir = Path(output_dir)
    flatmaps, reports, coordinate_rows = {}, [], []
    for side in ("left", "right"):
        vertices, faces = meshes[side]
        labels = np.rint(atlas_surface.left if side == "left" else atlas_surface.right).astype(int)
        flat = build_insula_flatmap(vertices, faces, labels,
                                    grow_rings=int(cfg.get("flatmap_grow_rings", 1)))
        flatmaps[side] = flat
        reports.append({"hemisphere": side, **flat.report})
        if flat.report.get("flattening_method") != "harmonic_tutte":
            print(f"WARNING: {side} flatmap used {flat.report.get('flattening_method')} because the patch has no open boundary; this is a PCA fallback, not a proper surface unfolding.")
        if int(flat.report.get("n_discarded_component_triangles", 0)) > 0:
            print(f"WARNING: {side} flatmap omitted {flat.report['n_discarded_component_triangles']} triangles from disconnected small components. Check flatmap_distortion_report.csv.")
        for vertex_id in flat.vertex_ids:
            lid = int(labels[vertex_id])
            coordinate_rows.append({
                "hemisphere": side, "surface_vertex": int(vertex_id),
                "flat_x": float(flat.xy[vertex_id, 0]), "flat_y": float(flat.xy[vertex_id, 1]),
                "label_id": lid, "label": atlas_result.region_names.get(lid, "context")})
    pd.DataFrame(reports).to_csv(output_dir / "flatmap_distortion_report.csv", index=False)
    pd.DataFrame(coordinate_rows).to_csv(output_dir / "flatmap_vertex_coordinates.csv", index=False)

    norm = _tvalue_norm(mapped_sites, cfg)
    paths = [
        _make_flatmap_figure(output_dir, meshes, atlas_surface, flatmaps, mapped_sites,
                             atlas_result, cfg, "insula_flatmap_merged.png", combined=True),
        _make_flatmap_figure(output_dir, meshes, atlas_surface, flatmaps, mapped_sites,
                             atlas_result, cfg, "insula_flatmap_bilateral.png", combined=False),
        _make_flatmap_figure(output_dir, meshes, atlas_surface, flatmaps, mapped_sites,
                             atlas_result, cfg, "insula_flatmap_dorsal_ventral_merged.png",
                             combined=True, parcel_mode="dv_groups"),
    ]
    if norm is not None:
        paths.extend([
            _make_flatmap_figure(output_dir, meshes, atlas_surface, flatmaps, mapped_sites,
                                 atlas_result, cfg, "insula_flatmap_tvalue_merged.png",
                                 combined=True, site_color_mode="t_value", tvalue_norm=norm),
            _make_flatmap_figure(output_dir, meshes, atlas_surface, flatmaps, mapped_sites,
                                 atlas_result, cfg, "insula_flatmap_tvalue_bilateral.png",
                                 combined=False, site_color_mode="t_value", tvalue_norm=norm),
        ])
    return paths


def make_surface_figures(atlas_surface, mapped_sites, atlas_result, density, output_dir, plot_config):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    meshes = _load_fslr_surface(density)
    sites = mapped_sites.copy()

    # Dorsal/ventral assignment is based on the Julich probability region at
    # each original MNI coordinate unless a valid area label was supplied.
    area_col = str(plot_config.get("area_column", "area"))
    source = sites[area_col].astype(str).str.strip() if area_col in sites else pd.Series("", index=sites.index)
    valid = ~source.str.lower().isin(["", "nan", "none", "null"])
    fallback = sites.get("atlas_region_top", pd.Series("", index=sites.index)).astype(str)
    code_valid = source.map(lambda x: bool(_region_code(x)))
    sites["insula_area_for_dv"] = source.where(valid & code_valid, fallback)
    derived_dv = sites["insula_area_for_dv"].map(_dv_group)
    if "manual_dv_group" in sites.columns:
        manual = sites["manual_dv_group"].astype(str).str.strip()
        manual_valid = manual.isin(["Dorsale", "Ventrale"])
        sites["insula_dv_group"] = derived_dv.where(~manual_valid, manual)
    else:
        sites["insula_dv_group"] = derived_dv
    sites.to_csv(output_dir / "sites_surface_coordinates.csv", index=False)

    # Plot every site with a valid fsLR cortical candidate. Sites outside the
    # thresholded Julich mask are retained as hollow markers for transparency.
    display_frames = []
    for side in ("left", "right"):
        labs = np.rint(atlas_surface.left if side == "left" else atlas_surface.right).astype(int)
        display_frames.append(_get_display_sites(sites, side, labs, plot_config, meshes=meshes))
    drawn = pd.concat(display_frames, ignore_index=True) if display_frames else sites.iloc[0:0]
    _write_overlap_report(sites, drawn, output_dir)

    # Record exactly how coincident contacts are separated in the principal
    # fused view. These offsets are display-only and do not alter MNI or fsLR IDs.
    target_side = str(plot_config.get("combined_target_hemisphere", "right"))
    target_labels = np.rint(atlas_surface.left if target_side == "left" else atlas_surface.right).astype(int)
    fused = _get_display_sites(sites, target_side, target_labels, plot_config,
                               meshes=meshes, combined=True)
    if len(fused):
        fused = _spread_overlapping_sites_3d(
            fused, meshes[target_side][0], meshes[target_side][1], plot_config)
        position_cols = [c for c in [
            "site_id", "label", "hemisphere", "insula_dv_group",
            "surface_vertex", "surface_in_julich_mask", "projection_method",
            "surface_snap_distance_mm", "surface_x", "surface_y", "surface_z",
            "display_x", "display_y", "display_z", "display_offset_mm",
        ] if c in fused.columns]
        fused[position_cols].to_csv(output_dir / "site_display_positions.csv", index=False)
    else:
        pd.DataFrame(columns=["site_id", "hemisphere", "surface_vertex", "display_offset_mm"]).to_csv(
            output_dir / "site_display_positions.csv", index=False)

    paths = []
    norm = _tvalue_norm(sites, plot_config)
    main_name = plot_config.get("figure_name", "insula_sites_fslr32k.png")

    # The main output is now the Huang-like one-surface view: left and right
    # contacts are collapsed onto one homologous fsLR hemisphere. The classic
    # two-hemisphere view is kept as a secondary comparison figure.
    paths.append(_make_bilateral_panel_figure(
        output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
        "region_labels", main_name, combined=True, site_color_mode="input_color"))

    # Optional display-only local flattening of the insular fold, preserving the
    # original oblique view and the flatmaps as separate complementary outputs.
    if bool(plot_config.get("save_fold_flattened_view", True)):
        flattened_cfg = dict(plot_config)
        flattened_cfg["fold_flatten_strength"] = float(plot_config.get("fold_flatten_strength", 0.75))
        flattened_cfg["title"] = "Insular sites — stronger local fold flattening (display only)"
        paths.append(_make_bilateral_panel_figure(
            output_dir, meshes, atlas_surface, sites, atlas_result, flattened_cfg,
            "region_labels", "insula_sites_fslr32k_fold_flattened.png", combined=True,
            site_color_mode="input_color"))
    paths.append(_make_bilateral_panel_figure(
        output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
        "region_labels", "insula_sites_fslr32k_bilateral.png", combined=False,
        site_color_mode="input_color"))
    # Merged dorsal/ventral map: two parcel classes rather than individual parcels.
    paths.append(_make_bilateral_panel_figure(
        output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
        "dv_groups", "insula_dorsal_ventral_merged_bilateral.png", combined=False,
        site_color_mode="input_color"))
    # Huang-like fused left+right coverage on one homologous fsLR hemisphere,
    # with every Julich insular parcel retaining its own atlas colour (3D).
    paths.append(_make_bilateral_panel_figure(
        output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
        "region_labels", "insula_bilateral_merged_regions_3d.png", combined=True,
        site_color_mode="input_color"))

    # Complete Julich map for anatomical comparison.
    paths.append(_make_bilateral_panel_figure(
        output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
        "all_labels", "insula_all_julich_labels_bilateral.png", combined=False,
        site_color_mode="input_color"))
    # Huang-like bilateral coverage projected onto one fsLR hemisphere.
    paths.append(_make_bilateral_panel_figure(
        output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
        "dv_labels", "insula_bilateral_on_one_hemisphere.png", combined=True,
        site_color_mode="input_color"))
    # Same merged dorsal/ventral parcellation, with both hemispheres collapsed
    # onto one fsLR hemisphere (the Huang-style single-insula summary view).
    paths.append(_make_bilateral_panel_figure(
        output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
        "dv_groups", "insula_dorsal_ventral_merged_one_hemisphere.png", combined=True,
        site_color_mode="input_color"))

    # Replot the same anatomical views with a diverging t-value colormap.
    if norm is not None:
        paths.append(_make_bilateral_panel_figure(
            output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
            "dv_labels", "insula_tvalue_dv_labels_bilateral.png", combined=False,
            site_color_mode="t_value", tvalue_norm=norm))
        paths.append(_make_bilateral_panel_figure(
            output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
            "dv_groups", "insula_tvalue_dv_merged_bilateral.png", combined=False,
            site_color_mode="t_value", tvalue_norm=norm))
        paths.append(_make_bilateral_panel_figure(
            output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
            "dv_labels", "insula_tvalue_bilateral_on_one_hemisphere.png", combined=True,
            site_color_mode="t_value", tvalue_norm=norm))
        paths.append(_make_bilateral_panel_figure(
            output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
            "dv_groups", "insula_tvalue_dv_merged_one_hemisphere.png", combined=True,
            site_color_mode="t_value", tvalue_norm=norm))
        # Explicit 3D colourbar figure for fused bilateral coverage, with
        # individually coloured Julich parcels in the anatomical background.
        paths.append(_make_bilateral_panel_figure(
            output_dir, meshes, atlas_surface, sites, atlas_result, plot_config,
            "region_labels", "insula_bilateral_merged_regions_tvalue_3d.png", combined=True,
            site_color_mode="t_value", tvalue_norm=norm))
        if bool(plot_config.get("save_fold_flattened_view", True)):
            flattened_cfg = dict(plot_config)
            flattened_cfg["fold_flatten_strength"] = float(plot_config.get("fold_flatten_strength", 0.75))
            flattened_cfg["title"] = "Insular sites — stronger local fold flattening (t_value)"
            paths.append(_make_bilateral_panel_figure(
                output_dir, meshes, atlas_surface, sites, atlas_result, flattened_cfg,
                "region_labels", "insula_sites_fslr32k_fold_flattened_tvalue.png", combined=True,
                site_color_mode="t_value", tvalue_norm=norm))

    # Additional views of the fused left+right coverage on one homologous
    # fsLR hemisphere. Angles are camera positions; the anatomical coordinates
    # and projection are unchanged between panels.
    view_specs = {
        "superior": {"elev": 62, "right_azim": 15},
        "inferior": {"elev": -62, "right_azim": 15},
        "anterior_oblique": {"elev": 12, "right_azim": 88},
        "posterior_oblique": {"elev": 12, "right_azim": -72},
    }
    for view_name, overrides in view_specs.items():
        view_cfg = dict(plot_config)
        view_cfg.update(overrides)
        view_cfg["title"] = f"Insular sites on fsLR inflated surface — {view_name.replace('_', ' ')} view"
        paths.append(_make_bilateral_panel_figure(
            output_dir, meshes, atlas_surface, sites, atlas_result, view_cfg,
            "region_labels", f"insula_fused_view_{view_name}.png", combined=True,
            site_color_mode="input_color"))
        if norm is not None and bool(plot_config.get("save_tvalue_multiview", True)):
            paths.append(_make_bilateral_panel_figure(
                output_dir, meshes, atlas_surface, sites, atlas_result, view_cfg,
                "region_labels", f"insula_fused_tvalue_{view_name}.png", combined=True,
                site_color_mode="t_value", tvalue_norm=norm))

    if bool(plot_config.get("save_separate_hemispheres", True)):
        for side in ("left", "right"):
            for site_mode, suffix in (("input_color", "input_colors"), ("t_value", "tvalue")):
                if site_mode == "t_value" and norm is None:
                    continue
                fig = plt.figure(figsize=tuple(plot_config.get("single_figsize", [7.5, 6.5])), facecolor="white")
                ax = fig.add_subplot(1, 1, 1, projection="3d")
                _add_surface_panel(ax, side, meshes, atlas_surface, sites, atlas_result,
                                   plot_config, mode="dv_labels", site_color_mode=site_mode,
                                   tvalue_norm=norm)
                if site_mode == "t_value":
                    import matplotlib as mpl
                    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plot_config.get("tvalue_cmap", "coolwarm"))
                    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02); cb.set_label("t_value")
                fig.subplots_adjust(left=0.02, right=0.90 if site_mode == "t_value" else 0.98, top=0.94, bottom=0.05)
                path = output_dir / f"insula_{side}_{suffix}.png"
                fig.savefig(path, dpi=int(plot_config.get("dpi", 300)), bbox_inches="tight", facecolor="white")
                plt.close(fig); paths.append(path)

    # Detailed counts: no fake zero counts for outcome categories absent from the CSV.
    rows = []
    for side in ("left", "right"):
        subset = sites[sites.hemisphere == side]
        drawn_side = drawn[drawn.hemisphere == side]
        for group in ("Dorsale", "Ventrale", "Non classée"):
            g = subset[subset.insula_dv_group == group]
            ids = set(g.site_id)
            drawn_g = drawn_side[drawn_side.site_id.isin(ids)]
            in_mask_drawn = int(drawn_g.get("display_on_julich_mask", pd.Series(False, index=drawn_g.index)).astype(bool).sum())
            row = {"hemisphere": side, "group": group, "n_input": int(len(g)),
                   "n_displayed_total": int(len(drawn_g)),
                   "n_displayed_on_julich_mask": in_mask_drawn,
                   "n_displayed_outside_julich_mask": int(len(drawn_g) - in_mask_drawn),
                   "n_displayed_projection_uncertain": int(drawn_g.get("display_projection_uncertain", pd.Series(False, index=drawn_g.index)).astype(bool).sum()),
                   "n_not_displayed": int(len(g) - len(drawn_g))}
            if "status" in g.columns:
                for status in STATUS_ORDER:
                    row[status] = int(g["status"].map(_normalize_status).eq(status).sum())
            rows.append(row)
    counts = pd.DataFrame(rows)
    counts.to_csv(output_dir / "site_counts_dorsal_ventral.csv", index=False)

    # Since this input has no status column, record the actual source colour counts.
    color_rows = []
    for color, group in sites.groupby("color", sort=False):
        color_drawn = drawn[drawn.site_id.isin(group.site_id)]
        n_in_mask = int(color_drawn.get("display_on_julich_mask", pd.Series(False, index=color_drawn.index)).astype(bool).sum())
        color_rows.append({"csv_color": color, "n_input": len(group),
                           "n_displayed_total": int(len(color_drawn)),
                           "n_displayed_on_julich_mask": n_in_mask,
                           "n_displayed_outside_julich_mask": int(len(color_drawn) - n_in_mask),
                           "n_displayed_projection_uncertain": int(color_drawn.get("display_projection_uncertain", pd.Series(False, index=color_drawn.index)).astype(bool).sum()),
                           "n_not_displayed": int(len(group) - len(color_drawn))})
    pd.DataFrame(color_rows).to_csv(output_dir / "site_counts_by_csv_color.csv", index=False)

    # Explicit one-row-per-contact visibility audit. The display-on-mask flag is
    # evaluated against the actual hemisphere atlas labels, independent of the
    # projection routine's preliminary flag.
    visibility = sites.copy()
    displayed_ids = set(drawn.site_id.astype(str))
    visibility["is_displayed"] = visibility.site_id.astype(str).isin(displayed_ids)
    mask_flags = {}
    for side in ("left", "right"):
        labels_side = np.rint(atlas_surface.left if side == "left" else atlas_surface.right).astype(int)
        side_rows = visibility[visibility.hemisphere == side]
        for row in side_rows.itertuples(index=False):
            vertex_id = int(getattr(row, "surface_vertex", -1))
            mask_flags[str(row.site_id)] = bool(0 <= vertex_id < len(labels_side) and labels_side[vertex_id] > 0)
    visibility["display_on_julich_mask"] = visibility.site_id.astype(str).map(mask_flags).fillna(False).astype(bool)
    vis_radius = pd.to_numeric(visibility.get("projection_radius_mm", pd.Series(np.nan, index=visibility.index)), errors="coerce")
    vis_method = visibility.get("projection_method", pd.Series("", index=visibility.index)).astype(str).str.lower()
    vis_snap = pd.to_numeric(visibility.get("surface_snap_distance_mm", pd.Series(np.nan, index=visibility.index)), errors="coerce")
    visibility["display_projection_uncertain"] = (
        (~visibility["display_on_julich_mask"])
        | (vis_radius > float(plot_config.get("filled_marker_max_projection_radius_mm", 7.5)))
        | vis_method.str.contains("snap|single_site", regex=True)
        | (vis_snap > float(plot_config.get("max_snap_distance_mm", 6.0)))
    )
    visibility["visibility_status"] = np.where(
        ~visibility.is_displayed, "not_projected_or_invalid_vertex",
        np.where(~visibility.display_on_julich_mask, "displayed_outside_julich_mask_hollow",
                 np.where(visibility.get("display_projection_uncertain", pd.Series(False, index=visibility.index)).astype(bool),
                          "displayed_on_julich_mask_uncertain_hollow", "displayed_on_julich_mask_filled")))
    visibility.to_csv(output_dir / "site_visibility_report.csv", index=False)

    _export_workbench(meshes, atlas_surface, sites, atlas_result, output_dir, plot_config)

    # Export true 2-D surface-unfolding maps in addition to the 3-D views.
    if bool(plot_config.get("save_flatmaps", True)):
        paths.extend(_make_flatmaps(meshes, atlas_surface, sites, atlas_result, output_dir, plot_config))
    return paths

