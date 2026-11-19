"""Small offline smoke tests for v0.7 plotting helpers (no atlas downloads)."""
import sys
import types

# nibabel is required by the full project but is intentionally stubbed in this
# offline test: these tests exercise pure plotting helpers only.
sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from insula_plotter.plotting import (
    _shade_face_colors, _region_handles, _add_surface_panel,
    _make_bilateral_panel_figure, _tvalue_norm,
)


def test_region_legend_includes_posterior_parcels():
    class Atlas:
        region_names = {
            1: "Id4 (left)", 2: "Ia2 (left)", 3: "Ig1 (left)",
            4: "Id2 (left)", 5: "Id3 (left)", 6: "Id5 (left)",
        }
        region_colors = {lid: "#00ff00" for lid in region_names}
    handles = _region_handles(Atlas(), "region_labels", {}, Atlas.region_names)
    swatches = {h.get_label(): h.get_facecolor() for h in handles}
    posterior_labels = {f"{code} — postérieure" for code in ("IG1", "ID2", "ID3", "ID5")}
    assert set(swatches) == posterior_labels | {"ID4 — dorsale", "IA2 — ventrale"}
    yellows = {swatches[label] for label in posterior_labels}
    assert len(yellows) == len(posterior_labels)
    assert all(red >= green > blue for red, green, blue, _ in yellows)


def test_face_shading_is_finite_and_preserves_alpha():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2]])
    colors = np.array([[0.5, 0.4, 0.3, 1.0]])
    shaded = _shade_face_colors(vertices, faces, colors, [0.2, 0.3, 1.0])
    assert shaded.shape == (1, 4)
    assert np.isfinite(shaded).all()
    assert shaded[0, 3] == 1.0
    assert not np.allclose(shaded[0, :3], colors[0, :3])


def _synthetic_inputs():
    # Small octahedral surface, used only to smoke-test a 3D figure path.
    vertices = np.array([
        [0., 0., 1.], [0., 0., -1.], [1., 0., 0.],
        [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.],
    ])
    faces = np.array([
        [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
        [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
    ])
    atlas_surface = types.SimpleNamespace(left=np.array([1, 1, 2, 2, 1, 2]),
                                          right=np.array([1, 1, 2, 2, 1, 2]))
    atlas = types.SimpleNamespace(
        region_names={1: "Id4 (left)", 2: "Ia2 (left)"},
        region_colors={1: "#e77b7b", 2: "#7ba7e7"},
    )
    rows = []
    for site, hemi, vertex, color, tv, area in [
        ("L1", "left", 0, "#A80000", -2.0, "Id4"),
        ("R1", "right", 4, "#16803C", 1.5, "Ia2"),
    ]:
        xyz = vertices[vertex]
        rows.append(dict(site_id=site, hemisphere=hemi, surface_vertex=vertex,
                         surface_x=xyz[0], surface_y=xyz[1], surface_z=xyz[2],
                         color=color, t_value=tv, insula_area_for_dv=area,
                         insula_dv_group="Dorsale" if area == "Id4" else "Ventrale"))
    sites = pd.DataFrame(rows)
    meshes = {"left": (vertices.copy(), faces.copy()),
              "right": (vertices.copy(), faces.copy())}
    cfg = dict(context_distance_mm=2.0, context_smoothing_rings=0,
               context_color="#D9DDE2", context_alpha=1.0,
               context_shading_strength=0.8, context_ambient=0.45,
               parcel_shading_strength=0.2, parcel_ambient=0.8,
               marker_size=24, marker_edge_color="#222222", marker_edge_width=0.5,
               point_offset_mm=0.5, elev=18, left_azim=180, right_azim=0,
               projection_type="persp", focal_length=0.92,
               zoom_padding_fraction=0.05, zoom_min_padding_mm=0.2,
               parcel_boundary_color="#777777", parcel_boundary_linewidth=0.5,
               dv_boundary_color="#cc2222", dv_boundary_offset_mm=0.3,
               dv_boundary_linewidth=1.2, side_title_fontsize=8,
               region_legend_ncol=6, legend_fontsize=7,
               legend_title_fontsize=8, title="synthetic 3D test",
               title_fontsize=10, combined_figsize=[7., 6.],
               status_colors={})
    return meshes, atlas_surface, sites, atlas, cfg


def test_combined_3d_colour_and_tvalue_figures_are_saved(tmp_path):
    meshes, atlas_surface, sites, atlas, cfg = _synthetic_inputs()
    p1 = _make_bilateral_panel_figure(
        tmp_path, meshes, atlas_surface, sites, atlas, cfg,
        "region_labels", "combined_colours.png", combined=True,
        site_color_mode="input_color")
    p2 = _make_bilateral_panel_figure(
        tmp_path, meshes, atlas_surface, sites, atlas, cfg,
        "region_labels", "combined_tvalue.png", combined=True,
        site_color_mode="t_value", tvalue_norm=_tvalue_norm(sites, {"tvalue_vlim": 3.0}))
    assert p1.exists() and p1.stat().st_size > 1000
    assert p2.exists() and p2.stat().st_size > 1000
