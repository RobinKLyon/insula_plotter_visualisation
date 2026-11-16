"""Offline regression tests for the v0.8 plotting changes."""
import sys
import types

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from insula_plotter import plotting
from insula_plotter.plotting import _make_bilateral_panel_figure, _tvalue_norm


def _synthetic_inputs():
    vertices = np.array([
        [0., 0., 1.], [0., 0., -1.], [1., 0., 0.],
        [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.],
    ])
    faces = np.array([
        [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
        [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
    ])
    atlas_surface = types.SimpleNamespace(
        left=np.array([1, 1, 2, 2, 1, 2]), right=np.array([1, 1, 2, 2, 1, 2]))
    atlas = types.SimpleNamespace(
        region_names={1: "Id4 (left)", 2: "Ia2 (left)"},
        region_colors={1: "#e77b7b", 2: "#7ba7e7"})
    rows = []
    for site, hemi, vertex, color, tv, area in [
        ("L1", "left", 0, "#A80000", -2.0, "Id4"),
        ("R1", "right", 4, "#16803C", 1.5, "Ia2"),
    ]:
        xyz = vertices[vertex]
        rows.append(dict(site_id=site, label=site, hemisphere=hemi,
                         surface_vertex=vertex, surface_x=xyz[0],
                         surface_y=xyz[1], surface_z=xyz[2], color=color,
                         t_value=tv, area=area))
    sites = pd.DataFrame(rows)
    meshes = {"left": (vertices.copy(), faces.copy()),
              "right": (vertices.copy(), faces.copy())}
    cfg = dict(
        context_distance_mm=2.0, context_smoothing_rings=0,
        context_color="#D9DDE2", context_alpha=1.0,
        context_shading_strength=0.8, context_ambient=0.45,
        parcel_shading_strength=0.2, parcel_ambient=0.8,
        marker_size=15, marker_edge_color="#222222", marker_edge_width=0.35,
        point_offset_mm=0.18, elev=18, left_azim=180, right_azim=0,
        projection_type="persp", focal_length=0.92,
        zoom_to_insula_only=True, zoom_context_margin_mm=5.0,
        zoom_padding_fraction=0.10, zoom_min_padding_mm=5.0,
        parcel_boundary_color="#777777", parcel_boundary_linewidth=0.5,
        dv_boundary_color="#cc2222", dv_boundary_offset_mm=0.3,
        dv_boundary_linewidth=1.2, side_title_fontsize=8,
        region_legend_ncol=6, legend_fontsize=7,
        legend_title_fontsize=8, title="synthetic 3D test",
        title_fontsize=10, combined_figsize=[7., 6.], status_colors={},
        tvalue_vlim=3.0, tvalue_cmap="coolwarm", save_tvalue_multiview=True,
        save_separate_hemispheres=False, export_workbench=False,
    )
    return meshes, atlas_surface, sites, atlas, cfg


def test_close_up_defaults_reduce_marker_float_and_zoom_to_parcel():
    _, _, _, _, cfg = _synthetic_inputs()
    assert cfg["marker_size"] < 25
    assert cfg["point_offset_mm"] < 0.85
    assert cfg["zoom_to_insula_only"] is True


def test_alternative_fused_views_and_tvalue_views_are_generated(tmp_path, monkeypatch):
    meshes, atlas_surface, sites, atlas, cfg = _synthetic_inputs()
    monkeypatch.setattr(plotting, "_load_fslr_surface", lambda density: meshes)
    monkeypatch.setattr(plotting, "_export_workbench", lambda *args, **kwargs: None)

    paths = plotting.make_surface_figures(
        atlas_surface, sites, atlas, "32k", tmp_path, cfg)
    names = {p.name for p in paths}
    expected = {
        "insula_fused_view_superior.png",
        "insula_fused_view_inferior.png",
        "insula_fused_view_anterior_oblique.png",
        "insula_fused_view_posterior_oblique.png",
        "insula_fused_tvalue_superior.png",
        "insula_fused_tvalue_inferior.png",
    }
    assert expected.issubset(names)
    for name in expected:
        assert (tmp_path / name).exists()
        assert (tmp_path / name).stat().st_size > 1000
