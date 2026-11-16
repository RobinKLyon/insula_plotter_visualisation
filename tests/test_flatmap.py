import sys
import types
sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from insula_plotter.flatmap import build_insula_flatmap
from insula_plotter import plotting


def _grid_mesh(nx=9, ny=7):
    pts = []
    for j in range(ny):
        for i in range(nx):
            y = float(i)
            z = float(j)
            # Curved surface patch to make flattening nontrivial.
            x = 0.25 * np.sin(i / 2.0) + 0.12 * np.cos(j / 1.7)
            pts.append([x, y, z])
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b, c, d = a + 1, a + nx, a + nx + 1
            faces.extend([[a, b, d], [a, d, c]])
    return np.asarray(pts), np.asarray(faces, dtype=int)


def test_harmonic_flatmap_maps_all_labelled_regions_and_keeps_vertex_ids():
    vertices, faces = _grid_mesh()
    labels = np.zeros(len(vertices), dtype=int)
    for j in range(1, 6):
        for i in range(1, 8):
            idx = j * 9 + i
            labels[idx] = 1 if i <= 3 else 2 if i <= 5 else 3
    result = build_insula_flatmap(vertices, faces, labels, grow_rings=1)
    assert result.report["flattening_method"] == "harmonic_tutte"
    assert result.report["n_patch_triangles"] > 10
    assert np.isfinite(result.xy[result.vertex_ids]).all()
    assert set(np.unique(result.face_labels)) >= {1, 2, 3}
    assert result.xy.shape == (len(vertices), 2)


def test_flatmap_figures_include_merged_and_tvalue_views(tmp_path):
    vertices, faces = _grid_mesh()
    labels_l = np.zeros(len(vertices), dtype=int)
    labels_r = np.zeros(len(vertices), dtype=int)
    for j in range(1, 6):
        for i in range(1, 8):
            idx = j * 9 + i
            code = 1 if i <= 3 else 2 if i <= 5 else 3
            labels_l[idx] = code
            labels_r[idx] = code + 3
    meshes = {"left": (vertices.copy(), faces.copy()), "right": (vertices.copy(), faces.copy())}
    atlas_surface = types.SimpleNamespace(left=labels_l, right=labels_r)
    atlas = types.SimpleNamespace(
        region_names={1:"Id4 (left)", 2:"Ia2 (left)", 3:"Ig1 (left)",
                      4:"Id4 (right)", 5:"Ia2 (right)", 6:"Ig1 (right)"},
        region_colors={1:"#e77b7b", 2:"#7ba7e7", 3:"#b7d9a8",
                       4:"#e77b7b", 5:"#7ba7e7", 6:"#b7d9a8"})
    rows = []
    for sid, side, vertex, color, area, tv in [
        ("L1", "left", 20, "#A80000", "Id4", -2.5),
        ("L2", "left", 30, "#16803C", "Ia2", 1.2),
        ("R1", "right", 22, "#A67C52", "Id4", -1.1),
        ("R2", "right", 32, "#BDBDBD", "Ia2", 2.0),
    ]:
        xyz = vertices[vertex]
        rows.append(dict(site_id=sid, label=sid, hemisphere=side, surface_vertex=vertex,
                         surface_x=xyz[0], surface_y=xyz[1], surface_z=xyz[2],
                         color=color, t_value=tv, insula_area_for_dv=area,
                         insula_dv_group="Dorsale" if area == "Id4" else "Ventrale"))
    sites = pd.DataFrame(rows)
    cfg = {
        "combined_target_hemisphere": "right", "dpi": 80,
        "flatmap_grow_rings": 1, "flatmap_marker_size": 10,
        "flatmap_bilateral_figsize": [8, 4], "flatmap_combined_figsize": [6, 5],
        "tvalue_vlim": 3, "save_flatmaps": True,
        "flatmap_show_dv_region_labels": True,
        "region_legend_ncol": 4, "legend_fontsize": 6,
        "title_fontsize": 9, "side_title_fontsize": 7,
        "tvalue_cmap": "coolwarm", "colorbar_label_fontsize": 7,
        "colorbar_tick_fontsize": 6, "parcel_boundary_linewidth": 0.4,
        "dv_boundary_linewidth": 1.2, "flatmap_boundary_linewidth": 0.4,
        "flatmap_dv_linewidth": 1.2, "flatmap_region_fontsize": 5,
        "status_colors": {}, "marker_edge_color": "#292929",
    }
    paths = plotting._make_flatmaps(meshes, atlas_surface, sites, atlas, tmp_path, cfg)
    names = {p.name for p in paths}
    assert "insula_flatmap_merged.png" in names
    assert "insula_flatmap_bilateral.png" in names
    assert "insula_flatmap_tvalue_merged.png" in names
    assert "flatmap_vertex_coordinates.csv" in {p.name for p in tmp_path.iterdir()}
    assert (tmp_path / "flatmap_distortion_report.csv").exists()
