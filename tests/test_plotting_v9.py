"""Offline regression test: context transparency is separate from insula opacity."""
import sys
import types
sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from insula_plotter.plotting import _add_surface_panel


def test_context_faces_translucent_but_julich_parcels_opaque():
    # First triangle is a labelled insula parcel; last triangle is unlabelled context.
    vertices = np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        [0., 0., 1.], [1., 1., 1.], [2., 0., 0.],
        [2., 1., 0.], [2., 0., 1.],
    ])
    faces = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7]])
    meshes = {"right": (vertices, faces), "left": (vertices, faces)}
    atlas_surface = types.SimpleNamespace(
        right=np.array([1, 1, 1, 0, 0, 0, 0, 0]),
        left=np.array([1, 1, 1, 0, 0, 0, 0, 0]))
    atlas_result = types.SimpleNamespace(
        region_names={1: "Id4"}, region_colors={1: "#dd7777"})
    sites = pd.DataFrame(columns=["site_id", "hemisphere", "surface_vertex",
                                  "surface_x", "surface_y", "surface_z", "color"])
    cfg = {
        "context_distance_mm": 10., "context_smoothing_rings": 0,
        "context_alpha": 0.12, "parcel_alpha": 1.0,
        "context_shading_strength": 0.3, "context_ambient": 0.7,
        "parcel_shading_strength": 0.1, "parcel_ambient": 0.8,
        "elev": 18, "right_azim": 0, "marker_size": 10,
        "zoom_to_insula_only": False, "zoom_context_margin_mm": 1.,
        "zoom_padding_fraction": 0.1, "zoom_min_padding_mm": 1.,
        "parcel_boundary_linewidth": 0.2, "dv_boundary_linewidth": 0.5,
    }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _add_surface_panel(ax, "right", meshes, atlas_surface, sites,
                       atlas_result, cfg, mode="region_labels")
    collections = [c for c in ax.collections if hasattr(c, "get_facecolors")]
    assert collections, "Expected a 3D surface collection"
    rgba = np.asarray(collections[0].get_facecolors())
    assert np.any(np.isclose(rgba[:, 3], 0.12)), rgba
    assert np.any(np.isclose(rgba[:, 3], 1.0)), rgba
    plt.close(fig)
