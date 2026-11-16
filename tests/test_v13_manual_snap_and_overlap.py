import numpy as np
import pandas as pd
from types import SimpleNamespace

from insula_plotter import plotting, projection


def test_manual_dorsal_group_snaps_to_nearest_dorsal_surface_without_6mm_cutoff(monkeypatch):
    def fake_load_vertices(density):
        return {
            "left": np.array([[0., 0., 0.], [10., 0., 0.], [25., 0., 0.]]),
            "right": np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        }

    def fake_make_marker_volume(reference_img, sites, radius_mm):
        return {"site_ids": sites.site_id.tolist(), "radius": float(radius_mm)}

    def fake_map(image, density="32k", method="nearest"):
        left = np.zeros(3, dtype=np.int16)
        right = np.zeros(3, dtype=np.int16)
        if len(image["site_ids"]) == 1:
            left[1] = 1  # Raw MNI sphere lands on a ventral vertex.
        return projection.SurfaceProjection(left=left, right=right)

    monkeypatch.setattr(projection, "_load_inflated_vertices", fake_load_vertices)
    monkeypatch.setattr(projection, "_make_marker_volume", fake_make_marker_volume)
    monkeypatch.setattr(projection, "_map_mni_to_fslr", fake_map)

    sites = pd.DataFrame({
        "site_id": ["S_manual"], "label": ["manual"],
        "x": [-30.], "y": [10.], "z": [2.],
        "color": ["#ff0000"], "hemisphere": ["left"],
        "manual_dv_group": ["Dorsale"],
    })
    # 1=Id6 dorsal, 2=Id9 ventral.
    atlas = SimpleNamespace(left=np.array([1, 2, 1]), right=np.array([0, 0, 0]))
    _none, mapped = projection.project_sites_to_fslr(
        reference_img=object(), sites=sites, radius_mm=5.0,
        fallback_radii_mm=(), atlas_surface=atlas,
        last_resort_radii_mm=(),
        region_names={1: "Id6 (left)", 2: "Id9 (left)"},
    )
    assert int(mapped.loc[0, "surface_vertex"]) == 0
    assert mapped.loc[0, "projection_method"] == "nearest_julich_dorsale_surface"
    assert bool(mapped.loc[0, "surface_in_julich_mask"])
    assert mapped.loc[0, "surface_snap_distance_mm"] == 10.0


def test_all_members_of_exact_collision_are_offset_symmetrically():
    sites = pd.DataFrame({
        "site_id": ["S17", "S38"],
        "surface_x": [1.0, 1.0],
        "surface_y": [2.0, 2.0],
        "surface_z": [3.0, 3.0],
    })
    # Small planar neighbourhood used to estimate the local tangent plane.
    vertices = np.array([
        [0., 0., 3.], [1., 0., 3.], [2., 0., 3.],
        [0., 1., 3.], [1., 1., 3.], [2., 1., 3.],
        [0., 2., 3.], [1., 2., 3.], [2., 2., 3.],
    ])
    faces = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
                      [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]])
    result = plotting._spread_overlapping_sites_3d(
        sites, vertices, faces,
        {"overlap_detect_distance_mm": 0.8, "overlap_offset_mm": 1.5,
         "overlap_tangent_neighborhood_mm": 5.0},
    )
    display = result[["display_x", "display_y", "display_z"]].to_numpy()
    assert np.linalg.norm(display[0] - display[1]) > 2.9
    assert (result.display_offset_mm > 1.4).all()
    # Source surface positions are retained unchanged.
    assert np.allclose(result[["surface_x", "surface_y", "surface_z"]].to_numpy(),
                       np.array([[1., 2., 3.], [1., 2., 3.]]))
