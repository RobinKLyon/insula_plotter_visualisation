import numpy as np
import pandas as pd
from types import SimpleNamespace

from insula_plotter import plotting, projection


def test_display_sites_keeps_contacts_outside_julich_mask():
    mapped = pd.DataFrame({
        "site_id": ["inside", "outside", "inside2"],
        "hemisphere": ["right", "right", "right"],
        "surface_vertex": [0, 1, 2],
        "surface_x": [0.0, 1.0, 2.0],
        "surface_y": [0.0, 0.0, 0.0],
        "surface_z": [0.0, 0.0, 0.0],
        "surface_in_julich_mask": [True, False, True],
    })
    labels = np.array([4, 0, 8])
    meshes = {"right": (np.array([[10., 0., 0.], [11., 0., 0.], [12., 0., 0.]]), np.array([[0, 1, 2]]))}
    result = plotting._get_display_sites(mapped, "right", labels, {}, meshes=meshes)
    assert result.site_id.tolist() == ["inside", "outside", "inside2"]
    assert result.display_on_julich_mask.tolist() == [True, False, True]


def test_combined_display_keeps_outside_mask_contacts_and_maps_homologous_vertices():
    mapped = pd.DataFrame({
        "site_id": ["L1", "R1"],
        "hemisphere": ["left", "right"],
        "surface_vertex": [1, 2],
        "surface_x": [0.0, 0.0], "surface_y": [0.0, 0.0], "surface_z": [0.0, 0.0],
    })
    labels = np.array([0, 0, 3, 4])
    target = np.arange(12, dtype=float).reshape(4, 3)
    meshes = {"right": (target, np.array([[0, 1, 2]]))}
    result = plotting._get_display_sites(mapped, "right", labels, {"combined_target_hemisphere": "right"}, meshes=meshes, combined=True)
    assert result.site_id.tolist() == ["L1", "R1"]
    assert result.display_on_julich_mask.tolist() == [False, True]
    assert np.allclose(result.loc[result.site_id == "L1", ["surface_x", "surface_y", "surface_z"]].to_numpy(), target[[1]])


def test_stronger_fold_flattening_reduces_patch_depth_more():
    xx, yy = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-2, 2, 7))
    zz = 0.5 * xx**2 + 0.1 * yy**2
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    faces = []
    for j in range(6):
        for i in range(6):
            a = j * 7 + i
            faces.extend([[a, a + 1, a + 8], [a, a + 8, a + 7]])
    faces = np.asarray(faces)
    mask = np.zeros(len(vertices), dtype=bool)
    mask[8:41] = True
    mild = plotting._flatten_local_fold(vertices, faces, mask, {"fold_flatten_strength": 0.38, "fold_flatten_fade_mm": 28.0})
    stronger = plotting._flatten_local_fold(vertices, faces, mask, {"fold_flatten_strength": 0.75, "fold_flatten_fade_mm": 28.0})
    assert not np.allclose(mild[mask], stronger[mask])
    # PCA-normal depth of the patch is compressed to roughly 25% at strength .75.
    roi = vertices[mask]
    center = roi.mean(axis=0)
    _u, _s, vh = np.linalg.svd(roi - center, full_matrices=False)
    normal = vh[-1]
    before_depth = (roi - center) @ normal
    after_depth = (stronger[mask] - center) @ normal
    ratio = np.std(after_depth) / np.std(before_depth)
    assert ratio < 0.35
    # The flattening remains display-only and does not mutate source coordinates.
    assert np.allclose(vertices, np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
    assert np.isfinite(stronger).all()


def test_outside_sites_are_drawn_as_hollow_markers():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sites = pd.DataFrame({
        "surface_x": [0., 1.], "surface_y": [0., 0.], "surface_z": [0., 0.],
        "display_on_julich_mask": [True, False],
        "status": ["unknown", "unknown"],
        "color": ["#ff0000", "#00ff00"],
    })
    plotting._draw_contacts(ax, sites, np.array([0., 0., 1.]), {"marker_size": 18}, "input_color")
    # Two collections: filled marker(s) on Julich, hollow marker(s) outside.
    assert len(ax.collections) == 2
    plt.close(fig)


def test_unprojected_sites_get_individual_large_radius_retry(monkeypatch):
    def fake_load_vertices(density):
        return {
            "left": np.array([[0., 0., 0.], [100., 0., 0.], [101., 0., 0.]]),
            "right": np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        }

    def fake_make_marker_volume(reference_img, sites, radius_mm):
        return {"site_ids": sites.site_id.tolist(), "radius": float(radius_mm)}

    def fake_map(image, density="32k", method="nearest"):
        left = np.zeros(3, dtype=np.int16)
        right = np.zeros(3, dtype=np.int16)
        # Batched projection has no labelled vertices (simulating nearby markers
        # competing in a common marker volume). Individual retries recover one.
        if len(image["site_ids"]) == 1 and image["radius"] >= 10:
            left[1] = 1
        return projection.SurfaceProjection(left=left, right=right)

    monkeypatch.setattr(projection, "_load_inflated_vertices", fake_load_vertices)
    monkeypatch.setattr(projection, "_make_marker_volume", fake_make_marker_volume)
    monkeypatch.setattr(projection, "_map_mni_to_fslr", fake_map)
    sites = pd.DataFrame({
        "site_id": ["Sx", "Sy"], "label": ["a", "b"],
        "x": [-30., -31.], "y": [10., 11.], "z": [2., 3.],
        "color": ["#ff0000", "#00ff00"], "hemisphere": ["left", "left"],
    })
    atlas = SimpleNamespace(left=np.array([1, 0, 0]), right=np.array([0, 0, 0]))
    _none, mapped = projection.project_sites_to_fslr(
        reference_img=object(), sites=sites, radius_mm=5.0,
        fallback_radii_mm=(7.5,), atlas_surface=atlas,
        snap_to_insula=True, max_snap_distance_mm=6.0,
        last_resort_radii_mm=(10.0, 15.0),
    )
    assert (mapped.surface_vertex >= 0).all()
    # A valid cortical candidate is now snapped to the nearest Julich insula
    # vertex even when the historical maximum snap distance would be exceeded.
    assert mapped.projection_method.str.contains("nearest_julich_insula_surface").all()
    assert mapped.surface_in_julich_mask.all()
    assert (mapped.surface_snap_distance_mm >= 0).all()
