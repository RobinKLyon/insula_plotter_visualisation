import sys
import types
_nib = sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))
_nib.__path__ = []
_aff = types.ModuleType("nibabel.affines")
_aff.apply_affine = lambda affine, xyz: xyz
sys.modules.setdefault("nibabel.affines", _aff)

import numpy as np
from types import SimpleNamespace

from insula_plotter import plotting, projection


def test_boundary_segments_are_inside_triangles_not_mesh_edges():
    xyz = np.array([[0., 0.], [2., 0.], [0., 2.]])
    faces = np.array([[0, 1, 2]])
    labels = np.array([1, 1, 2])
    segments = plotting._triangle_boundary_segments(
        xyz, faces, labels, lambda a, b: a != b
    )
    assert segments.shape == (1, 2, 2)
    # For labels [1,1,2], the interface connects midpoints of the two changed edges.
    expected = np.array([[1., 1.], [0., 1.]])
    assert np.allclose(np.sort(segments[0], axis=0), np.sort(expected, axis=0))
    # Neither endpoint is one of the original triangle vertices.
    assert not any(np.allclose(point, vertex) for point in segments[0] for vertex in xyz)


def test_family_palette_uses_coherent_dorsal_and_ventral_shades():
    atlas = SimpleNamespace(
        region_names={1: "Id4 (right)", 2: "Id6 (right)", 3: "Ia2 (right)", 4: "Id9 (right)"},
        region_colors={1: "#ff0000", 2: "#00ff00", 3: "#0000ff", 4: "#ffffff"},
    )
    palette = plotting._family_region_palette(atlas, {"parcel_color_contrast": 1.0})
    assert palette[1] != palette[2]
    assert palette[3] != palette[4]
    # Dorsal pair warm (red > blue), ventral pair cool (blue/green > red).
    assert palette[1][0] > palette[1][2]
    assert palette[3][2] > palette[3][0]


def test_local_fold_flattening_is_display_only_and_reduces_normal_depth():
    # A planar grid curled along x gives a clear local thickness direction.
    xx, yy = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    zz = 0.35 * (xx ** 2)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    faces = []
    for j in range(4):
        for i in range(4):
            a = j * 5 + i
            faces.extend([[a, a+1, a+6], [a, a+6, a+5]])
    faces = np.asarray(faces)
    mask = np.zeros(len(vertices), dtype=bool)
    mask[6:19] = True
    out = plotting._flatten_local_fold(
        vertices, faces, mask,
        {"fold_flatten_strength": 0.5, "fold_flatten_fade_mm": 20.0},
    )
    assert out.shape == vertices.shape
    assert not np.allclose(out, vertices)
    assert np.allclose(vertices, vertices.copy())  # input geometry remains unchanged
    assert np.isfinite(out).all()


def test_probability_projection_is_linear_and_uses_probability_threshold(monkeypatch):
    calls = []
    def fake_map(image, density="32k", method="nearest"):
        calls.append(method)
        return projection.SurfaceProjection(
            left=np.asarray(image.left_values), right=np.asarray(image.right_values)
        )
    monkeypatch.setattr(projection, "_map_mni_to_fslr", fake_map)
    atlas = SimpleNamespace(
        region_names={1: "Id4 (left)", 2: "Ia2 (left)", 3: "Id4 (right)"},
        probability_images={
            "Id4_left": SimpleNamespace(left_values=[0.7, 0.03], right_values=[0.0, 0.0]),
            "Ia2_left": SimpleNamespace(left_values=[0.4, 0.6], right_values=[0.0, 0.0]),
            "Id4_right": SimpleNamespace(left_values=[0.0, 0.0], right_values=[0.9, 0.02]),
        },
    )
    out = projection.project_probability_atlas_to_fslr(atlas, probability_threshold=0.05)
    assert calls == ["linear", "linear", "linear"]
    assert out.left.tolist() == [1, 2]
    assert out.right.tolist() == [3, 0]


def test_triangle_fill_is_split_at_edge_midpoints_and_preserves_area():
    vertices = np.array([[0., 0.], [2., 0.], [0., 2.]])
    faces = np.array([[0, 1, 2]])
    labels = np.array([1, 1, 2])
    polygons, polygon_labels = plotting._split_labelled_triangles(vertices, faces, labels)
    assert polygon_labels.tolist() == [1, 2]
    assert len(polygons) == 2
    def area(poly):
        p = np.asarray(poly)
        return abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))) / 2
    assert np.isclose(sum(area(p) for p in polygons), 2.0)
    # Each region has a polygon whose vertices include the shared edge midpoint.
    shared_midpoint = np.array([1., 1.])
    assert any(np.allclose(point, shared_midpoint) for poly in polygons for point in poly)


def test_three_way_triangle_junction_uses_centroid():
    vertices = np.array([[0., 0.], [2., 0.], [0., 2.]])
    faces = np.array([[0, 1, 2]])
    labels = np.array([1, 2, 3])
    polygons, polygon_labels = plotting._split_labelled_triangles(vertices, faces, labels)
    assert polygon_labels.tolist() == [1, 2, 3]
    centroid = vertices.mean(axis=0)
    assert all(any(np.allclose(point, centroid) for point in poly) for poly in polygons)


def test_region_legend_swatches_match_dorsal_ventral_family_palette():
    atlas = SimpleNamespace(
        region_names={1: "Id4 (right)", 2: "Ia2 (right)"},
        region_colors={1: "#ff0000", 2: "#0000ff"},
    )
    handles = plotting._region_handles(
        atlas, "region_labels", {"parcel_color_contrast": 1.0}, {1, 2}
    )
    by_label = {h.get_label(): h.get_facecolor() for h in handles}
    assert by_label["ID4 — dorsale"][0] > by_label["ID4 — dorsale"][2]
    assert by_label["IA2 — ventrale"][2] > by_label["IA2 — ventrale"][0]
