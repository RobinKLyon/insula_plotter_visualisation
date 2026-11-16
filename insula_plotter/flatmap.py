"""Flatten an fsLR insular surface patch with a harmonic (Tutte) map.

This is a topology-preserving 2-D parameterisation for visualisation. It is not
an areal/metric-preserving atlas surface: some geometric distortion is expected.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


@dataclass
class FlatmapResult:
    xy: np.ndarray                 # (n_full_vertices, 2); NaN outside patch
    faces: np.ndarray              # faces in original full-surface vertex indices
    face_labels: np.ndarray        # Julich label id per flat face
    vertex_ids: np.ndarray         # original full-surface vertex ids used
    report: dict


def _unique_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    sorted_edges = np.sort(all_edges, axis=1)
    edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    return edges, counts


def _boundary_loops(boundary_edges: np.ndarray, vertices: np.ndarray) -> list[list[int]]:
    """Return boundary loops from an approximately manifold triangular patch."""
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    # If the border has a few non-manifold vertices, retain only degree-2 paths.
    unused = {tuple(sorted((int(a), int(b)))) for a, b in boundary_edges}
    loops: list[list[int]] = []
    while unused:
        edge = next(iter(unused))
        start, current = edge
        path = [start, current]
        unused.remove(edge)
        previous = start
        guard = 0
        while current != start and guard <= len(boundary_edges) + 5:
            guard += 1
            candidates = [n for n in adjacency[current] if n != previous and tuple(sorted((current, n))) in unused]
            if not candidates:
                break
            # A loop should have one continuation. If branching occurs, choose
            # the unused continuation with the shortest 3-D edge.
            nxt = min(candidates, key=lambda n: np.linalg.norm(vertices[n] - vertices[current]))
            unused.remove(tuple(sorted((current, nxt))))
            previous, current = current, nxt
            if current != start:
                path.append(current)
        if current == start and len(path) >= 3:
            loops.append(path)
    return loops


def _component_faces(faces: np.ndarray, vertex_count: int):
    """Connected components of the submesh used by the selected faces."""
    from scipy.sparse.csgraph import connected_components
    used = np.unique(faces)
    remap = np.full(vertex_count, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    local_faces = remap[faces]
    edges, _ = _unique_edges(local_faces)
    row = np.concatenate((edges[:, 0], edges[:, 1]))
    col = np.concatenate((edges[:, 1], edges[:, 0]))
    data = np.ones(len(row), dtype=np.int8)
    graph = coo_matrix((data, (row, col)), shape=(len(used), len(used))).tocsr()
    n_components, vertex_components = connected_components(graph, directed=False)
    face_components = vertex_components[local_faces[:, 0]]
    component_sizes = np.bincount(vertex_components, minlength=n_components)
    return face_components, component_sizes


def _flatten_connected_patch(vertices: np.ndarray, faces: np.ndarray):
    """Map one connected triangular patch to a disk using a positive uniform Laplacian."""
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    local_faces = remap[faces]
    local_vertices = vertices[used]

    edges, counts = _unique_edges(local_faces)
    boundary_edges = edges[counts == 1]
    loops = _boundary_loops(boundary_edges, local_vertices)
    if not loops:
        # A closed mesh has no natural boundary to cut (usually indicates that
        # a synthetic/test atlas labels the entire sphere). Fall back to PCA and
        # make this explicit in the output report; this is not a proper flattening.
        centered = local_vertices - local_vertices.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        local_xy = centered @ vt[:2].T
        scale = np.percentile(np.linalg.norm(local_xy, axis=1), 99)
        if scale > 0:
            local_xy /= scale
        full_xy = np.full((len(vertices), 2), np.nan, dtype=float)
        full_xy[used] = local_xy
        return full_xy, used, 0, 0, "pca_fallback_closed_mesh"

    # The longest perimeter is the exterior outline. Other loops (if any) are
    # treated as interior vertices and are filled by the harmonic solution.
    outer = max(loops, key=lambda loop: sum(
        np.linalg.norm(local_vertices[loop[(i + 1) % len(loop)]] - local_vertices[loop[i]])
        for i in range(len(loop))))
    boundary = np.asarray(outer, dtype=np.int64)

    # Start the boundary at its most anterior vertex; arc length determines the
    # circular boundary parameter, making the result reproducible and oriented.
    y = local_vertices[boundary, 1]
    start = int(np.argmax(y))
    boundary = np.roll(boundary, -start)
    loop_xyz = local_vertices[boundary]
    seg = np.linalg.norm(np.roll(loop_xyz, -1, axis=0) - loop_xyz, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    theta = 2.0 * np.pi * cumulative / max(float(seg.sum()), 1e-12)
    boundary_xy = np.column_stack((np.cos(theta), np.sin(theta)))

    # Uniform graph Laplacian: positive weights make the Tutte embedding stable
    # for a triangulated disk with a convex boundary.
    e01 = local_faces[:, [0, 1]]
    e12 = local_faces[:, [1, 2]]
    e20 = local_faces[:, [2, 0]]
    e = np.unique(np.sort(np.vstack((e01, e12, e20)), axis=1), axis=0)
    row = np.concatenate((e[:, 0], e[:, 1]))
    col = np.concatenate((e[:, 1], e[:, 0]))
    data = np.ones(len(row), dtype=float)
    adjacency = coo_matrix((data, (row, col)), shape=(len(used), len(used))).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    lap = diags(degree) - adjacency

    boundary_mask = np.zeros(len(used), dtype=bool)
    boundary_mask[boundary] = True
    interior = np.flatnonzero(~boundary_mask)
    local_xy = np.zeros((len(used), 2), dtype=float)
    local_xy[boundary] = boundary_xy
    if len(interior):
        lii = lap[interior][:, interior].tocsc()
        lib = lap[interior][:, boundary]
        rhs = -(lib @ boundary_xy)
        for dim in range(2):
            local_xy[interior, dim] = spsolve(lii, rhs[:, dim])

    # Rotate/reflection-align the map to the original anterior-superior plane.
    # This only sets display orientation; it does not change the parameterisation.
    target = local_vertices[:, [1, 2]].copy()
    target -= target.mean(axis=0, keepdims=True)
    source = local_xy - local_xy.mean(axis=0, keepdims=True)
    cross = source.T @ target
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rot = u @ vt
    local_xy = source @ rot
    # Standardise display orientation: anterior (+Y) to the right and dorsal (+Z) upward.
    corr_ap = np.corrcoef(local_xy[:, 0], local_vertices[:, 1])[0, 1] if np.std(local_xy[:, 0]) > 0 and np.std(local_vertices[:, 1]) > 0 else 1.0
    corr_dv = np.corrcoef(local_xy[:, 1], local_vertices[:, 2])[0, 1] if np.std(local_xy[:, 1]) > 0 and np.std(local_vertices[:, 2]) > 0 else 1.0
    if np.isfinite(corr_ap) and corr_ap < 0:
        local_xy[:, 0] *= -1
    if np.isfinite(corr_dv) and corr_dv < 0:
        local_xy[:, 1] *= -1
    scale = np.percentile(np.linalg.norm(local_xy, axis=1), 99)
    if scale > 0:
        local_xy /= scale

    full_xy = np.full((len(vertices), 2), np.nan, dtype=float)
    full_xy[used] = local_xy
    return full_xy, used, len(loops), len(outer), "harmonic_tutte"


def build_insula_flatmap(vertices: np.ndarray, faces: np.ndarray, vertex_labels: np.ndarray,
                         grow_rings: int = 1) -> FlatmapResult:
    """Create a 2-D flatmap from all Julich-labelled insular regions.

    One ring of neighbouring vertices is included as a neutral grey border,
    which closes small mesh gaps at the probabilistic atlas boundary. Faces are
    included when at least two of their vertices belong to the insular patch.
    The largest connected patch is retained; small disconnected fragments are
    reported explicitly rather than silently merged at arbitrary coordinates.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(vertex_labels, dtype=np.int64)
    seed = labels > 0
    roi = seed.copy()
    for _ in range(max(0, int(grow_rings))):
        edges, _ = _unique_edges(faces)
        frontier = np.unique(edges[roi[edges].any(axis=1)])
        roi[frontier] = True

    keep = np.count_nonzero(roi[faces], axis=1) >= 2
    patch_faces = faces[keep]
    if len(patch_faces) < 4:
        raise RuntimeError(f"Insular patch too small for flattening ({len(patch_faces)} triangles). Check atlas projection/threshold.")

    # Keep the largest connected mesh patch. Small disconnected atlas fragments
    # are counted in the report; this prevents invalid overlap in 2-D.
    face_comp, comp_sizes = _component_faces(patch_faces, len(vertices))
    valid_comps = np.flatnonzero(comp_sizes > 0)
    main_comp = int(valid_comps[np.argmax(comp_sizes[valid_comps])])
    main_faces = patch_faces[face_comp == main_comp]
    face_labels = np.zeros(len(main_faces), dtype=np.int64)
    tri_labs = labels[main_faces]
    for i, tri in enumerate(tri_labs):
        vals, counts = np.unique(tri[tri > 0], return_counts=True)
        if len(vals):
            face_labels[i] = int(vals[np.argmax(counts)])

    xy, used, n_loops, outer_n, method = _flatten_connected_patch(vertices, main_faces)
    # Area distortion summary compares each original 3-D triangle with its 2-D
    # counterpart. It is useful context when interpreting flatmap geometry.
    tri3 = vertices[main_faces]
    tri2 = xy[main_faces]
    area3 = 0.5 * np.linalg.norm(np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0]), axis=1)
    v1 = tri2[:, 1] - tri2[:, 0]
    v2 = tri2[:, 2] - tri2[:, 0]
    area2 = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    ratio = np.divide(area2, area3, out=np.full_like(area2, np.nan), where=area3 > 1e-12)
    finite = ratio[np.isfinite(ratio) & (ratio > 0)]
    # Flat coordinates are scaled to a convenient display size, so absolute
    # area ratios are arbitrary. Report ratios normalised by their median.
    reference = float(np.median(finite)) if len(finite) else np.nan
    relative = finite / reference if len(finite) and reference > 0 else np.array([], dtype=float)
    abs_log2 = np.abs(np.log2(relative)) if len(relative) else np.array([], dtype=float)
    report = {
        "n_patch_triangles": int(len(main_faces)),
        "n_patch_vertices": int(len(used)),
        "n_connected_components": int(len(valid_comps)),
        "n_boundary_loops": int(n_loops),
        "n_outer_boundary_vertices": int(outer_n),
        "flattening_method": method,
        "area_comparison_basis": "fsLR inflated surface (relative ratios; not native cortical area)",
        "n_discarded_component_triangles": int(len(patch_faces) - len(main_faces)),
        "median_relative_area_ratio": float(np.median(relative)) if len(relative) else np.nan,
        "p05_relative_area_ratio": float(np.percentile(relative, 5)) if len(relative) else np.nan,
        "p95_relative_area_ratio": float(np.percentile(relative, 95)) if len(relative) else np.nan,
        "p95_absolute_log2_area_distortion": float(np.percentile(abs_log2, 95)) if len(abs_log2) else np.nan,
    }
    return FlatmapResult(xy=xy, faces=main_faces, face_labels=face_labels,
                         vertex_ids=used, report=report)
