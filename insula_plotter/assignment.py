from __future__ import annotations

import numpy as np
import nibabel as nib
import pandas as pd
from nibabel.affines import apply_affine


def assign_sites_to_julich_regions(sites: pd.DataFrame, atlas_result) -> pd.DataFrame:
    """Sample the fetched Julich probability maps at each MNI coordinate."""
    out = sites.copy()
    top_names, top_probs, all_probs = [], [], []

    for row in out.itertuples(index=False):
        xyz = np.array([row.x, row.y, row.z], dtype=float)
        best_name = "Outside / no Julich probability"
        best_prob = 0.0
        site_probs = {}
        for key, img in atlas_result.probability_images.items():
            # Only compare maps from the same hemisphere as the electrode.
            if not key.endswith(f"_{row.hemisphere}"):
                continue
            inv_affine = np.linalg.inv(img.affine)
            ijk = np.rint(apply_affine(inv_affine, xyz)).astype(int)
            if np.any(ijk < 0) or np.any(ijk >= np.asarray(img.shape[:3])):
                probability = 0.0
            else:
                probability = float(np.nan_to_num(np.asarray(img.dataobj[tuple(ijk)]), nan=0.0))
            site_probs[key] = probability
            if probability > best_prob:
                best_prob = probability
                best_name = key.replace("_", " ")
        top_names.append(best_name)
        top_probs.append(best_prob)
        all_probs.append(site_probs)

    out["atlas_region_top"] = top_names
    out["atlas_probability_top"] = top_probs
    for key in sorted({key for row_dict in all_probs for key in row_dict}):
        out[f"p_{key}"] = [row_dict.get(key, 0.0) for row_dict in all_probs]
    return out
