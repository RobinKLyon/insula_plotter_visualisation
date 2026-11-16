from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import nibabel as nib
import numpy as np


# Insular cytoarchitectonic areas available in the expanded Julich-Brain atlas
# (including the later-added Ia2, Ia3 and Id8-Id10 maps). Some siibra releases
# use expanded region names, so discovery checks abbreviations and full names.
REGION_ALIASES = {
    "Ig1": (r"\big1\b", r"granular.{0,25}area\s*1", r"area\s*1.{0,25}granular"),
    "Ig2": (r"\big2\b", r"granular.{0,25}area\s*2", r"area\s*2.{0,25}granular"),
    "Ig3": (r"\big3\b", r"granular.{0,25}area\s*3", r"area\s*3.{0,25}granular"),
    "Ia1": (r"\bia1\b", r"agranular.{0,25}area\s*1", r"area\s*1.{0,25}agranular"),
    "Ia2": (r"\bia2\b", r"agranular.{0,25}area\s*2", r"area\s*2.{0,25}agranular"),
    "Ia3": (r"\bia3\b", r"agranular.{0,25}area\s*3", r"area\s*3.{0,25}agranular"),
    "Id1": (r"\bid1\b", r"dysgranular.{0,25}area\s*1", r"area\s*1.{0,25}dysgranular"),
    "Id2": (r"\bid2\b", r"dysgranular.{0,25}area\s*2", r"area\s*2.{0,25}dysgranular"),
    "Id3": (r"\bid3\b", r"dysgranular.{0,25}area\s*3", r"area\s*3.{0,25}dysgranular"),
    "Id4": (r"\bid4\b", r"dysgranular.{0,25}area\s*4", r"area\s*4.{0,25}dysgranular"),
    "Id5": (r"\bid5\b", r"dysgranular.{0,25}area\s*5", r"area\s*5.{0,25}dysgranular"),
    "Id6": (r"\bid6\b", r"dysgranular.{0,25}area\s*6", r"area\s*6.{0,25}dysgranular"),
    "Id7": (r"\bid7\b", r"dysgranular.{0,25}area\s*7", r"area\s*7.{0,25}dysgranular"),
    "Id8": (r"\bid8\b", r"dysgranular.{0,25}area\s*8", r"area\s*8.{0,25}dysgranular"),
    "Id9": (r"\bid9\b", r"dysgranular.{0,25}area\s*9", r"area\s*9.{0,25}dysgranular"),
    "Id10": (r"\bid10\b", r"dysgranular.{0,25}area\s*10", r"area\s*10.{0,25}dysgranular"),
}

REGION_COLORS = {
    "Ig1": "#8DD3C7",
    "Ig2": "#80B1D3",
    "Ig3": "#B3DE69",
    "Ia1": "#FDB462",
    "Ia2": "#FB9A99",
    "Ia3": "#33A02C",
    "Id1": "#BC80BD",
    "Id2": "#FB8072",
    "Id3": "#BEBADA",
    "Id4": "#FCCDE5",
    "Id5": "#CCEBC5",
    "Id6": "#FFED6F",
    "Id7": "#A6CEE3",
    "Id8": "#FDBF6F",
    "Id9": "#CAB2D6",
    "Id10": "#1F78B4",
}


@dataclass
class AtlasResult:
    image: nib.spatialimages.SpatialImage
    region_names: dict[int, str]
    region_colors: dict[int, str]
    probability_images: dict[str, nib.spatialimages.SpatialImage]
    source_map_name: str


def _get_region_text(region) -> str:
    pieces = [getattr(region, "name", ""), str(region)]
    identifier = getattr(region, "id", None)
    if identifier:
        pieces.append(str(identifier))
    return " ".join(p for p in pieces if p).lower()


def _matches_code(region, code: str) -> bool:
    text = _get_region_text(region)
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in REGION_ALIASES[code])


def _hemisphere(region) -> str | None:
    text = _get_region_text(region)
    if re.search(r"\b(left|lh)\b|_left\b", text):
        return "left"
    if re.search(r"\b(right|rh)\b|_right\b", text):
        return "right"
    return None


def fetch_julich_insula_atlas(
    parcellation: str = "julich 3.1",
    space: str = "mni152",
    region_codes: list[str] | None = None,
    probability_threshold: float = 0.05,
) -> AtlasResult:
    """Fetch Julich statistical maps and build a winner-takes-all insula volume.

    The output is a MNI-space NIfTI label image. The selected maps are the
    probability maps, not a claim of deterministic cytoarchitectonic borders.
    """
    try:
        import siibra
    except ImportError as exc:
        raise RuntimeError(
            "siibra is required to download the Julich-Brain atlas. "
            "Install the project requirements first."
        ) from exc

    if region_codes is None:
        region_codes = list(REGION_ALIASES)
    unknown = [code for code in region_codes if code not in REGION_ALIASES]
    if unknown:
        raise ValueError(f"Unknown region codes: {unknown}. Available: {list(REGION_ALIASES)}")

    print(f"Fetching Julich-Brain statistical maps ({parcellation}, {space})...")
    with siibra.QUIET:
        maps = siibra.get_map(space=space, parcellation=parcellation, maptype="statistical")

    # siibra maps commonly contain separate volumes for each region and side.
    found: dict[tuple[str, str], object] = {}
    available = []
    for vol_index in range(len(maps)):
        region = maps.get_region(volume=vol_index)
        if region is None:
            continue
        available.append(_get_region_text(region))
        side = _hemisphere(region)
        if side is None:
            continue
        for code in region_codes:
            if _matches_code(region, code):
                found[(code, side)] = region

    missing = [(code, side) for code in region_codes for side in ("left", "right") if (code, side) not in found]
    if missing:
        print("Atlas areas not found in the selected siibra map:")
        for code, side in missing:
            print(f"  - {code} ({side})")
        print("Some versions of the Julich atlas do not expose every area in this map.")
        if not found:
            sample = sorted(set(available))[:80]
            raise RuntimeError(
                "Could not identify any requested insular regions in the siibra map.\n"
                "Run `python scripts/list_atlas_regions.py` to inspect available names.\n"
                f"First available region names: {sample}"
            )

    # Fetch probability images and verify that they all share the same MNI grid.
    probability_images: dict[str, nib.spatialimages.SpatialImage] = {}
    probability_arrays: dict[str, np.ndarray] = {}
    reference_img = None
    for code in region_codes:
        for side in ("left", "right"):
            region = found.get((code, side))
            if region is None:
                continue
            with siibra.QUIET:
                pimg = maps.fetch(region=region)
            if not isinstance(pimg, nib.spatialimages.SpatialImage):
                raise TypeError(f"Expected a NIfTI probability map for {code} {side}; got {type(pimg)}")
            if reference_img is None:
                reference_img = pimg
            elif pimg.shape != reference_img.shape or not np.allclose(pimg.affine, reference_img.affine, atol=1e-3):
                raise RuntimeError(
                    f"Julich map grid differs for {code} {side}. The project currently expects one common MNI grid."
                )
            key = f"{code}_{side}"
            probability_images[key] = pimg
            probability_arrays[key] = np.nan_to_num(pimg.get_fdata(dtype=np.float32), nan=0.0)
            print(f"  loaded {key}")

    if reference_img is None or not probability_arrays:
        raise RuntimeError("No usable Julich probability images were downloaded.")

    # Construct one non-overlapping categorical image from the selected regions.
    # At each voxel, assign the region with the highest probability, only if it
    # reaches probability_threshold.
    best_prob = np.zeros(reference_img.shape, dtype=np.float32)
    labels = np.zeros(reference_img.shape, dtype=np.uint16)
    label_to_name: dict[int, str] = {}
    label_to_color: dict[int, str] = {}
    next_label = 1
    for code in region_codes:
        for side in ("left", "right"):
            key = f"{code}_{side}"
            if key not in probability_arrays:
                continue
            prob = probability_arrays[key]
            update = (prob > best_prob) & (prob >= float(probability_threshold))
            labels[update] = next_label
            best_prob[update] = prob[update]
            label_to_name[next_label] = f"{code} ({side})"
            label_to_color[next_label] = REGION_COLORS[code]
            next_label += 1

    header = reference_img.header.copy()
    header.set_data_dtype(np.uint16)
    label_img = nib.Nifti1Image(labels, reference_img.affine, header=header)
    label_img.set_qform(reference_img.affine, code=1)
    label_img.set_sform(reference_img.affine, code=1)

    return AtlasResult(
        image=label_img,
        region_names=label_to_name,
        region_colors=label_to_color,
        probability_images=probability_images,
        source_map_name=str(maps),
    )
