"""Discovery of the upstream capsule outputs this library repackages.

This is the pipeline's terminal fan-in: ``main.nf`` calls the NWB capsule once
with every upstream channel ``.collect()``ed, so one run sees every plane's
outputs from motion correction, decrosstalk, extraction, classification, dF/F
and OASIS event detection at the same time.

Each of those capsules writes ``<plane_id>/<step>/<plane_id>_<artifact>``
under its results root, and the pipeline copies all of them into one
``processed`` mount. v2 changed the *directory* nesting but kept the
``<plane_id>_`` filename prefix, so this module globs recursively on the
filename rather than walking a fixed directory layout.

Kept import-light: every function here is pure path logic, so the upstream
contract stays testable without pynwb or real HDF5 payloads.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from aind_pophys_metadata import find_one

logger = logging.getLogger(__name__)

# Filename suffix of every upstream artifact the NWB builders actually read.
#
# Only what is consumed is listed. Notably absent is ``processing.json``: v2
# capsules write it unprefixed, so globbing for it would match every plane's
# and no builder reads it anyway.
ARTIFACT_PARTS: Dict[str, str] = {
    # aind-ophys-extraction -> <plane>/extraction/<plane>_extraction.h5
    "extraction_h5": "extraction.h5",
    # aind-ophys-dff -> <plane>/dff/<plane>_dff.h5
    "dff_h5": "dff.h5",
    # aind-ophys-oasis-event-detection -> <plane>/events/<plane>_events_oasis.h5
    "events_oasis_h5": "events_oasis.h5",
    # aind-ophys-classifier -> <plane>/classification/<plane>_classification.h5
    "classification_h5": "classification.h5",
    # aind-ophys-motion-correction -> <plane>/motion_correction/<plane>_*.png
    "average_projection_png": "average_projection.png",
    "maximum_projection_png": "maximum_projection.png",
}

# Traces and ROI masks come from the extraction file; without it there is
# nothing to package. Everything else degrades to an omitted NWB container.
REQUIRED_ARTIFACTS: Tuple[str, ...] = ("extraction_h5",)

EPOCH_LOCATIONS_JSON = "epoch_locations.json"


@dataclass(frozen=True)
class PlaneArtifacts:
    """Resolved upstream artifact paths for one imaging plane.

    Only ``extraction_h5`` is guaranteed non-``None``; the rest are optional
    because a plane can legitimately be missing dF/F, events, classification
    or projections (an upstream stage was skipped or failed) and the NWB should
    still carry what does exist.
    """

    plane_id: str
    extraction_h5: Path
    dff_h5: Optional[Path] = None
    events_oasis_h5: Optional[Path] = None
    classification_h5: Optional[Path] = None
    average_projection_png: Optional[Path] = None
    maximum_projection_png: Optional[Path] = None


def find_artifact(
    search_root: Path,
    part: str,
    plane_id: Optional[str] = None,
) -> Optional[Path]:
    """Return the first recursive match for an upstream artifact.

    Parameters
    ----------
    search_root : Path
        Directory searched recursively (the ``processed`` mount).
    part : str
        Artifact filename suffix, e.g. ``"extraction.h5"``.
    plane_id : str, optional
        Canonical plane id. When given, only files whose name starts with
        ``f"{plane_id}_"`` match, which is what keeps two planes' artifacts
        from being confused. When ``None`` the prefix is not required.

    Returns
    -------
    Path or None
        The first match in sorted order, or ``None`` when there is none.
    """
    matches = find_artifacts(search_root, part, plane_id)
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "%s matches for '%s%s' under %s; using %s",
            len(matches),
            f"{plane_id}_" if plane_id else "",
            part,
            search_root,
            matches[0],
        )
    return matches[0]


def find_artifacts(
    search_root: Path,
    part: str,
    plane_id: Optional[str] = None,
) -> List[Path]:
    """Return every match for an artifact suffix, in sorted order.

    Parameters
    ----------
    search_root : Path
        Directory searched recursively.
    part : str
        Artifact filename suffix, e.g. ``"extraction.h5"``.
    plane_id : str, optional
        Canonical plane id. When given, only files whose name starts with
        ``f"{plane_id}_"`` match. When ``None`` the prefix is not required.

    Returns
    -------
    list of Path
        All matches, sorted for determinism; empty when there are none.
    """
    prefix = f"{plane_id}_" if plane_id else ""
    return sorted(Path(search_root).rglob(f"{prefix}*{part}"))


def collect_plane_artifacts(
    search_root: Path,
    plane_id: str,
    *,
    allow_unprefixed: bool = False,
) -> PlaneArtifacts:
    """Resolve every upstream artifact for one plane.

    Parameters
    ----------
    search_root : Path
        The ``processed`` mount, searched recursively.
    plane_id : str
        Canonical plane id (``"VISp_0"`` multi-plane, ``"plane_0"`` single).
    allow_unprefixed : bool, optional
        When ``True``, fall back to an unprefixed match if the
        ``<plane_id>_`` prefixed glob finds nothing. Only safe for a
        single-plane session, where the upstream ``unique_id`` is derived from
        the movie filename and therefore need not equal ``plane_0``.

    Returns
    -------
    PlaneArtifacts
        The resolved paths.

    Raises
    ------
    FileNotFoundError
        If a required artifact cannot be located; the message names the glob
        pattern and the directory searched, because a bare ``StopIteration``
        or ``None`` is the least debuggable failure in the pipeline.
    """
    resolved: Dict[str, Optional[Path]] = {}
    for key, part in ARTIFACT_PARTS.items():
        match = find_artifact(search_root, part, plane_id)
        if match is None and allow_unprefixed:
            # The prefix is the only thing tying an artifact to a plane, so
            # dropping it is safe ONLY when the mount is unambiguous. If more
            # than one candidate exists the metadata and the mount disagree
            # about how many planes there are, and picking sorted()[0] would
            # publish one plane's data under another plane's name while
            # exiting 0 -- the worst possible outcome for a terminal stage.
            candidates = find_artifacts(search_root, part)
            if len(candidates) > 1:
                raise ValueError(
                    f"The acquisition metadata resolved a single plane "
                    f"({plane_id}), but {len(candidates)} files match "
                    f"'*{part}' under {search_root}: "
                    f"{', '.join(str(c) for c in candidates)}. Refusing to "
                    f"guess which one is {plane_id} -- publishing another "
                    f"plane's data under this name would be silent. Check "
                    f"whether the acquisition metadata is missing planes."
                )
            match = candidates[0] if candidates else None
            if match is not None:
                logger.info(
                    "Resolved '%s' for %s without the plane prefix: %s",
                    part,
                    plane_id,
                    match,
                )
        if match is None:
            if key in REQUIRED_ARTIFACTS:
                raise FileNotFoundError(
                    f"Required upstream artifact '{plane_id}_*{part}' not "
                    f"found under {search_root}. NWB packaging is the "
                    f"terminal stage, so the producing capsule either did "
                    f"not run for this plane or wrote a different filename."
                )
            logger.warning(
                "No match for '%s_*%s' under %s; the corresponding NWB "
                "container will be omitted for this plane.",
                plane_id,
                part,
                search_root,
            )
        resolved[key] = match
    return PlaneArtifacts(plane_id=plane_id, **resolved)


def collect_session_artifacts(
    search_root: Path,
    plane_ids: Sequence[str],
) -> List[PlaneArtifacts]:
    """Resolve upstream artifacts for every plane in the session.

    Parameters
    ----------
    search_root : Path
        The ``processed`` mount, searched recursively.
    plane_ids : sequence of str
        Canonical plane ids in acquisition order.

    Returns
    -------
    list of PlaneArtifacts
        One entry per plane, in the same order as ``plane_ids``.

    Raises
    ------
    ValueError
        If ``plane_ids`` is empty -- there would be nothing to package, and
        silently writing an empty NWB hides a metadata read failure.
    """
    if not plane_ids:
        raise ValueError(
            "No imaging planes resolved from the acquisition metadata; "
            "refusing to write an empty NWB file."
        )
    single_plane = len(plane_ids) == 1
    return [
        collect_plane_artifacts(
            search_root,
            plane_id,
            allow_unprefixed=single_plane,
        )
        for plane_id in plane_ids
    ]


def find_epoch_locations(search_root: Path) -> Optional[Path]:
    """Locate an ``epoch_locations.json`` under the processed mount.

    Written by motion correction for single-plane sessions that carry
    photostim/spontaneous epochs.

    Parameters
    ----------
    search_root : Path
        The ``processed`` mount, searched recursively.

    Returns
    -------
    Path or None
        The first match, or ``None`` when the session has no epochs.
    """
    return find_one(search_root, EPOCH_LOCATIONS_JSON, required=False)


def find_sync_file(raw_root: Path) -> Optional[Path]:
    """Locate the raw sync HDF5 file for a multiplane session.

    Patterns are tried in order -- ``*pophys/data_main.h5``, then
    ``behavior/*.h5``, then any ``*ophys/*.h5`` -- matching the layout the
    transfer service produces for each acquisition style.

    Parameters
    ----------
    raw_root : Path
        The ``raw`` mount, searched recursively.

    Returns
    -------
    Path or None
        The sync file, or ``None`` when none of the patterns match.
    """
    patterns = ("*pophys/data_main.h5", "behavior/*.h5", "*ophys/*.h5")
    for pattern in patterns:
        match = find_one(raw_root, pattern, required=False)
        if match is not None:
            return match
    logger.error(
        "No sync file found under %s (tried %s)",
        raw_root,
        ", ".join(patterns),
    )
    return None
