"""Metadata reads for NWB packaging, on top of the shared read layer.

:class:`aind_pophys_metadata.CoreMetadata` owns file discovery, schema-version
dispatch, the canonical plane ids and every per-plane imaging field; this module
adds only what is NWB-specific: the ``ImagingPlane.location`` string, the
indicator read from ``procedures.json``, the session-type fork and the
``Processing`` builders for this capsule.

Two invariants shape the design:

* **No schema version is compared here.** ``CoreMetadata`` answers the same
  questions for v1, v2 and minimal input alike, so nothing downstream learns
  which document it was given.
* **Reads are raw dicts.** Nothing here calls ``model_validate`` on an input
  file; messy-but-sufficient metadata must still package.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aind_data_schema.components.identifiers import Code, DataAsset
from aind_pophys_metadata import CoreMetadata, find_one, io
from aind_pophys_metadata import processing as shared_processing


logger = logging.getLogger(__name__)

#: ``Code.name``, and the ``process_name`` logging context. The matching
#: ``Code.url`` and ``Code.version`` are derived by the shared builder from
#: :data:`LIBRARY_NAME` alone, never from this string.
PROCESS_NAME = "aind-ophys-nwb"

#: Distribution name of this backing library, used by the shared builder to
#: resolve a version from installed package metadata when none is supplied.
LIBRARY_NAME = "aind-ophys-nwb-library"

PROCEDURES_FILE = "procedures.json"

#: ``Code.url``. Only this repo knows where its own code lives, so the shared
#: builder no longer derives it from :data:`LIBRARY_NAME`.
CODE_URL = "https://github.com/AllenNeuralDynamics/aind-ophys-nwb-library"

#: Key added to each shared plane record, pairing it with its canonical id.
PLANE_ID_KEY = "plane_id"

#: ``DataProcess.name``. Set explicitly because the converter also uses
#: ``ProcessName.FILE_FORMAT_CONVERSION``, so both would otherwise derive the
#: same default name ("File format conversion") and the run-level aggregator
#: rejects a merged document with duplicate process names. Both capsules run
#: once per session, so a plane id cannot separate them.
DATA_PROCESS_NAME = "NWB packaging"


class SessionType(Enum):
    """How the acquisition is packaged into NWB.

    The fork is decided by **plane count**, not by
    ``data_description.platform`` (a v1-only field that v2 removed) and not by
    ``instrument_id``. Plane count is also what
    :func:`aind_pophys_metadata.build_fov_ids` uses to choose the
    plane-naming scheme, so routing and naming agree by construction.
    """

    MULTIPLANE = "multiplane"
    SINGLE_PLANE_WITH_EPOCHS = "single_plane_with_epochs"
    SINGLE_PLANE = "single_plane"


def plane_location(plane: Dict[str, Any]) -> str:
    """Render one plane's NWB ``ImagingPlane.location`` description string.

    Parameters
    ----------
    plane : dict
        One shared plane record.

    Returns
    -------
    str
        The location string; missing fields read as ``unknown`` rather than
        rendering ``None`` into the NWB file.
    """
    structure = plane.get("structure") or "unknown"
    depth = plane.get("depth")
    return f"Structure: {structure} Depth: {'unknown' if depth is None else depth}"


@dataclass(frozen=True)
class NwbMetadata:
    """Minimal metadata the NWB packaging capsule consumes.

    ``planes`` and ``frame_rate`` are the only required fields: the planes name
    the NWB processing modules and drive the upstream artifact globs, and the
    frame rate sets every ``ImagingPlane.imaging_rate`` (and the single-plane
    trace sampling rate, where there is no sync file). The identifiers are
    logging context and stay optional. ``schema_version`` is provenance for
    logs and error messages; nothing branches on it.

    Each entry of ``planes`` is a shared plane record dict carrying the extra
    ``plane_id`` key this library pairs onto it, so the records are ordered by
    ``plane_index`` and their ids are the canonical ones.
    """

    schema_version: str
    core_file: Path
    planes: Tuple[Dict[str, Any], ...]
    frame_rate: float
    instrument_id: Optional[str] = None
    excitation_wavelength: Optional[float] = None
    emission_wavelength: Optional[float] = None
    indicator: Optional[str] = None
    genotype: Optional[str] = None
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None

    @property
    def plane_ids(self) -> Tuple[str, ...]:
        """Return the canonical plane ids in acquisition order."""
        return tuple(plane[PLANE_ID_KEY] for plane in self.planes)

    @property
    def single_plane(self) -> bool:
        """Return whether the acquisition has exactly one imaging plane."""
        return len(self.planes) == 1


def _material_name(node: dict) -> Optional[str]:
    """Return the first named entry of ``node["injection_materials"]``.

    Parameters
    ----------
    node : dict
        A raw procedures sub-document.

    Returns
    -------
    str or None
        The material name, or ``None`` when the key is absent or unnamed.
    """
    for material in node.get("injection_materials") or []:
        if isinstance(material, dict) and material.get("name"):
            return str(material["name"])
    return None


def _walk_for_material(node: Any) -> Optional[str]:
    """Return the first injection-material name at or below ``node``.

    Parameters
    ----------
    node : Any
        Any node of the raw procedures document.

    Returns
    -------
    str or None
        The material name, or ``None`` when none is present in this subtree.
    """
    if isinstance(node, dict):
        found = _material_name(node)
        if found is not None:
            return found
        children: Any = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _walk_for_material(child)
        if found is not None:
            return found
    return None


def get_injection_material(procedures_raw: Optional[dict]) -> Optional[str]:
    """First injected material name found anywhere in ``procedures.json``.

    Searched recursively rather than at the v1 path
    ``subject_procedures[0].procedures[*].injection_materials[0].name``,
    because v2 moved the procedure content under ``components`` and re-nested
    it; a recursive search reads both shapes without a version branch.

    Parameters
    ----------
    procedures_raw : dict, optional
        Raw ``procedures.json`` dict, or ``None`` when absent.

    Returns
    -------
    str or None
        The material name (e.g. the GCaMP variant), or ``None``.
    """
    if not procedures_raw:
        return None
    return _walk_for_material(procedures_raw)


def session_type(plane_count: int, has_epochs: bool) -> SessionType:
    """Classify the acquisition from plane count and epoch presence.

    Parameters
    ----------
    plane_count : int
        Number of imaging planes resolved from the acquisition metadata.
    has_epochs : bool
        Whether an ``epoch_locations.json`` was found in the processed mount.

    Returns
    -------
    SessionType
        ``MULTIPLANE`` when more than one plane, otherwise
        ``SINGLE_PLANE_WITH_EPOCHS`` or ``SINGLE_PLANE``.
    """
    if plane_count > 1:
        return SessionType.MULTIPLANE
    return SessionType.SINGLE_PLANE_WITH_EPOCHS if has_epochs else SessionType.SINGLE_PLANE


def identified_planes(core: CoreMetadata) -> Tuple[Dict[str, Any], ...]:
    """Order the shared plane records and pair each with its canonical id.

    ``get_plane_records`` is unsorted and carries no id, while
    ``get_fov_ids`` is ordered by plane index, so the two are zipped after
    sorting.

    Parameters
    ----------
    core : aind_pophys_metadata.CoreMetadata
        The located core metadata document.

    Returns
    -------
    tuple of dict
        One record per plane, ordered by ``plane_index`` and each carrying an
        added ``plane_id``.
    """
    records = sorted(core.get_plane_records(), key=lambda record: record["plane_index"])
    return tuple(
        {**record, PLANE_ID_KEY: plane_id}
        for record, plane_id in zip(records, core.get_fov_ids())
    )


def load_nwb_metadata(raw_dir: Path) -> NwbMetadata:
    """Read metadata for NWB packaging from the raw mount.

    The shared layer locates the core file, normalizes it and falls back to
    ``platform.json`` for the frame rate when the core file omits it.

    Parameters
    ----------
    raw_dir : Path
        Directory searched recursively for the core metadata files.

    Returns
    -------
    NwbMetadata
        The minimal fields this capsule consumes.

    Raises
    ------
    FileNotFoundError
        If no core metadata file is found.
    ValueError
        If the input is ambiguous, if no imaging plane can be resolved, or if
        no frame rate can be resolved from any source.
    """
    raw_dir = Path(raw_dir)
    core = CoreMetadata.load(raw_dir)
    procedures_path = find_one(raw_dir, PROCEDURES_FILE, required=False)
    procedures_raw = io.load_json(procedures_path) if procedures_path else None

    planes = identified_planes(core)
    if not planes:
        raise ValueError(
            f"No imaging planes found in {core.core_path.name} (schema "
            f"{core.version}); NWB packaging has nothing to write."
        )

    # Validated here rather than ~150 lines later in job.py: every NWB imaging
    # plane needs an imaging_rate, so an asset without one cannot be packaged
    # at all and should fail at load, naming the file and schema version.
    frame_rate = core.get_frame_rate(required=True)

    genotype = (core.subject_raw or {}).get("genotype")
    indicator = get_injection_material(procedures_raw) or genotype
    return NwbMetadata(
        schema_version=str(core.version),
        core_file=core.core_path,
        planes=planes,
        frame_rate=frame_rate,
        instrument_id=core.get_instrument_id(),
        excitation_wavelength=core.get_excitation_wavelength(),
        emission_wavelength=core.get_emission_wavelength(),
        indicator=indicator,
        genotype=genotype,
        subject_id=core.get_subject_id(),
        dataset_name=core.get_dataset_name(),
    )


def build_nwb_code(parameters: dict) -> Code:
    """Construct the v2 ``Code`` block for NWB packaging.

    ``input_data`` is not set here: the upstream artifacts are only resolved
    once the run is under way, while the code block must exist before it so
    that a failure still yields a ``processing.json``. Attach them later with
    :func:`attach_input_data`.

    Parameters
    ----------
    parameters : dict
        Resolved run parameters recorded on the code block.

    Returns
    -------
    Code
        The populated code block.
    """
    return shared_processing.build_code(
        name=PROCESS_NAME,
        library_name=LIBRARY_NAME,
        url=CODE_URL,
        parameters=parameters,
    )


def attach_input_data(code: Code, input_data: List[str]) -> None:
    """Record resolved input artifacts on an already-built ``Code``.

    Parameters
    ----------
    code : Code
        The code block to populate.
    input_data : list of str
        Input artifact path(s)/asset name(s), relative to their mount root.
    """
    code.input_data = [DataAsset(name=name) for name in input_data]
