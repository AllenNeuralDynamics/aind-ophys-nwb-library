"""aind-data-schema v1/v2 metadata reads for NWB packaging.

A capsule-specific layer over the shared ``aind_pophys_metadata`` library. The
shared layer owns file discovery, schema-version dispatch, the canonical plane
ids and the ``Processing`` builders; this module adds only the *rich per-plane
imaging fields* that NWB needs and that the shared ``fields`` module does not
expose (depth, FOV scale factor and units, coupled-plane index, FOV width and
height, excitation and emission wavelengths).

Two invariants shape the design:

* **Plane ids come from the shared layer, never re-derived.** They name the
  output folders every upstream capsule wrote, so
  :func:`aind_pophys_metadata.fields.get_fov_ids` is the single source of
  truth. The rich per-plane records here are built with the *same traversal
  order and the same sort key* and then paired to those ids **by position**,
  which is the documented cross-capsule contract.
* **Reads are raw dicts.** Nothing here calls ``model_validate`` on an input
  file; messy-but-sufficient metadata must still package.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple

from aind_data_schema.components.configs import (
    ImagingConfig,
    LaserConfig,
    PlanarImage,
    PlanarImageStack,
)
from aind_data_schema.components.coordinates import Scale
from aind_data_schema.components.identifiers import Code, DataAsset
from aind_pophys_metadata import fields, io
from aind_pophys_metadata import processing as shared_processing
from aind_pophys_metadata.fields import _as_int
from aind_pophys_metadata.paths import find_one


logger = logging.getLogger(__name__)

#: Landing page of the capsule wrapper this library is deployed as. No
#: ``/tree/main/code`` suffix and no branch ref: the shape every v2 capsule
#: records, so one URL identifies the repository rather than a moving ref.
PROCESS_NAME = "aind-ophys-nwb"

#: Distribution name of this backing library, used by the shared builder to
#: resolve a version from installed package metadata when none is supplied.
LIBRARY_NAME = "aind-ophys-nwb-library"

PROCEDURES_FILE = "procedures.json"

#: ``DataProcess.name``. Set explicitly because the converter also uses
#: ``ProcessName.FILE_FORMAT_CONVERSION``, so both would otherwise derive the
#: same default name ("File format conversion") and the run-level aggregator
#: rejects a merged document with duplicate process names. Both capsules run
#: once per session, so a plane id cannot separate them.
DATA_PROCESS_NAME = "NWB packaging"

#: Unit assumed when the acquisition metadata omits a length unit.
DEFAULT_LENGTH_UNIT = "micrometer"

# v2 discriminators read off the schema classes rather than hardcoded, so they
# track the installed aind-data-schema version.
_IMAGING_CONFIG_TYPE = io.object_type_value(ImagingConfig)
_PLANAR_IMAGE_TYPES = {
    io.object_type_value(PlanarImage),
    io.object_type_value(PlanarImageStack),
}
_LASER_CONFIG_TYPE = io.object_type_value(LaserConfig)
_SCALE_TYPE = io.object_type_value(Scale)


class SessionType(Enum):
    """How the acquisition is packaged into NWB.

    The fork is decided by **plane count**, not by
    ``data_description.platform`` (a v1-only field that v2 removed) and not by
    ``instrument_id``. Plane count is also what
    :func:`aind_pophys_metadata.fields.build_fov_ids` uses to choose the
    plane-naming scheme, so routing and naming agree by construction.
    """

    MULTIPLANE = "multiplane"
    SINGLE_PLANE_WITH_EPOCHS = "single_plane_with_epochs"
    SINGLE_PLANE = "single_plane"


@dataclass(frozen=True)
class PlaneMetadata:
    """Per-plane imaging metadata the NWB builders consume.

    Only ``plane_id`` and ``plane_index`` are required. Everything else is
    optional because it feeds descriptive NWB fields (an imaging-plane
    ``location`` string, a grid spacing, an image resolution) rather than the
    conversion itself, and a partial-metadata asset must still package.
    """

    plane_id: str
    plane_index: int
    structure: Optional[str] = None
    depth: Optional[float] = None
    depth_unit: str = DEFAULT_LENGTH_UNIT
    coupled_plane_index: Optional[int] = None
    scale_factor: Optional[float] = None
    coordinate_unit: str = DEFAULT_LENGTH_UNIT
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def location(self) -> str:
        """Return the NWB ``ImagingPlane.location`` description string."""
        structure = self.structure if self.structure else "unknown"
        depth = "unknown" if self.depth is None else f"{self.depth}"
        return f"Structure: {structure} Depth: {depth}"


@dataclass(frozen=True)
class NwbMetadata:
    """Minimal metadata the NWB packaging capsule consumes.

    ``planes`` and ``frame_rate`` are the only required fields: the planes name
    the NWB processing modules and drive the upstream artifact globs, and the
    frame rate sets every ``ImagingPlane.imaging_rate`` (and the single-plane
    trace sampling rate, where there is no sync file). The identifiers are
    logging context and stay optional.
    """

    schema_version: str
    core_file: Path
    planes: Tuple[PlaneMetadata, ...]
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
        return tuple(plane.plane_id for plane in self.planes)

    @property
    def single_plane(self) -> bool:
        """Return whether the acquisition has exactly one imaging plane."""
        return len(self.planes) == 1


def _as_float(value: Any) -> Optional[float]:
    """Coerce a raw metadata value to float, or return ``None``.

    Parameters
    ----------
    value : Any
        Raw value from the metadata dict; strings are common in v1.

    Returns
    -------
    float or None
        The coerced value, or ``None`` when absent or uncoercible.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric metadata value %r", value)
        return None


def _imaging_configs(acquisition: dict) -> List[dict]:
    """Return every v2 ``ImagingConfig`` dict in the acquisition.

    Parameters
    ----------
    acquisition : dict
        Raw ``acquisition.json`` dict.

    Returns
    -------
    list of dict
        Imaging configs, in ``data_streams`` then ``configurations`` order.
    """
    configs = []
    for stream in acquisition.get("data_streams") or []:
        for config in stream.get("configurations") or []:
            if config.get("object_type") == _IMAGING_CONFIG_TYPE:
                configs.append(config)
    return configs


def _first_scale(transforms: Any) -> Optional[float]:
    """Return the first axis of the first ``Scale`` in a v2 transform list.

    v1 carried a scalar ``fov_scale_factor`` (micrometres per pixel). v2
    replaces it with the image-to-acquisition transform chain, whose ``Scale``
    entry holds the same information per axis; NWB needs one isotropic number.

    Parameters
    ----------
    transforms : Any
        Raw ``image_to_acquisition_transform`` list.

    Returns
    -------
    float or None
        The first scale component, or ``None`` when no ``Scale`` is present.
    """
    for transform in transforms or []:
        if not isinstance(transform, dict):
            continue
        if transform.get("object_type") != _SCALE_TYPE:
            continue
        scale = transform.get("scale") or []
        if scale:
            return _as_float(scale[0])
    return None


def plane_records_v1(session: dict) -> List[PlaneMetadata]:
    """Rich per-plane records from a v1 ``session.json``.

    Traverses ``data_streams[*].ophys_fovs[*]`` in the same order as
    :func:`aind_pophys_metadata.fields.get_fov_pairs_v1`, so the results pair
    positionally with the shared layer's canonical plane ids.

    ``plane_id`` is left as the empty string here and filled in by
    :func:`attach_plane_ids`.

    Parameters
    ----------
    session : dict
        Raw ``session.json`` dict.

    Returns
    -------
    list of PlaneMetadata
        Unsorted per-plane records.
    """
    records: List[PlaneMetadata] = []
    for stream in session.get("data_streams") or []:
        for fov in stream.get("ophys_fovs") or []:
            coordinate_unit = (
                fov.get("fov_coordinate_unit")
                or fov.get("fov_scale_factor_unit")
                or DEFAULT_LENGTH_UNIT
            )
            records.append(
                PlaneMetadata(
                    plane_id="",
                    plane_index=_as_int(fov.get("index")) or 0,
                    structure=fields.acronym_from_targeted_structure(fov.get("targeted_structure")),
                    depth=_as_float(fov.get("imaging_depth")),
                    depth_unit=fov.get("imaging_depth_unit") or DEFAULT_LENGTH_UNIT,
                    coupled_plane_index=_as_int(fov.get("coupled_fov_index")),
                    scale_factor=_as_float(fov.get("fov_scale_factor")),
                    coordinate_unit=coordinate_unit,
                    width=_as_int(fov.get("fov_width")),
                    height=_as_int(fov.get("fov_height")),
                )
            )
    return records


def plane_records_v2(acquisition: dict) -> List[PlaneMetadata]:
    """Rich per-plane records from a v2 ``acquisition.json``.

    Traverses ``data_streams[*].configurations[ImagingConfig].images[*]
    .planes[*]`` in the same order as
    :func:`aind_pophys_metadata.fields.get_fov_pairs_v2`. The v1 FOV fields map
    as: ``imaging_depth`` -> ``Plane.depth``, ``coupled_fov_index`` ->
    ``CoupledPlane.coupled_plane_index``, ``fov_width``/``fov_height`` ->
    ``PlanarImage.dimensions.scale``, ``fov_scale_factor`` -> the ``Scale``
    entry of ``PlanarImage.image_to_acquisition_transform``, and
    ``fov_coordinate_unit`` -> ``PlanarImage.dimensions_unit``.

    Parameters
    ----------
    acquisition : dict
        Raw ``acquisition.json`` dict.

    Returns
    -------
    list of PlaneMetadata
        Unsorted per-plane records.
    """
    records: List[PlaneMetadata] = []
    for config in _imaging_configs(acquisition):
        for image in config.get("images") or []:
            if image.get("object_type") not in _PLANAR_IMAGE_TYPES:
                continue
            dimensions = (image.get("dimensions") or {}).get("scale") or []
            scale_factor = _first_scale(image.get("image_to_acquisition_transform"))
            coordinate_unit = image.get("dimensions_unit") or DEFAULT_LENGTH_UNIT
            for plane in image.get("planes") or []:
                records.append(
                    PlaneMetadata(
                        plane_id="",
                        plane_index=_as_int(plane.get("plane_index")) or 0,
                        structure=fields.acronym_from_targeted_structure(
                            plane.get("targeted_structure")
                        ),
                        depth=_as_float(plane.get("depth")),
                        depth_unit=plane.get("depth_unit") or DEFAULT_LENGTH_UNIT,
                        coupled_plane_index=_as_int(plane.get("coupled_plane_index")),
                        scale_factor=scale_factor,
                        coordinate_unit=coordinate_unit,
                        width=_as_int(dimensions[0]) if len(dimensions) > 0 else None,
                        height=_as_int(dimensions[1]) if len(dimensions) > 1 else None,
                    )
                )
    return records


def attach_plane_ids(
    records: List[PlaneMetadata],
    plane_ids: Tuple[str, ...],
) -> Tuple[PlaneMetadata, ...]:
    """Pair rich plane records to the shared layer's canonical plane ids.

    Both sides are ordered by plane index with a stable sort, so pairing is
    positional -- the documented cross-capsule contract. A length mismatch
    means the two traversals disagree and is raised rather than silently
    truncated, because a wrong plane id names the wrong output folder.

    Parameters
    ----------
    records : list of PlaneMetadata
        Records from :func:`plane_records_v1` / :func:`plane_records_v2`.
    plane_ids : tuple of str
        Canonical ids from
        :func:`aind_pophys_metadata.fields.get_fov_ids`.

    Returns
    -------
    tuple of PlaneMetadata
        Records ordered by plane index with ``plane_id`` populated.

    Raises
    ------
    ValueError
        If the record count and the id count differ.
    """
    ordered = sorted(records, key=lambda record: record.plane_index)
    if len(ordered) != len(plane_ids):
        raise ValueError(
            f"Resolved {len(ordered)} imaging planes but "
            f"{len(plane_ids)} canonical plane ids "
            f"({', '.join(plane_ids) or 'none'}); the shared plane-id "
            f"traversal and the NWB plane traversal disagree."
        )
    return tuple(
        PlaneMetadata(
            plane_id=plane_id,
            plane_index=record.plane_index,
            structure=record.structure,
            depth=record.depth,
            depth_unit=record.depth_unit,
            coupled_plane_index=record.coupled_plane_index,
            scale_factor=record.scale_factor,
            coordinate_unit=record.coordinate_unit,
            width=record.width,
            height=record.height,
        )
        for record, plane_id in zip(ordered, plane_ids)
    )


def get_excitation_wavelength(version: str, core_raw: dict) -> Optional[float]:
    """Excitation wavelength in nm, for either schema version.

    v1 reads ``data_streams[*].light_sources[0].wavelength``; v2 reads the
    first ``LaserConfig`` under ``ImagingConfig.channels[*].light_sources``.

    Parameters
    ----------
    version : str
        Schema version string.
    core_raw : dict
        Raw core acquisition dict.

    Returns
    -------
    float or None
        The wavelength, or ``None`` when no light source declares one.
    """
    if version == io.SCHEMA_V1:
        for stream in core_raw.get("data_streams") or []:
            for source in stream.get("light_sources") or []:
                wavelength = _as_float(source.get("wavelength"))
                if wavelength is not None:
                    return wavelength
        return None
    for config in _imaging_configs(core_raw):
        for channel in config.get("channels") or []:
            for source in channel.get("light_sources") or []:
                if source.get("object_type") not in (None, _LASER_CONFIG_TYPE):
                    continue
                wavelength = _as_float(source.get("wavelength"))
                if wavelength is not None:
                    return wavelength
    return None


def get_emission_wavelength(version: str, core_raw: dict) -> Optional[float]:
    """Emission wavelength in nm from a v2 channel, else ``None``.

    v1 had no per-channel emission wavelength (the capsule hardcoded 514 nm as
    a GCaMP6 placeholder), so this returns ``None`` for v1 input and the
    caller keeps the placeholder.

    Parameters
    ----------
    version : str
        Schema version string.
    core_raw : dict
        Raw core acquisition dict.

    Returns
    -------
    float or None
        The emission wavelength, or ``None`` when unavailable.
    """
    if version == io.SCHEMA_V1:
        return None
    for config in _imaging_configs(core_raw):
        for channel in config.get("channels") or []:
            wavelength = _as_float(channel.get("emission_wavelength"))
            if wavelength is not None:
                return wavelength
    return None


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


def load_nwb_metadata(raw_dir: Path) -> NwbMetadata:
    """Read v1/v2 metadata for NWB packaging from the raw mount.

    Dispatches on which core file is present (``acquisition.json`` -> v2,
    ``session.json`` -> v1) and falls back to ``platform.json`` for the frame
    rate when the core file omits it.

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
        If neither ``acquisition.json`` nor ``session.json`` is found.
    ValueError
        If both core files are present (ambiguous input), if no imaging plane
        can be resolved, or if no frame rate can be resolved from any source.
    """
    raw_dir = Path(raw_dir)
    common = io.load_common(raw_dir)
    core_path = common.core_path
    version = common.version
    core_raw = common.core_raw
    procedures_path = find_one(raw_dir, PROCEDURES_FILE, required=False)
    procedures_raw = io.load_json(procedures_path) if procedures_path else None

    records = plane_records_v1(core_raw) if version == io.SCHEMA_V1 else plane_records_v2(core_raw)
    if not records:
        raise ValueError(
            f"No imaging planes found in {core_path.name} (schema {version}); "
            "NWB packaging has nothing to write."
        )
    planes = attach_plane_ids(records, fields.get_fov_ids(version, core_raw))

    # Validated here rather than ~150 lines later in job.py: every NWB imaging
    # plane needs an imaging_rate, so an asset without one cannot be packaged
    # at all and should fail at load, naming the file and schema version.
    frame_rate = fields.resolve_frame_rate(
        version,
        core_raw,
        common.platform_raw,
        core_file=core_path,
        input_dir=raw_dir,
        required=True,
    )

    subject_raw = common.subject_raw
    genotype = (subject_raw or {}).get("genotype")
    indicator = get_injection_material(procedures_raw) or genotype
    return NwbMetadata(
        schema_version=version,
        core_file=core_path,
        planes=planes,
        frame_rate=frame_rate,
        instrument_id=fields.get_instrument_id(version, core_raw),
        excitation_wavelength=get_excitation_wavelength(version, core_raw),
        emission_wavelength=get_emission_wavelength(version, core_raw),
        indicator=indicator,
        genotype=genotype,
        subject_id=fields.get_subject_id(core_raw, subject_raw),
        dataset_name=fields.get_dataset_name(common.data_description_raw),
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
        parameters=parameters,
    )


def attach_input_data(code: Code, input_data: List[str]) -> None:
    """Record resolved input artifacts on an already-built ``Code``.

    The same ephemeral-path rejection :func:`build_code` applies is repeated
    here, since this bypasses it.

    Parameters
    ----------
    code : Code
        The code block to populate.
    input_data : list of str
        Input artifact path(s)/asset name(s), relative to their mount root.

    Raises
    ------
    ValueError
        If any entry is a Nextflow task scratch path.
    """
    shared_processing.reject_ephemeral_paths(list(input_data), "input_data")
    code.input_data = [DataAsset(name=name) for name in input_data]
