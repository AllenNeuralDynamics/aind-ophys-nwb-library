"""Top-level NWB packaging orchestration.

``run()`` is the library entry point called by the thin capsule wrapper: read
the acquisition metadata (v1 or v2) -> configure logging -> resolve every
upstream capsule's artifacts -> build one NWB file covering every plane ->
write it as a Zarr store -> emit a full ``processing.json`` and
``quality_control.json``.

Everything is written into ``<output_dir>/nwb/`` rather than at the results
root. ``main.nf`` publishes with ``saveAs { new File(filename).getName() }``,
which flattens every published path to its basename, so a root-level
``processing.json`` would collide with the aggregator tier's. One subdirectory
also matches the ``<subdir>/<step>`` shape every other v2 capsule adopted.

There is exactly **one** ``processing.json`` and **one**
``quality_control.json`` per run, not one per plane: ``main.nf`` calls this
capsule once with every upstream channel ``.collect()``ed, so a run produces a
single NWB file describing all planes. ``processing.json`` is written by the
shared ``stage_guard``, which times the stage and emits a document whether or
not the body raised, with the error in ``notes``.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import relative_to_root
from aind_pophys_metadata.processing import collect_upstream_process_names
from aind_pophys_metadata.runner import StageContext, stage_guard
from log_schema.main import setup_logging

from . import file_handling, nwb_builders, qc
from .settings import NwbSettings
from .utils import metadata_utils as mu
from .utils.metadata_utils import SessionType

logger = logging.getLogger("Ophys NWB packaging")

PROCESS_NAME = mu.PROCESS_NAME

#: Subdirectory of the results root that every output is written into.
NWB_SUBDIR = "nwb"

VSYNC_LINES = ("2p_vsync", "vsync_2p")


def _verify_outputs(output_dir: Path) -> None:
    """Round-trip the emitted metadata through the core v2 schema objects."""
    from aind_data_schema.core.processing import Processing
    from aind_data_schema.core.quality_control import QualityControl

    Processing.model_validate_json((Path(output_dir) / "processing.json").read_text())
    QualityControl.model_validate_json((Path(output_dir) / "quality_control.json").read_text())


def _resolve_mount(input_dir: Path, subdir: str) -> Path:
    """Return ``input_dir/subdir``, falling back to ``input_dir`` itself.

    In the pipeline the raw and processed assets are copied into
    ``/data/raw`` and ``/data/processed``; in a standalone capsule run they can
    be mounted flat. Falling back keeps both usable instead of failing with a
    misleading "no acquisition file" error.

    Parameters
    ----------
    input_dir : Path
        The mount root.
    subdir : str
        Expected subdirectory name.

    Returns
    -------
    Path
        The resolved directory.
    """
    candidate = Path(input_dir) / subdir
    if candidate.is_dir():
        return candidate
    logger.warning(
        "No '%s' subdirectory under %s; searching the mount root instead.",
        subdir,
        input_dir,
    )
    return Path(input_dir)


def _sync_rising_edges(sync_file: Path) -> np.ndarray:
    """Return the 2p vsync rising edges from a raw sync HDF5 file.

    Parameters
    ----------
    sync_file : Path
        Path to the sync ``.h5``.

    Returns
    -------
    numpy.ndarray
        Rising-edge times in seconds.

    Raises
    ------
    ValueError
        If neither known vsync line name is present, naming both.
    """
    from aind_behavior_utils.sync.sync_dataset import SyncDataset

    dataset = SyncDataset(sync_file)
    for line in VSYNC_LINES:
        try:
            return dataset.get_rising_edges(line=line, units="seconds")
        except ValueError:
            logger.warning("Sync line '%s' not found in %s", line, sync_file)
    raise ValueError(
        f"None of the sync lines {VSYNC_LINES} are present in {sync_file}; "
        f"multiplane traces cannot be timestamped."
    )


def _plane_timestamps(
    metadata: mu.NwbMetadata,
    raw_dir: Path,
) -> Dict[str, np.ndarray]:
    """Return per-plane sync timestamps for a multiplane session.

    Parameters
    ----------
    metadata : mu.NwbMetadata
        The acquisition metadata.
    raw_dir : Path
        The raw mount, searched for the sync file.

    Returns
    -------
    dict of str to numpy.ndarray
        Per-plane timestamps, keyed by plane id.

    Raises
    ------
    FileNotFoundError
        If no sync file can be located.
    """
    sync_file = file_handling.find_sync_file(raw_dir)
    if sync_file is None:
        raise FileNotFoundError(
            f"No sync file found under {raw_dir}; a multiplane session needs "
            f"one to timestamp each plane's traces."
        )
    logger.info("Using sync file %s", sync_file)
    return nwb_builders.sync_timestamps_per_plane(metadata.planes, _sync_rising_edges(sync_file))


def _package_session(
    settings: NwbSettings,
    metadata: mu.NwbMetadata,
    raw_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    code: Code,
    ctx: StageContext,
) -> None:
    """Build and write the NWB file plus its quality_control.json.

    ``code`` and ``ctx`` are populated as the run progresses so the
    ``processing.json`` the stage guard writes reports partial provenance even
    when this raises.
    """
    import aind_nwb_utils.utils as nwb_utils

    results_root = settings.output_dir.resolve()

    epoch_json = file_handling.find_epoch_locations(processed_dir)
    session = mu.session_type(len(metadata.planes), epoch_json is not None)
    logger.info("Detected session type %s", session.value)

    artifacts = file_handling.collect_session_artifacts(processed_dir, metadata.plane_ids)
    mu.attach_input_data(
        code,
        [relative_to_root(processed_dir, plane.extraction_h5) for plane in artifacts],
    )

    # create_base_nwb_file reads the core files from one directory (not
    # recursively), and dispatches session.json vs acquisition.json itself.
    nwbfile = nwb_utils.create_base_nwb_file(metadata.core_file.parent)
    device, optical_channel = nwb_builders.create_microscope(nwbfile, metadata)

    timestamps: Dict[str, np.ndarray] = {}
    if session is SessionType.MULTIPLANE:
        timestamps = _plane_timestamps(metadata, raw_dir)
    elif session is SessionType.SINGLE_PLANE_WITH_EPOCHS:
        nwb_builders.add_epoch_intervals(nwbfile, epoch_json, metadata.frame_rate)

    metrics = []
    total_rois = 0
    for plane, plane_artifacts in zip(metadata.planes, artifacts):
        contents, num_rois = nwb_builders.add_plane(
            nwbfile,
            metadata,
            plane,
            plane_artifacts,
            device,
            optical_channel,
            timestamps=timestamps.get(plane[mu.PLANE_ID_KEY]),
        )
        total_rois += num_rois
        plane_id = plane[mu.PLANE_ID_KEY]
        metrics.append(qc.build_plane_contents_metric(plane_id, contents, num_rois))

    if session is SessionType.MULTIPLANE and not settings.skip_lab_metadata:
        nwb_builders.add_plane_lab_metadata(nwbfile, metadata.planes, settings.extension_namespace)

    nwb_path = nwb_builders.write_nwb_zarr(nwbfile, output_dir / settings.nwb_name)
    nwb_reference = relative_to_root(results_root, nwb_path)
    logger.info("Wrote %s", nwb_path)

    readable = _reopened(nwb_path, metadata.plane_ids)
    metrics.append(qc.build_nwb_written_metric(metadata.plane_ids, nwb_reference, readable))
    metrics.append(qc.build_nwb_review_metric(nwb_reference))
    qc.write_quality_control_json(qc.build_quality_control(metrics), output_dir)

    # Scalar summary statistics describing THIS step's outputs, so a reviewer
    # can triage the run without opening the NWB store. Input parameters
    # belong on the code block, never here.
    ctx.output_parameters.update(
        {
            "session_type": session.value,
            "num_planes": len(metadata.planes),
            "num_rois": total_rois,
            "nwb_path": nwb_reference,
            "nwb_reopened": readable,
        }
    )


def _reopened(nwb_path: Path, plane_ids: Sequence[str]) -> bool:
    """Re-open the written store and report whether every plane is present.

    Parameters
    ----------
    nwb_path : Path
        The written ``*.nwb.zarr`` directory.
    plane_ids : sequence of str
        Plane ids expected as processing modules.

    Returns
    -------
    bool
        ``True`` when the store re-opened and carries every plane.
    """
    try:
        found = nwb_builders.read_nwb_zarr(nwb_path)
    except Exception as exc:
        logger.error("Could not re-open %s: %s", nwb_path, exc)
        return False
    missing = [plane_id for plane_id in plane_ids if plane_id not in found]
    if missing:
        logger.error("Re-opened %s is missing planes %s", nwb_path, missing)
        return False
    return True


def run(settings: Optional[NwbSettings] = None) -> None:
    """Package the ophys pipeline's outputs into one NWB Zarr store.

    Parameters
    ----------
    settings : NwbSettings, optional
        Parsed settings; constructed from CLI/environment when omitted. When
        ``settings.verify`` is True, the emitted metadata is round-tripped
        through the core v2 schema objects after writing.
    """
    settings = settings or NwbSettings()

    raw_dir = _resolve_mount(settings.input_dir, settings.raw_subdir)
    processed_dir = _resolve_mount(settings.input_dir, settings.processed_subdir)

    metadata = mu.load_nwb_metadata(raw_dir)
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )
    logger.info("Packaging %s plane(s)", len(metadata.planes))

    results_root = settings.output_dir.resolve()
    output_dir = results_root / NWB_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    code = mu.build_nwb_code(settings.recorded_parameters())

    with stage_guard(
        output_dir,
        ProcessName.FILE_FORMAT_CONVERSION,
        code,
        logger=logger,
        name=mu.DATA_PROCESS_NAME,
        output_path=relative_to_root(results_root, output_dir),
    ) as ctx:
        # Every upstream processing.json staged into this task is a flat
        # sibling dependency: no ordering, no parent/grandparent distinction.
        ctx.upstream_names = collect_upstream_process_names(
            processed_dir, exclude=mu.DATA_PROCESS_NAME
        )
        _package_session(settings, metadata, raw_dir, processed_dir, output_dir, code, ctx)

    if settings.verify:
        _verify_outputs(output_dir)
