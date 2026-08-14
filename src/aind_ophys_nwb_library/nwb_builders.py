"""pynwb object construction from upstream ophys pipeline artifacts.

Multiplane and single-plane sessions share one :func:`add_plane`, parameterised
by the timing source and the container-name suffix, so a fix cannot land on one
path and miss the other. The microscope device is described from the acquisition
metadata's ``instrument_id``, which the shared read layer supplies for both
schema versions.

An optional artifact that cannot be read is logged and its NWB container
omitted; the extraction traces, without which there is nothing to package,
raise instead.

Verified 2026-07-30 against pynwb 4.1.0 / hdmf 6.1.0 / hdmf-zarr 0.13.0: every
symbol below exists and the resulting object graph round-trips through
``NWBZarrIO``.

Coverage-omitted (see pyproject): it needs real HDF5 payloads and a writable
Zarr store, i.e. a Code Ocean data run.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynwb
import sparse
from hdmf.common import VectorData
from hdmf_zarr import NWBZarrIO
from pynwb import NWBFile
from pynwb.epoch import TimeIntervals
from pynwb.image import GrayscaleImage, Images
from pynwb.base import ProcessingModule
from pynwb.ophys import (
    DfOverF,
    Fluorescence,
    ImageSegmentation,
    ImagingPlane,
    OpticalChannel,
    PlaneSegmentation,
    RoiResponseSeries,
)

from .file_handling import PlaneArtifacts
from .utils.metadata_utils import NwbMetadata, plane_location

logger = logging.getLogger(__name__)

# Per-ROI suite2p morphology stats surfaced from the extraction file's rois/
# group into the NWB roi_table.
SUITE2P_ROI_STATS: Tuple[str, ...] = (
    "aspect_ratio",
    "compact",
    "solidity",
    "radius",
    "footprint",
)

EXTENSION_NAMESPACE_NAME = "ndx-aibs-behavior-ophys"
EXTENSION_TYPE = "OphysMetadataSchema"

OPTICAL_CHANNEL_NAME = "channel_0"
# GCaMP6 emission placeholder, used only when the acquisition metadata carries
# no per-channel emission wavelength (always the case for v1 input).
DEFAULT_EMISSION_LAMBDA = 514.0
MESOSCOPE_DESCRIPTION = "AIND Multiplane Mesoscope 2P Rig"


class SegmentationApproach(Enum):
    """How suite2p produced the ROI masks in an extraction file."""

    SUITE2P_ANATOMICAL = {
        "method": "suite2p-cellpose",
        "description": ("Suite2p's anatomical mode using either the max or mean projection"),
    }
    SUITE2P_ACTIVITY = {
        "method": "suite2p-functional",
        "description": "Suite2p's activity-based ROI detection using 'sparse mode'",
    }


def load_pynwb_extension(path: Path) -> type:
    """Load the ndx-aibs-behavior-ophys namespace and return its metadata type.

    Parameters
    ----------
    path : Path
        Path to ``ndx-aibs-behavior-ophys.namespace.yaml``. It must live under
        the capsule's ``code/`` directory: ``main.nf`` stages only ``code/``,
        so a file anywhere else in the repo is absent in every pipeline run.

    Returns
    -------
    type
        The generated ``OphysMetadataSchema`` class.

    Raises
    ------
    FileNotFoundError
        If the namespace YAML is missing, named so the failure is one line to
        diagnose rather than a pynwb parse error.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"pynwb extension namespace not found at {path}. It must be "
            f"committed under the capsule's code/ directory, which is the "
            f"only part of the repo main.nf stages."
        )
    pynwb.load_namespaces(str(path))
    return pynwb.get_class(EXTENSION_TYPE, EXTENSION_NAMESPACE_NAME)


def load_dataset(h5_file: Path, key: str, group: Optional[str] = None) -> np.ndarray:
    """Read one dataset out of an upstream HDF5 file.

    Parameters
    ----------
    h5_file : Path
        Path to the HDF5 file.
    key : str
        Dataset name.
    group : str, optional
        Group containing ``key``; read from the root when omitted.

    Returns
    -------
    numpy.ndarray
        The dataset contents.
    """
    with h5py.File(h5_file, "r") as f:
        node = f[group][key] if group else f[key]
        return node[:]


def try_load_dataset(
    h5_file: Optional[Path],
    key: str,
    group: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Read an optional dataset, logging rather than swallowing a failure.

    Parameters
    ----------
    h5_file : Path or None
        Path to the HDF5 file, or ``None`` when the artifact is absent.
    key : str
        Dataset name.
    group : str, optional
        Group containing ``key``.

    Returns
    -------
    numpy.ndarray or None
        The dataset, or ``None`` when the file or the key is missing.
    """
    if h5_file is None:
        return None
    try:
        return load_dataset(h5_file, key, group)
    except (KeyError, OSError) as exc:
        logger.warning(
            "Could not read %s%s from %s: %s",
            f"{group}/" if group else "",
            key,
            h5_file,
            exc,
        )
        return None


def load_roi_masks(extraction_h5: Path) -> np.ndarray:
    """Expand the sparse ROI footprints in an extraction file to dense masks.

    Parameters
    ----------
    extraction_h5 : Path
        Path to ``<plane>_extraction.h5``.

    Returns
    -------
    numpy.ndarray
        Dense boolean-ish array of shape ``(n_rois, height, width)``.
    """
    with h5py.File(extraction_h5, "r") as f:
        data = f["rois"]["data"][:]
        coords = f["rois"]["coords"][:]
        shape = f["rois"]["shape"][:]
    return sparse.COO(coords, data, shape).todense()


def load_neuropil_masks_and_r_values(
    extraction_h5: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load neuropil pixel masks and per-ROI neuropil r-values.

    Parameters
    ----------
    extraction_h5 : Path
        Path to ``<plane>_extraction.h5``.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(n_rois, height, width)``.
    numpy.ndarray
        Per-ROI neuropil correction coefficients.
    """
    with h5py.File(extraction_h5, "r") as f:
        indices = f["rois"]["neuropil_coords"][:].T
        shape = f["rois"]["shape"][:]
        r_values = f["traces"]["neuropil_rcoef"][:]
    neuropil_mask = np.zeros(shape, dtype=bool)
    neuropil_mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return neuropil_mask, r_values


def collapse_masks_to_label_image(masks: np.ndarray) -> np.ndarray:
    """Collapse 3-D per-ROI masks into a 2-D 1-based label image.

    Parameters
    ----------
    masks : numpy.ndarray
        Array of shape ``(n_rois, height, width)``.

    Returns
    -------
    numpy.ndarray
        ``(height, width)`` array whose pixels hold the 1-based index of the
        mask covering them (0 = no mask). All zeros when there are no ROIs.
    """
    label_image = np.zeros(masks.shape[1:], dtype="i2")
    # A plane can legitimately have zero ROIs: extraction's CaImAn branch
    # emits an empty mask array, and np.argmax raises on one.
    if masks.shape[0] == 0:
        return label_image
    covered = masks.any(0) if masks.dtype == bool else masks.sum(0) > 0
    label_image[covered] = np.argmax(masks[:, covered], 0) + 1
    return label_image


def segmentation_approach(extraction_h5: Path) -> SegmentationApproach:
    """Classify how the extraction file's ROIs were segmented.

    Parameters
    ----------
    extraction_h5 : Path
        Path to ``<plane>_extraction.h5``.

    Returns
    -------
    SegmentationApproach
        ``SUITE2P_ANATOMICAL`` when a ``cellpose`` group is present, else
        ``SUITE2P_ACTIVITY``.
    """
    with h5py.File(extraction_h5, "r") as f:
        if f.get("cellpose") is not None:
            return SegmentationApproach.SUITE2P_ANATOMICAL
    return SegmentationApproach.SUITE2P_ACTIVITY


def instrument_description(instrument_id: Optional[str]) -> str:
    """Describe the microscope from its instrument id.

    Replaces the v1 read of ``rig.json`` (a v1-only file; v2 renamed it to
    ``instrument.json`` and ``rig_id`` to ``instrument_id``).

    Parameters
    ----------
    instrument_id : str or None
        v1 ``rig_id`` / v2 ``instrument_id``.

    Returns
    -------
    str
        A human-readable description.
    """
    if instrument_id and instrument_id.upper().startswith("MESO"):
        return MESOSCOPE_DESCRIPTION
    return f"AIND 2P rig {instrument_id}" if instrument_id else "Unknown 2P rig"


def create_microscope(
    nwbfile: NWBFile,
    metadata: NwbMetadata,
) -> Tuple[pynwb.device.Device, OpticalChannel]:
    """Create the NWB device and optical channel for the acquisition.

    Parameters
    ----------
    nwbfile : NWBFile
        The file being built.
    metadata : NwbMetadata
        The acquisition metadata.

    Returns
    -------
    pynwb.device.Device
        The microscope device.
    OpticalChannel
        The optical channel, using the acquisition's emission wavelength when
        it carries one.
    """
    name = metadata.instrument_id or "unknown-instrument"
    device = nwbfile.create_device(
        name=name,
        description=instrument_description(metadata.instrument_id),
        manufacturer="Thorlabs",
    )
    emission = metadata.emission_wavelength
    channel = OpticalChannel(
        name=OPTICAL_CHANNEL_NAME,
        description=(
            "Emission wavelength read from the acquisition metadata"
            if emission is not None
            else (
                "Emission wavelength not present in the acquisition "
                f"metadata; using the {DEFAULT_EMISSION_LAMBDA} nm GCaMP6 "
                "placeholder"
            )
        ),
        emission_lambda=float(emission if emission is not None else DEFAULT_EMISSION_LAMBDA),
    )
    return device, channel


#: roi_table columns sourced from the CLASSIFIER's output. Present only when
#: the classification file is usable.
CLASSIFICATION_COLUMN_SPECS = (
    ("is_soma", "Soma predictions"),
    ("soma_probability", "Soma probabilities"),
    ("is_dendrite", "Dendrite predictions"),
    ("dendrite_probability", "Dendrite probabilities"),
)

#: roi_table columns sourced from EXTRACTION's own output. Kept separate from
#: the classification specs so a classifier ROI-count mismatch cannot drop
#: columns that extraction.h5 supplied perfectly well.
EXTRACTION_COLUMN_SPECS = (
    (
        "cellpose_soma_probability",
        (
            "Mean cellpose cell-probability under each ROI footprint "
            "(probability in [0, 1]); NaN when cellpose segmentation was "
            "not run"
        ),
    ),
) + tuple((stat, f"suite2p ROI stat: {stat}") for stat in SUITE2P_ROI_STATS)


def _roi_table_columns(include_classification: bool) -> Tuple[list, list]:
    """Return the roi_table VectorData columns and their names.

    Parameters
    ----------
    include_classification : bool
        Whether the classifier's four columns are available. The
        extraction-derived columns are always included.

    Returns
    -------
    list of hdmf.common.VectorData
        Column definitions.
    list of str
        Column names, parallel to the definitions.
    """
    specs = list(EXTRACTION_COLUMN_SPECS)
    if include_classification:
        specs = list(CLASSIFICATION_COLUMN_SPECS) + specs
    columns = [VectorData(name=name, description=desc) for name, desc in specs]
    return columns, [name for name, _ in specs]


def _classification_columns(
    artifacts: PlaneArtifacts,
    num_rois: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Load the per-ROI classification columns, or ``None`` when unusable.

    Pairing with the ROI masks is **positional** (suite2p order) -- the
    cross-capsule contract with ``aind-ophys-classifier`` -- so a length
    mismatch is rejected rather than silently truncated.

    Parameters
    ----------
    artifacts : PlaneArtifacts
        Resolved upstream artifacts for this plane.
    num_rois : int
        Number of ROI masks in the extraction file.

    Returns
    -------
    dict of str to numpy.ndarray, or None
        The four classifier columns, or ``None`` when the classification file
        is absent or disagrees on the ROI count. ``soma/probabilities`` and
        ``dendrites/probabilities`` are ``predict_proba`` output and stay 2-D
        ``(n_rois, n_classes)`` here; :func:`_add_rois` takes the positive
        class off the last column. The extraction-derived columns are built
        separately by :func:`_extraction_columns`, so they survive a
        classifier problem.
    """
    classification = artifacts.classification_h5
    if classification is None:
        logger.warning(
            "No classification file for %s; roi_table will omit the " "soma/dendrite columns.",
            artifacts.plane_id,
        )
        return None
    columns: Dict[str, np.ndarray] = {}
    for column, group, key in (
        ("is_soma", "soma", "predictions"),
        ("soma_probability", "soma", "probabilities"),
        ("is_dendrite", "dendrites", "predictions"),
        ("dendrite_probability", "dendrites", "probabilities"),
    ):
        values = try_load_dataset(classification, key, group)
        if values is None:
            return None
        if len(values) != num_rois:
            logger.error(
                "%s/%s in %s has %s entries but the extraction file has %s "
                "ROIs; classification columns are paired by positional ROI "
                "index, so they are omitted rather than misaligned.",
                group,
                key,
                classification,
                len(values),
                num_rois,
            )
            return None
        columns[column] = values
    return columns


def _extraction_columns(
    artifacts: PlaneArtifacts,
    num_rois: int,
) -> Dict[str, np.ndarray]:
    """Load the per-ROI columns that come from ``extraction.h5`` itself.

    Always returns a full set: any dataset the extraction run did not produce
    (notably on its CaImAn branch, where ``stat = {}``) becomes an all-NaN
    column rather than removing the column entirely.

    Parameters
    ----------
    artifacts : PlaneArtifacts
        Resolved upstream artifacts for this plane.
    num_rois : int
        Number of ROI masks in the extraction file.

    Returns
    -------
    dict of str to numpy.ndarray
        One entry per name in :data:`EXTRACTION_COLUMN_SPECS`.
    """
    columns: Dict[str, np.ndarray] = {}
    cellpose = try_load_dataset(artifacts.extraction_h5, "cellpose_soma_probability", "rois")
    columns["cellpose_soma_probability"] = (
        cellpose if cellpose is not None else np.full(num_rois, np.nan)
    )
    for stat in SUITE2P_ROI_STATS:
        values = try_load_dataset(artifacts.extraction_h5, stat, "rois")
        columns[stat] = values if values is not None else np.full(num_rois, np.nan)
    return columns


def _add_rois(
    plane_segmentation: PlaneSegmentation,
    masks: np.ndarray,
    columns: Optional[Dict[str, np.ndarray]],
    extraction_columns: Dict[str, np.ndarray],
) -> None:
    """Add every ROI mask (and its optional columns) to a plane segmentation.

    Parameters
    ----------
    plane_segmentation : pynwb.ophys.PlaneSegmentation
        The segmentation table being filled.
    masks : numpy.ndarray
        Dense per-ROI masks of shape ``(n_rois, height, width)``.
    columns : dict of str to numpy.ndarray, or None
        The classifier's four columns, or ``None`` when unavailable. The two
        probability columns must be 2-D ``(n_rois, n_classes)``
        ``predict_proba`` output: the positive class is read as ``[idx][-1]``,
        so a 1-D array passes the ROI-count check and then indexes the wrong
        axis.
    extraction_columns : dict of str to numpy.ndarray
        The extraction-derived columns, always present.
    """
    for idx, mask in enumerate(masks):
        row = {name: extraction_columns[name][idx] for name, _ in EXTRACTION_COLUMN_SPECS}
        if columns is not None:
            row.update(
                {
                    "is_soma": columns["is_soma"][idx],
                    "soma_probability": columns["soma_probability"][idx][-1],
                    "is_dendrite": columns["is_dendrite"][idx],
                    "dendrite_probability": columns["dendrite_probability"][idx][-1],
                }
            )
        plane_segmentation.add_roi(image_mask=mask, **row)


def _add_neuropil_segmentation(
    extraction_h5: Path,
    img_seg: ImageSegmentation,
    imaging_plane: ImagingPlane,
    resolution: float,
) -> Optional[GrayscaleImage]:
    """Add the neuropil segmentation table and return its label image.

    Parameters
    ----------
    extraction_h5 : Path
        Path to ``<plane>_extraction.h5``.
    img_seg : ImageSegmentation
        The segmentation container to add the table to.
    imaging_plane : ImagingPlane
        The plane the table belongs to.
    resolution : float
        Pixel resolution recorded on the returned image.

    Returns
    -------
    GrayscaleImage or None
        The collapsed neuropil label image, or ``None`` when the extraction
        file carries no neuropil masks.
    """
    try:
        masks, r_values = load_neuropil_masks_and_r_values(extraction_h5)
    except (KeyError, OSError) as exc:
        logger.warning(
            "No neuropil masks in %s (%s); the neuropil_table is omitted.",
            extraction_h5,
            exc,
        )
        return None
    table = img_seg.create_plane_segmentation(
        name="neuropil_table",
        description="Neuropil segmentation masks from suite2p",
        imaging_plane=imaging_plane,
    )
    for mask in masks:
        table.add_roi(image_mask=mask)
    table.add_column(
        name="neuropil_r_value",
        description="Coefficients used for neuropil correction",
        data=r_values,
    )
    return GrayscaleImage(
        name="neuropil_mask_image",
        data=collapse_masks_to_label_image(masks),
        resolution=resolution,
        description="Neuropil segmentation mask image",
    )


def _as_grayscale(image: np.ndarray, path: Path) -> np.ndarray:
    """Reduce a decoded PNG to the 2-D array ``GrayscaleImage`` requires.

    Motion correction writes 2-D uint8 projections via ``PIL.Image.fromarray``,
    so this is normally a no-op. A colour-mapped PNG decodes to ``(H, W, 3|4)``
    and would abort the whole terminal stage inside ``GrayscaleImage``, so it is
    collapsed to its first channel with a warning instead.

    Parameters
    ----------
    image : numpy.ndarray
        The decoded image.
    path : Path
        Source path, for the warning message.

    Returns
    -------
    numpy.ndarray
        A 2-D array.
    """
    if image.ndim == 2:
        return image
    logger.warning(
        "%s decoded to shape %s, expected 2-D; using the first channel.",
        path,
        image.shape,
    )
    return image[..., 0]


def _summary_images(
    artifacts: PlaneArtifacts,
    approach: SegmentationApproach,
    resolution: float,
    neuropil_image: Optional[GrayscaleImage],
) -> list:
    """Build the summary GrayscaleImages for one plane.

    Parameters
    ----------
    artifacts : PlaneArtifacts
        Resolved upstream artifacts for this plane.
    approach : SegmentationApproach
        How the ROIs were segmented; selects the segmentation-mask source.
    resolution : float
        Pixel resolution recorded on every image.
    neuropil_image : GrayscaleImage or None
        The neuropil label image, appended when present.

    Returns
    -------
    list of GrayscaleImage
        The images to place in the plane's ``images`` container.
    """
    images = []
    for name, path, description in (
        (
            "average_projection",
            artifacts.average_projection_png,
            "Average intensity projection of entire session",
        ),
        (
            "max_projection",
            artifacts.maximum_projection_png,
            "Max intensity projection of entire session",
        ),
    ):
        if path is None:
            logger.warning("No %s for %s; omitted.", name, artifacts.plane_id)
            continue
        images.append(
            GrayscaleImage(
                name=name,
                data=_as_grayscale(plt.imread(path), path),
                resolution=resolution,
                description=description,
            )
        )

    if approach is SegmentationApproach.SUITE2P_ANATOMICAL:
        mask = try_load_dataset(artifacts.extraction_h5, "masks", "cellpose")
    else:
        mask = collapse_masks_to_label_image(load_roi_masks(artifacts.extraction_h5))
    if mask is not None:
        images.append(
            GrayscaleImage(
                name="segmentation_mask_image",
                data=mask,
                resolution=resolution,
                description="Segmentation projection of entire session",
            )
        )
    if neuropil_image is not None:
        images.append(neuropil_image)
    return images


def _series_kwargs(
    timestamps: Optional[np.ndarray],
    frame_rate: float,
) -> Dict[str, Any]:
    """Return the timing kwargs for a RoiResponseSeries.

    Multiplane sessions carry per-plane sync timestamps; single-plane sessions
    have no sync file and fall back to the acquisition frame rate.

    Parameters
    ----------
    timestamps : numpy.ndarray or None
        Per-frame timestamps for this plane, or ``None``.
    frame_rate : float
        Acquisition frame rate in Hz.

    Returns
    -------
    dict
        Either ``{"timestamps": ...}`` or ``{"rate": ..., "starting_time": 0}``.
    """
    if timestamps is not None:
        return {"timestamps": timestamps}
    return {"rate": float(frame_rate), "starting_time": 0.0}


def _add_traces(
    module: ProcessingModule,
    artifacts: PlaneArtifacts,
    plane_segmentation: PlaneSegmentation,
    num_rois: int,
    suffix: str,
    timing: Dict[str, Any],
) -> Dict[str, bool]:
    """Add every available trace container for one plane.

    ``traces/roi`` and ``traces/neuropil`` come from the extraction file and
    are required -- without them the plane has no timeseries at all. The
    corrected, dF/F and event traces are optional.

    Parameters
    ----------
    module : pynwb.base.ProcessingModule
        The plane's processing module.
    artifacts : PlaneArtifacts
        Resolved upstream artifacts.
    plane_segmentation : pynwb.ophys.PlaneSegmentation
        Segmentation table the series reference.
    num_rois : int
        Expected ROI count; every trace array must agree.
    suffix : str
        Container-name suffix (``""`` single-plane, ``"_timeseries"`` multi),
        preserving the v1 naming of both paths.
    timing : dict
        Timing kwargs from :func:`_series_kwargs`.

    Returns
    -------
    dict of str to bool
        Which containers were added, for the QC metric.

    Raises
    ------
    KeyError
        If the extraction file has no ``traces/roi`` dataset.
    ValueError
        If a trace array's ROI count disagrees with the mask count.
    """
    added = {name: False for name in ("roi_fluorescence", "neuropil_fluorescence", "dff", "events")}

    def series(name: str, data: np.ndarray, unit: str) -> RoiResponseSeries:
        """Build one RoiResponseSeries over the plane's ROI table region."""
        if data.shape[0] != num_rois:
            raise ValueError(
                f"{name} for {artifacts.plane_id} has {data.shape[0]} traces "
                f"but the extraction file has {num_rois} ROI masks; traces "
                f"and masks are paired by positional ROI index."
            )
        region = plane_segmentation.create_roi_table_region(
            region=list(range(num_rois)), description="List of measured ROIs"
        )
        local_timing = dict(timing)
        ts = local_timing.get("timestamps")
        # A debug run caps the movie to its first N frames, so the traces are
        # short while the sync file still describes the whole acquisition --
        # that is the expected cause of a mismatch here. Every upstream cap is
        # a head slice, so the head of the timestamp vector is the matching
        # window. Any other cause means the movie and the sync record genuinely
        # disagree, hence the warning rather than a silent trim.
        if ts is not None and len(ts) != data.shape[1]:
            logger.warning(
                "%s for %s has %s samples but %s sync timestamps; trimming "
                "timestamps to the trace length. Expected only for a "
                "debug-truncated movie -- otherwise the movie and the sync "
                "record disagree.",
                name,
                artifacts.plane_id,
                data.shape[1],
                len(ts),
            )
            local_timing["timestamps"] = ts[: data.shape[1]]
        return RoiResponseSeries(name=name, data=data.T, rois=region, unit=unit, **local_timing)

    roi = load_dataset(artifacts.extraction_h5, "roi", "traces")
    module.add(
        Fluorescence(
            roi_response_series=series(f"roi_fluorescence{suffix}", roi, "a.u."),
            name=f"raw{suffix}",
        )
    )
    added["roi_fluorescence"] = True

    neuropil = load_dataset(artifacts.extraction_h5, "neuropil", "traces")
    module.add(series(f"neuropil_fluorescence{suffix}", neuropil, "a.u."))
    added["neuropil_fluorescence"] = True

    corrected = try_load_dataset(artifacts.extraction_h5, "corrected", "traces")
    if corrected is not None:
        module.add(series(f"neuropil_corrected{suffix}", corrected, "a.u."))

    dff = try_load_dataset(artifacts.dff_h5, "data")
    if dff is not None:
        module.add(
            DfOverF(
                roi_response_series=series(f"dff{suffix}", dff, "%"),
                name=f"dff{suffix}",
            )
        )
        added["dff"] = True

    events = try_load_dataset(artifacts.events_oasis_h5, "events")
    if events is not None:
        module.add(series(f"event{suffix}", events, "a.u."))
        added["events"] = True
    return added


def add_plane(
    nwbfile: NWBFile,
    metadata: NwbMetadata,
    plane: Dict[str, Any],
    artifacts: PlaneArtifacts,
    device: pynwb.device.Device,
    optical_channel: OpticalChannel,
    timestamps: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, bool], int]:
    """Add one imaging plane's segmentation, images and traces to the NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The file being built.
    metadata : NwbMetadata
        The acquisition metadata.
    plane : dict
        This plane's imaging metadata record.
    artifacts : PlaneArtifacts
        Resolved upstream artifacts for this plane.
    device : pynwb.device.Device
        The microscope device.
    optical_channel : OpticalChannel
        The optical channel.
    timestamps : numpy.ndarray, optional
        Per-frame sync timestamps; when omitted the acquisition frame rate is
        used instead (single-plane sessions have no sync file).

    Returns
    -------
    dict of str to bool
        Which NWB containers this plane contributed, for QC.
    int
        The number of ROIs written.
    """
    suffix = "_timeseries" if timestamps is not None else ""
    resolution = float(plane["scale_factor"]) if plane["scale_factor"] else 1.0
    logger.info("Packaging plane %s", plane["plane_id"])

    imaging_plane = nwbfile.create_imaging_plane(
        name=plane["plane_id"],
        optical_channel=optical_channel,
        imaging_rate=float(metadata.frame_rate),
        description="Two-photon imaging plane",
        device=device,
        excitation_lambda=float(
            metadata.excitation_wavelength
            if metadata.excitation_wavelength is not None
            else float("nan")
        ),
        indicator=metadata.indicator or "unknown",
        location=plane_location(plane),
        grid_spacing=[resolution, resolution],
        grid_spacing_unit=plane["scale_factor_unit"],
        origin_coords=[0.0, 0.0, float(plane["depth"] or 0.0)],
        origin_coords_unit=plane["depth_unit"],
    )
    module = nwbfile.create_processing_module(
        name=plane["plane_id"],
        description=f"Processed ophys data for {plane['plane_id']}",
    )

    approach = segmentation_approach(artifacts.extraction_h5)
    img_seg = ImageSegmentation(name="image_segmentation")
    masks = load_roi_masks(artifacts.extraction_h5)
    num_rois = int(masks.shape[0])
    columns = _classification_columns(artifacts, num_rois)
    # Extraction's own columns are unconditional: a classifier problem must not
    # remove data that came from extraction.h5.
    extraction_columns = _extraction_columns(artifacts, num_rois)
    column_defs, colnames = _roi_table_columns(include_classification=columns is not None)
    plane_segmentation = img_seg.create_plane_segmentation(
        name="roi_table",
        description=approach.value["method"] + approach.value["description"],
        imaging_plane=imaging_plane,
        columns=column_defs,
        colnames=colnames,
    )
    _add_rois(plane_segmentation, masks, columns, extraction_columns)

    neuropil_image = _add_neuropil_segmentation(
        artifacts.extraction_h5, img_seg, imaging_plane, resolution
    )
    module.add(img_seg)

    images = _summary_images(artifacts, approach, resolution, neuropil_image)
    if images:
        module.add(
            Images(
                name="images",
                images=images,
                description="Summary images of the two-photon movie",
            )
        )

    contents = _add_traces(
        module,
        artifacts,
        plane_segmentation,
        num_rois,
        suffix,
        _series_kwargs(timestamps, metadata.frame_rate),
    )
    contents["classification"] = columns is not None
    # Derived from the two PNG paths, NOT from bool(images): _summary_images
    # appends the segmentation and neuropil mask images to that same list, so a
    # non-empty list does not mean either projection was found.
    contents["projections"] = (
        artifacts.average_projection_png is not None
        and artifacts.maximum_projection_png is not None
    )
    return contents, num_rois


def sync_timestamps_per_plane(
    planes: Sequence[Dict[str, Any]],
    sync_timestamps: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Split multiplane sync edges into per-plane timestamp arrays.

    Mesoscope planes are acquired in coupled pairs, so the vsync edge train is
    decimated by the number of temporal groups and offset by each plane's
    ``coupled_plane_index`` (v1 ``coupled_fov_index``).

    Parameters
    ----------
    planes : sequence of dict
        Planes in acquisition order.
    sync_timestamps : numpy.ndarray
        Rising edges of the 2p vsync line, in seconds.

    Returns
    -------
    dict of str to numpy.ndarray
        Per-plane timestamps, keyed by plane id.

    Raises
    ------
    ValueError
        If the plane count is odd, or if a plane's coupled index does not match
        its position -- either means the acquisition metadata and the coupling
        assumption disagree, and silently mis-timestamping every trace is worse
        than failing.
    """
    count = len(planes)
    if count % 2 != 0:
        raise ValueError(
            f"Multiplane session has {count} planes; coupled-pair "
            f"timestamp assignment requires an even number."
        )
    groups = count // 2
    result: Dict[str, np.ndarray] = {}
    for position, plane in enumerate(planes):
        group = position // 2
        coupled = plane["coupled_plane_index"]
        if coupled is not None and coupled != group:
            raise ValueError(
                f"Plane {plane['plane_id']} is at position {position} "
                f"(temporal group {group}) but declares coupled index "
                f"{coupled}; the acquisition metadata and "
                f"the plane ordering disagree."
            )
        result[plane["plane_id"]] = sync_timestamps[group::groups]
    return result


def add_epoch_intervals(
    nwbfile: NWBFile,
    epoch_json: Path,
    frame_rate: float,
) -> NWBFile:
    """Add single-plane epoch intervals from ``epoch_locations.json``.

    Parameters
    ----------
    nwbfile : NWBFile
        The file being built.
    epoch_json : Path
        Path to ``epoch_locations.json``, mapping epoch name to
        ``[start_frame, stop_frame]``.
    frame_rate : float
        Frame rate in Hz, used to convert frame indices to seconds.

    Returns
    -------
    NWBFile
        The updated file.
    """
    with open(epoch_json, "r") as f:
        epochs = json.load(f)
    table = TimeIntervals(
        name="epochs",
        description="Epochs for spontaneous and photostim periods",
        columns=[
            {"name": "stimulus_name", "description": "Epoch type name from key"},
            {"name": "start_frame", "description": "start_frame"},
            {"name": "stop_frame", "description": "stop_frame"},
        ],
    )
    for name, (start_frame, stop_frame) in epochs.items():
        if "neuron" in name and "slm" not in name:
            name = "BCI"
        elif "slm" in name and "post" not in name:
            name = "photostim"
        elif "slm" in name and "post" in name:
            name = "photostim_post"
        table.add_row(
            start_time=start_frame / frame_rate,
            stop_time=stop_frame / frame_rate,
            start_frame=start_frame,
            stop_frame=stop_frame,
            stimulus_name=name,
        )
    nwbfile.add_time_intervals(table)
    return nwbfile


def add_plane_lab_metadata(
    nwbfile: NWBFile,
    planes: Sequence[Dict[str, Any]],
    namespace_path: Path,
) -> None:
    """Attach the per-plane ndx-aibs-behavior-ophys lab metadata.

    Parameters
    ----------
    nwbfile : NWBFile
        The file being built.
    planes : sequence of dict
        Planes in acquisition order.
    namespace_path : Path
        Path to the extension namespace YAML.
    """
    ophys_metadata = load_pynwb_extension(namespace_path)
    for plane in planes:
        nwbfile.add_lab_meta_data(
            ophys_metadata(
                name=plane["plane_id"],
                imaging_depth=str(plane["depth"]) if plane["depth"] is not None else "",
                imaging_plane_group=(
                    str(plane["coupled_plane_index"])
                    if plane["coupled_plane_index"] is not None
                    else "-1"
                ),
                field_of_view_width=str(plane["width"]) if plane["width"] is not None else "",
                field_of_view_height=str(plane["height"]) if plane["height"] is not None else "",
            )
        )


def write_nwb_zarr(nwbfile: NWBFile, path: Path) -> Path:
    """Write the NWB file to a Zarr store.

    Parameters
    ----------
    nwbfile : NWBFile
        The completed file.
    path : Path
        Destination ``*.nwb.zarr`` directory.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NWBZarrIO(str(path), "w") as io:
        io.write(nwbfile)
    return path


def read_nwb_zarr(path: Path) -> Tuple[str, ...]:
    """Re-open a written Zarr store and return its processing-module names.

    Parameters
    ----------
    path : Path
        The ``*.nwb.zarr`` directory to read.

    Returns
    -------
    tuple of str
        The processing-module (plane) names found in the file.
    """
    with NWBZarrIO(str(path), "r") as io:
        return tuple(io.read().processing.keys())
