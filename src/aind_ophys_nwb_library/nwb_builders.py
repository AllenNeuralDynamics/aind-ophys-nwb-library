"""pynwb object construction from upstream ophys pipeline artifacts."""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynwb
import sparse
from hdmf.common import VectorData
from hdmf_zarr import NWBZarrIO
from pynwb import NWBFile
from pynwb.image import GrayscaleImage
from pynwb.ophys import (
    ImageSegmentation,
    ImagingPlane,
    OpticalChannel,
    PlaneSegmentation,
)

from .file_handling import PlaneArtifacts
from .utils.metadata_utils import NwbMetadata

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
        Path to ``ndx-aibs-behavior-ophys.namespace.yaml``.

    Returns
    -------
    type
        The generated ``OphysMetadataSchema`` class.

    Raises
    ------
    FileNotFoundError
        If the namespace YAML is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"pynwb extension namespace not found at {path}. It must be "
            f"committed under the Code Ocean capsule's code/ directory, which is the "
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

    Columns pair with the ROI masks by position; a length mismatch returns
    ``None``.

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
            "No classification file for %s; roi_table will omit the soma/dendrite columns.",
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

    A dataset absent from the extraction file becomes an all-NaN column.

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

    A multi-channel image is reduced to its first channel, with a warning.

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


def nwb_ophys_single_plane(
    nwbfile: pynwb.NWBFile,
    file_paths: dict,
    rig_json_data: dict,
    session_json_data: dict,
    procedures_json_data: dict,
    frame_rate: float,
) -> Tuple[pynwb.NWBFile, dict]:
    """Create an NWB file for single-plane ophys data
       using frame rate instead of timestamps

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file
    file_paths : dict
        The paths to the processed files
    rig_json_data : dict
        The rig metadata
    session_json_data : dict
        The session metadata
    procedures_json_data : dict
        The subject metadata
    frame_rate : float
        The frame rate in Hz (frames per second)

    Returns
    -------
    pynwb.NWBFile
        The NWB file
    dict
        The overall metadata
    """
    device, optical_channel = get_microscope(
        nwbfile, rig_json_data, session_json_data
    )

    # Get the single plane name from file_paths
    plane_name = list(file_paths["planes"].keys())[0]
    logging.info(f"Processing single plane: {plane_name}")

    ophys_fov = get_ophys_fovs(session_json_data)[0]
    structure = str(ophys_fov["targeted_structure"])
    imaging_depth = str(ophys_fov["imaging_depth"])
    step_unit = str(ophys_fov["fov_scale_factor_unit"])
    imaging_depth_unit = str(ophys_fov["imaging_depth_unit"])

    # Create imaging plane with appropriate metadata
    location = (
        "Structure: " + structure + " Depth: " + imaging_depth
    )  # Update if available in metadata

    ophys_module = nwbfile.create_processing_module(
        name=plane_name, description="Single-plane ophys processing module"
    )
    injection = ""
    for proc in procedures_json_data["subject_procedures"][0]["procedures"]:
        if "injection_materials" in proc:
            injection = proc["injection_materials"][0]["name"]

    imaging_plane = nwbfile.create_imaging_plane(
        name=plane_name,
        optical_channel=optical_channel,
        imaging_rate=frame_rate,
        description="2-photon imaging plane",
        device=device,
        excitation_lambda=get_excitation_wavelength(session_json_data),
        indicator=injection,
        location=location,
        unit="",
        grid_spacing=[1.0, 1.0],  # Update if available
        grid_spacing_unit=step_unit,
        origin_coords=[0.0, 0.0, 0.0],
        origin_coords_unit=imaging_depth_unit,
    )

    # Add ImageSegmentation
    img_seg = ImageSegmentation(name="image_segmentation")

    # Determine segmentation approach
    try:
        segmentation_approach = get_segementation_approach(
            file_paths["planes"][plane_name]["extraction_h5"]
        )
        plane_seg_approach = segmentation_approach.value["method"]
        plane_seg_descr = segmentation_approach.value["description"]
    except Exception as e:
        logging.warning(f"Could not determine segmentation approach: {e}")
        plane_seg_approach = "unknown"
        plane_seg_descr = "Unknown segmentation method"

    # Create plane segmentation
    try:
        # Try to load classifier data
        soma_predictions = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="soma",
            h5_key="predictions",
        )
        soma_probabilities = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="soma",
            h5_key="probabilities",
        )
        dendrite_predictions = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="dendrites",
            h5_key="predictions",
        )
        dendrite_probabilities = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="dendrites",
            h5_key="probabilities",
        )

        # Per-ROI cellpose probability from the extraction file. Only
        # present when segmentation ran cellpose; when absent (e.g.
        # activity-based segmentation) the column is still created but
        # filled with NaN, so the roi_table schema stays consistent.
        try:
            cellpose_soma_probability = load_generic_group(
                file_paths["planes"][plane_name]["extraction_h5"],
                h5_group="rois",
                h5_key="cellpose_soma_probability",
            )
        except Exception:
            cellpose_soma_probability = None

        # Per-ROI suite2p morphology stats (always present in the
        # extraction file's rois/ group).
        suite2p_stats = {
            stat: load_generic_group(
                file_paths["planes"][plane_name]["extraction_h5"],
                h5_group="rois",
                h5_key=stat,
            )
            for stat in SUITE2P_ROI_STATS
        }

        columns = [
            VectorData(
                name="is_soma",
                description="Soma predictions",
            ),
            VectorData(
                name="soma_probability",
                description="Soma probabilities",
            ),
            VectorData(
                name="is_dendrite",
                description="Dendrite predictions",
            ),
            VectorData(
                name="dendrite_probability",
                description="Dendrite probabilities",
            ),
            VectorData(
                name="cellpose_soma_probability",
                description=(
                    "Mean cellpose cell-probability under each ROI "
                    "footprint (probability in [0, 1]); NaN when "
                    "cellpose segmentation was not run"
                ),
            ),
        ]
        colnames = [
            "is_soma",
            "soma_probability",
            "is_dendrite",
            "dendrite_probability",
            "cellpose_soma_probability",
        ]
        for stat in SUITE2P_ROI_STATS:
            columns.append(
                VectorData(
                    name=stat,
                    description=f"suite2p ROI stat: {stat}",
                )
            )
            colnames.append(stat)

        plane_segmentation = img_seg.create_plane_segmentation(
            name="roi_table",
            description=plane_seg_approach + plane_seg_descr,
            imaging_plane=imaging_plane,
            columns=columns,
            colnames=colnames,
        )

        for idx, pixel_mask in enumerate(
            load_sparse_array(
                file_paths["planes"][plane_name]["extraction_h5"]
            )
        ):
            if cellpose_soma_probability is not None:
                cellpose_value = cellpose_soma_probability[idx]
            else:
                cellpose_value = np.nan
            plane_segmentation.add_roi(
                image_mask=pixel_mask,
                is_soma=soma_predictions[idx],
                soma_probability=soma_probabilities[idx][-1],
                is_dendrite=dendrite_predictions[idx],
                dendrite_probability=dendrite_probabilities[idx][-1],
                cellpose_soma_probability=cellpose_value,
                **{
                    stat: suite2p_stats[stat][idx]
                    for stat in SUITE2P_ROI_STATS
                },
            )
    except Exception as e:
        logging.warning(f"Error adding ROIs with classifier data: {e}")
        # Fallback to simpler plane segmentation without classifier data
        plane_segmentation = img_seg.create_plane_segmentation(
            name="roi_table",
            description=plane_seg_approach + plane_seg_descr,
            imaging_plane=imaging_plane,
        )
        for idx, pixel_mask in enumerate(
            load_sparse_array(
                file_paths["planes"][plane_name]["extraction_h5"]
            )
        ):
            plane_segmentation.add_roi(image_mask=pixel_mask)

    neuropil_img = None
    if segmentation_approach in (
        SegmentationApproach.SUITE2P_ANATOMICAL,
        SegmentationApproach.SUITE2P_ACTIVITY,
    ):
        neuropil_img = add_neuropil_segmentation_and_r_values(
            file_paths["planes"][plane_name]["extraction_h5"],
            img_seg,
            imaging_plane,
        )
    else:
        logging.info(
            "Skipping neuropil segmentation. Currently only supported "
            "for suite2p-based segmentation approaches. "
            f"Segmentation approach used: {segmentation_approach}"
        )

    ophys_module.add(img_seg)

    # Add projections and images
    try:
        # Add average and max projections if available
        avg_projection = plt.imread(
            file_paths["planes"][plane_name]["average_projection_png"]
        )
        avg_img = GrayscaleImage(
            name="average_projection",
            data=avg_projection,
            resolution=1.0,  # Update if available
            description="Average intensity projection of entire session",
        )

        max_projection = plt.imread(
            file_paths["planes"][plane_name]["max_projection_png"]
        )
        max_img = GrayscaleImage(
            name="max_projection",
            data=max_projection,
            resolution=1.0,  # Update if available
            description="Max intensity projection of entire session",
        )

        # Try to add segmentation mask if available
        summary_images = [avg_img, max_img]
        if segmentation_approach == SegmentationApproach.SUITE2P_ANATOMICAL:
            segmentation_mask = load_generic_group(
                file_paths["planes"][plane_name]["extraction_h5"],
                h5_group="cellpose",
                h5_key="masks",
            )
            mask_img = GrayscaleImage(
                name="segmentation_mask_image",
                data=segmentation_mask,
                resolution=1.0,  # Update if available
                description="Segmentation projection of entire session",
            )
            summary_images.append(mask_img)

        if neuropil_img is not None:
            summary_images.append(neuropil_img)

        images = Images(
            name="images",
            images=summary_images,
            description="Summary images of the ophys movie",
        )

        ophys_module.add(images)
    except Exception as e:
        logging.warning(f"Error adding projection images: {e}")
    """
    if file_paths['planes'][plane_name]["classifier_h5"] is None:
        return nwbfile
    """
    # Add timeseries data
    try:
        # ROI traces
        roi_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["extraction_h5"],
            plane_segmentation,
            h5_group="traces",
            h5_key="roi",
        )

        # Create time series using frame rate
        roi_traces_series = RoiResponseSeries(
            name="roi_fluorescence",
            data=roi_traces.T,
            rois=roi_names,
            unit="a.u.",
            rate=frame_rate,
            starting_time=0.0,
        )

        # Neuropil traces
        neuropil_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["extraction_h5"],
            plane_segmentation,
            h5_group="traces",
            h5_key="neuropil",
        )

        neuropil_traces_series = RoiResponseSeries(
            name="neuropil_fluorescence",
            data=neuropil_traces.T,
            rois=roi_names,
            unit="a.u.",
            rate=frame_rate,
            starting_time=0.0,
        )

        # Neuropil corrected traces if available
        try:
            neuropil_corrected, roi_names = load_signals(
                file_paths["planes"][plane_name]["extraction_h5"],
                plane_segmentation,
                h5_group="traces",
                h5_key="corrected",
            )

            neuropil_corrected_series = RoiResponseSeries(
                name="neuropil_corrected",
                data=neuropil_corrected.T,
                rois=roi_names,
                unit="a.u.",
                rate=frame_rate,
                starting_time=0.0,
            )

            ophys_module.add(neuropil_corrected_series)
        except Exception as e:
            logging.warning(f"Error adding neuropil corrected traces: {e}")

        # DfOverF traces if available
        try:
            dfof_traces, roi_names = load_signals(
                file_paths["planes"][plane_name]["dff_h5"],
                plane_segmentation,
                h5_key="data",
            )

            dfof_traces_series = RoiResponseSeries(
                name="dff",
                data=dfof_traces.T,
                rois=roi_names,
                unit="%",
                rate=frame_rate,
                starting_time=0.0,
            )

            ophys_module.add(
                DfOverF(roi_response_series=dfof_traces_series, name="dff")
            )
        except Exception as e:
            logging.warning(f"Error adding DfOverF traces: {e}")

        # Event traces if available
        try:
            event_traces, roi_names = load_signals(
                file_paths["planes"][plane_name]["events_oasis_h5"],
                plane_segmentation,
                h5_key="events",
            )

            event_traces_series = RoiResponseSeries(
                name="event",
                data=event_traces.T,
                rois=roi_names,
                unit="a.u.",
                rate=frame_rate,
                starting_time=0.0,
            )

            ophys_module.add(event_traces_series)
        except Exception as e:
            logging.warning(f"Error adding event traces: {e}")

        # Add the base fluorescence data to the module
        ophys_module.add(
            Fluorescence(roi_response_series=roi_traces_series, name="raw")
        )
        ophys_module.add(neuropil_traces_series)

    except Exception as e:
        logging.error(f"Error adding timeseries data: {e}")
        raise

    return nwbfile


def nwb_ophys(
    nwbfile: pynwb.NWBFile,
    file_paths: dict,
    ophys_fovs: list,
    rig_json_data: dict,
    session_json_data: dict,
    subject_json_data: dict,
) -> Tuple[pynwb.NWBFile, dict]:
    """Create an NWB file for ophys data

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file
    file_paths : dict
        The paths to the processed files
    ophys_fovs : list
        The session metadata
    rig_json_data : dict
        The rig metadata
    session_json_data : dict
        The session metadata
    subject_json_data : dict
        The subject metadata
    processing_json_data : dict
        The processing metadata

    Returns
    -------
    pynwb.NWBFile
        The NWB file
    dict
        The overall metadata
    """
    device, optical_channel = get_microscope(
        nwbfile, rig_json_data, session_json_data
    )
    # Start plane specific processing
    for plane in ophys_fovs:
        plane_name = f"{plane['targeted_structure']}_{plane['index']}"
        logging.info(f"Processing plane: {plane_name}")
        location = (
            "Structure: "
            + plane["targeted_structure"]
            + " Depth: "
            + str(plane["imaging_depth"])
        )
        ophys_module = nwbfile.create_processing_module(
            name=plane_name, description=""
        )
        imaging_plane = nwbfile.create_imaging_plane(
            name=plane_name,  # ophys_plane_id
            optical_channel=optical_channel,
            imaging_rate=float(plane["frame_rate"]),
            description="Two-photon imaging plane",
            device=device,
            excitation_lambda=float(
                session_json_data["data_streams"][0]["light_sources"][0][
                    "wavelength"
                ]
            ),
            indicator=subject_json_data["genotype"],
            location=location,
            grid_spacing=[
                float(plane["fov_scale_factor"]),
                float(plane["fov_scale_factor"]),
            ],
            grid_spacing_unit=plane["fov_coordinate_unit"],
            origin_coords=[0.0, 0.0, 0.0],  # TODO: dunno
            origin_coords_unit=plane["fov_coordinate_unit"],
        )
        segmentation_approach = get_segementation_approach(
            file_paths["planes"][plane_name]["extraction_h5"]
        )
        plane_seg_approach = segmentation_approach.value["method"]
        plane_seg_descr = segmentation_approach.value["description"]
        img_seg = ImageSegmentation(name="image_segmentation")
        soma_predictions = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="soma",
            h5_key="predictions",
        )
        soma_probabilities = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="soma",
            h5_key="probabilities",
        )
        dendrite_predictions = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="dendrites",
            h5_key="predictions",
        )
        dendrite_probabilities = load_generic_group(
            file_paths["planes"][plane_name]["classifier_h5"],
            h5_group="dendrites",
            h5_key="probabilities",
        )

        # Per-ROI cellpose probability from the extraction file. Only
        # present when segmentation ran cellpose; when absent (e.g.
        # activity-based segmentation) the column is still created but
        # filled with NaN, so the roi_table schema stays consistent.
        try:
            cellpose_soma_probability = load_generic_group(
                file_paths["planes"][plane_name]["extraction_h5"],
                h5_group="rois",
                h5_key="cellpose_soma_probability",
            )
        except Exception:
            cellpose_soma_probability = None

        # Per-ROI suite2p morphology stats (always present in the
        # extraction file's rois/ group).
        suite2p_stats = {
            stat: load_generic_group(
                file_paths["planes"][plane_name]["extraction_h5"],
                h5_group="rois",
                h5_key=stat,
            )
            for stat in SUITE2P_ROI_STATS
        }

        columns = [
            VectorData(
                name="is_soma",
                description="Soma predictions",
            ),
            VectorData(
                name="soma_probability",
                description="Soma probabilities",
            ),
            VectorData(
                name="is_dendrite",
                description="Dendrite predictions",
            ),
            VectorData(
                name="dendrite_probability",
                description="Dendrite probabilities",
            ),
            VectorData(
                name="cellpose_soma_probability",
                description=(
                    "Mean cellpose cell-probability under each ROI "
                    "footprint (probability in [0, 1]); NaN when "
                    "cellpose segmentation was not run"
                ),
            ),
        ]
        colnames = [
            "is_soma",
            "soma_probability",
            "is_dendrite",
            "dendrite_probability",
            "cellpose_soma_probability",
        ]
        for stat in SUITE2P_ROI_STATS:
            columns.append(
                VectorData(
                    name=stat,
                    description=f"suite2p ROI stat: {stat}",
                )
            )
            colnames.append(stat)

        plane_segmentation = img_seg.create_plane_segmentation(
            name="roi_table",
            description=plane_seg_approach + plane_seg_descr,
            imaging_plane=imaging_plane,
            columns=columns,
            colnames=colnames,
        )

        neuropil_img = None
        if segmentation_approach in (
            SegmentationApproach.SUITE2P_ANATOMICAL,
            SegmentationApproach.SUITE2P_ACTIVITY,
        ):
            neuropil_img = add_neuropil_segmentation_and_r_values(
                file_paths["planes"][plane_name]["extraction_h5"],
                img_seg,
                imaging_plane,
                resolution=float(plane["fov_scale_factor"]),
            )
        else:
            logging.info(
                "Skipping neuropil segmentation. Currently only supported "
                "for suite2p-based segmentation approaches. "
                f"Segmentation approach used: {segmentation_approach}"
            )

        ophys_module.add(img_seg)

        avg_projection = plt.imread(
            file_paths["planes"][plane_name]["average_projection_png"]
        )
        avg_img = GrayscaleImage(
            name="average_projection",
            data=avg_projection,
            resolution=float(plane["fov_scale_factor"]),
            description="Average intensity projection of entire session",
        )
        max_projection = plt.imread(
            file_paths["planes"][plane_name]["max_projection_png"]
        )
        max_img = GrayscaleImage(
            name="max_projection",
            data=max_projection,
            resolution=float(plane["fov_scale_factor"]),  # pixels/cm
            description="Max intensity projection of entire session",
        )
        if segmentation_approach == SegmentationApproach.SUITE2P_ANATOMICAL:
            segmentation_mask = load_generic_group(
                file_paths["planes"][plane_name]["extraction_h5"],
                h5_group="cellpose",
                h5_key="masks",
            )
        else:
            segmentation_mask = convert_rois_to_segmentation_mask(
                file_paths["planes"][plane_name]["extraction_h5"]
            )
        mask_img = GrayscaleImage(
            name="segmentation_mask_image",
            data=segmentation_mask,
            resolution=float(plane["fov_scale_factor"]),  # pixels/cm
            description="Segmentation projection of entire session",
        )

        summary_images = [avg_img, max_img, mask_img]
        if neuropil_img is not None:
            summary_images.append(neuropil_img)

        images = Images(
            name="images",
            images=summary_images,
            description="Summary images of the two-photon movie",
        )
        ophys_module.add(images)

        # 4D. ROIS

        rois_shape = load_generic_group(
            file_paths["planes"][plane_name]["extraction_h5"],
            h5_group="rois",
            h5_key="shape",
        )
        for idx, pixel_mask in enumerate(
            load_sparse_array(
                file_paths["planes"][plane_name]["extraction_h5"]
            )
        ):
            if cellpose_soma_probability is not None:
                cellpose_value = cellpose_soma_probability[idx]
            else:
                cellpose_value = np.nan
            plane_segmentation.add_roi(
                image_mask=pixel_mask,
                is_soma=soma_predictions[idx],
                # last element is the probability
                soma_probability=soma_probabilities[idx][-1],
                is_dendrite=dendrite_predictions[idx],
                dendrite_probability=dendrite_probabilities[idx][-1],
                cellpose_soma_probability=cellpose_value,
                **{
                    stat: suite2p_stats[stat][idx]
                    for stat in SUITE2P_ROI_STATS
                },
            )

        roi_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["extraction_h5"],
            plane_segmentation,
            h5_group="traces",
            h5_key="roi",
        )
        roi_traces_series = RoiResponseSeries(
            name="roi_fluorescence_timeseries",
            data=roi_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps=plane["timestamps"],
        )

        assert roi_traces.shape[0] == int(
            rois_shape[0]
        ), "Mismatch in number of ROIs and traces"

        neuropil_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["extraction_h5"],
            plane_segmentation,
            h5_group="traces",
            h5_key="neuropil",
        )
        neuropil_traces_series = RoiResponseSeries(
            name="neuropil_fluorescence_timeseries",
            data=neuropil_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps=plane["timestamps"],
        )

        assert neuropil_traces.shape[0] == int(
            rois_shape[0]
        ), "Mismatch in number of ROIs and traces"

        neuropil_corrected, roi_names = load_signals(
            file_paths["planes"][plane_name]["extraction_h5"],
            plane_segmentation,
            h5_group="traces",
            h5_key="corrected",
        )
        neuropil_corrected_series = RoiResponseSeries(
            name="neuropil_corrected_timeseries",
            data=neuropil_corrected.T,
            rois=roi_names,
            unit="a.u.",
            timestamps=plane["timestamps"],
        )

        assert neuropil_corrected.shape[0] == int(
            rois_shape[0]
        ), "Mismatch in number of ROIs and traces"

        dfof_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["dff_h5"],
            plane_segmentation,
            h5_key="data",
        )

        dfof_traces_series = RoiResponseSeries(
            name="dff_timeseries",
            data=dfof_traces.T,
            rois=roi_names,
            unit="%",
            timestamps=plane["timestamps"],
        )

        assert dfof_traces.shape[0] == int(
            rois_shape[0]
        ), "Mismatch in number of ROIs and traces"

        event_traces, roi_names = load_signals(
            file_paths["planes"][plane_name]["events_oasis_h5"],
            plane_segmentation,
            h5_key="events",
        )

        event_traces_series = RoiResponseSeries(
            name="event_timeseries",
            data=event_traces.T,
            rois=roi_names,
            unit="a.u.",
            timestamps=plane["timestamps"],
        )

        assert event_traces.shape[0] == int(
            rois_shape[0]
        ), "Mismatch in number of ROIs and traces"

        ophys_module.add(
            DfOverF(
                roi_response_series=dfof_traces_series, name="dff_timeseries"
            )
        )

        ophys_module.add(
            Fluorescence(
                roi_response_series=roi_traces_series, name="raw_timeseries"
            )
        )
        ophys_module.add(neuropil_traces_series)
        ophys_module.add(neuropil_corrected_series)
        ophys_module.add(event_traces_series)
    return nwbfile


def sync_times_to_multiplane_fovs(
    ophys_fovs: list, sync_timestamps: np.array
) -> list:
    """Convert the timestamps to FOVs for multiplane only

    Parameters
    ----------
    ophys_fovs : list
        The FOVs
    sync_timestamps : np.array
        The sync timestamps

    Returns
    -------
    list
        The FOVs with timestamps
    """
    planes = len(ophys_fovs)
    if planes % 2 != 0:
        raise Exception("Odd number of planes, please check code")
    # Planes are paired, so we only want to get half of them
    image_groups = int(planes / 2)
    for plane_group in range(image_groups):
        for indiv_plane_in_group in range(2):
            index_plane = plane_group * 2 + indiv_plane_in_group
            # We double check, this is important
            if plane_group != ophys_fovs[index_plane]["coupled_fov_index"]:
                raise Exception("Mismatched temporal group, please check code")
            ophys_fovs[index_plane]["timestamps"] = sync_timestamps[
                plane_group::image_groups
            ]
    return ophys_fovs


def _sync_timestamps(sync_fp: Path) -> np.array:
    """Get the sync timestamps from the sync file
    Parameters
    ----------
    sync_fp : Path
        The path to the sync file
    Returns
    -------
    np.array
        The sync timestamps
    """
    sync_dataset = SyncDataset(sync_fp)

    try:
        return sync_dataset.get_rising_edges(
            line="2p_vsync",
            units="seconds"
        )
    except ValueError:
        logging.warning("Failed to find 2p_vsync line. Trying with vsync_2p")
        return sync_dataset.get_rising_edges(
            line="vsync_2p",
            units="seconds"
        )


def add_intervals_sp_nwb(json_path, frame_rate, nwbfile):
    """
    Add epoch intervals to the NWB file based on epochs.json.

    Parameters
    ----------
    json_path : Path or str
        Path to epochs.json file. Format:
            {epoch_name: [start_frame, stop_frame]}
    frame_rate : float
        Frame rate in Hz used to convert frame indices to seconds.
    nwbfile : NWBFile
        The NWB file to which the epochs will be added.

    Returns
    -------
    NWBFile
        The updated NWB file with epochs added to the TimeIntervals table.
    """
    with open(json_path, "r") as f:
        epoch_data = json.load(f)

    # Create a TimeIntervals table
    trial_table = TimeIntervals(
        name="epochs",
        description="Epochs for spontaneous and photostim periods",
        columns=[
            {
                "name": "stimulus_name",
                "description": "Epoch type name from key",
            },
            {"name": "start_frame", "description": "start_frame"},
            {"name": "stop_frame", "description": "stop_frame"},
        ],
    )

    # Add intervals
    for epoch_name, (start_frame, stop_frame) in epoch_data.items():
        if "neuron" in epoch_name and "slm" not in epoch_name:
            epoch_name = "BCI"
        elif "slm" in epoch_name and "post" not in epoch_name:
            epoch_name = "photostim"
        elif "slm" in epoch_name and "post" in epoch_name:
            epoch_name = "photostim_post"
        start_time = start_frame / frame_rate
        stop_time = stop_frame / frame_rate

        trial_table.add_row(
            start_time=start_time,
            stop_time=stop_time,
            start_frame=start_frame,
            stop_frame=stop_frame,
            stimulus_name=epoch_name,
        )

    # Add the table to the NWB file
    nwbfile.add_time_intervals(trial_table)

    return nwbfile


if __name__ == "__main__":
    args = parse_args()
    input_directory = Path(args.input_directory)
    output_directory = Path(args.output_directory)

    # Load data description
    json_path = next(input_directory.rglob("data_description.json"))
    with open(json_path, "r") as f:
        data_description = json.load(f)

    # Get processed and raw paths
    raw_data_fp = input_directory / "raw"
    processed_data_fp = input_directory / "processed"

    session_type = detect_session_type(data_description, processed_data_fp)
    logging.info(f"Detected session type: {session_type.value}")

    # Pull session, subject, rig, and procedures metadata
    session_data, subject_data, rig_data, procedures_data = get_metadata(
        raw_data_fp
    )
    try:
        ophys_fovs = session_data["data_streams"][1]["ophys_fovs"]
    except IndexError:
        ophys_fovs = session_data["data_streams"][0]["ophys_fovs"]

    file_paths = get_processed_file_paths(
        processed_data_fp,
        raw_data_fp,
        ophys_fovs,
        single_plane=session_type != SessionType.MULTIPLANE,
    )

    # Create a new NWB file using aind-nwb-utils
    nwb_file = nwb_utils.create_base_nwb_file(raw_data_fp)

    output_nwb_fp = output_directory / "pophys.nwb.zarr"
    io = NWBZarrIO(str(output_nwb_fp), "w")

    if session_type == SessionType.MULTIPLANE:
        sync_timestamps = get_sync_timestamps(raw_data_fp)
        ophys_fovs = sync_times_to_multiplane_fovs(ophys_fovs, sync_timestamps)

        nwb_file = nwb_ophys(
            nwb_file,
            file_paths,
            ophys_fovs,
            rig_data,
            session_data,
            subject_data,
        )

        # Load extension for plane metadata
        name_space = "../code/ndx-aibs-behavior-ophys.namespace.yaml"
        if not Path(name_space).is_file():
            raise FileNotFoundError(name_space)
        OphysMetadata = load_pynwb_extension(name_space)

        for fov in ophys_fovs:
            plane_metadata = OphysMetadata(
                name=f'{fov["targeted_structure"]}_{fov["index"]}',
                imaging_depth=str(fov["imaging_depth"]),
                imaging_plane_group=str(fov["coupled_fov_index"]),
                field_of_view_width=str(fov["fov_width"]),
                field_of_view_height=str(fov["fov_height"]),
            )
            nwb_file.add_lab_meta_data(plane_metadata)
    else:
        frame_rate = get_frame_rate(session_data)

        if session_type == SessionType.SINGLE_PLANE_WITH_EPOCHS:
            epoch_path = find_epoch_locations(processed_data_fp)
            nwb_file = add_intervals_sp_nwb(epoch_path, frame_rate, nwb_file)

        nwb_file = nwb_ophys_single_plane(
            nwb_file,
            file_paths,
            rig_data,
            session_data,
            procedures_data,
            frame_rate,
        )

    # Write out the NWB file
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing NWB file to {output_nwb_fp}")
    io.write(nwb_file)
