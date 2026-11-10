"""Tests for the pynwb builders over real HDF5 payloads."""

import json
import logging
import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import imageio.v3 as iio
import numpy as np
from pynwb import NWBFile

from aind_ophys_nwb_library import nwb_builders as nb
from aind_ophys_nwb_library.file_handling import PlaneArtifacts
from aind_ophys_nwb_library.utils.metadata_utils import NwbInstrument, NwbMetadata, NwbSession

# Production shapes, scaled down only in the frame count: masks and
# projections are full field of view, traces are (n_rois, n_frames).
N_ROIS = 3
N_FRAMES = 4000
HEIGHT = 512
WIDTH = 512

PLANE_ID = "VISp_0"
FRAME_RATE = 9.48
SCALE_FACTOR = 0.78


def roi_masks(n_rois: int = N_ROIS) -> np.ndarray:
    """Return dense per-ROI boolean masks of shape ``(n_rois, H, W)``.

    Parameters
    ----------
    n_rois : int, optional
        Number of ROIs; ``0`` gives the zero-ROI CaImAn-branch shape.

    Returns
    -------
    numpy.ndarray
        Boolean masks, one 8x8 footprint per ROI, none overlapping.
    """
    masks = np.zeros((n_rois, HEIGHT, WIDTH), dtype=bool)
    for index in range(n_rois):
        masks[index, 10 * index : 10 * index + 8, 20:28] = True
    return masks


def write_extraction(
    path: Path,
    masks: np.ndarray,
    n_frames: int = N_FRAMES,
    with_corrected: bool = True,
    with_neuropil_masks: bool = True,
    with_roi_stats: bool = True,
    cellpose: bool = False,
) -> Path:
    """Write an extraction HDF5 in the layout aind-ophys-extraction emits.

    Parameters
    ----------
    path : Path
        Destination file.
    masks : numpy.ndarray
        Dense per-ROI masks; stored sparsely as ``rois/{data,coords,shape}``.
    n_frames : int, optional
        Trace length.
    with_corrected : bool, optional
        Whether to write the optional neuropil-corrected traces.
    with_neuropil_masks : bool, optional
        Whether to write the neuropil coordinates and r-values.
    with_roi_stats : bool, optional
        Whether to write the per-ROI suite2p morphology stats.
    cellpose : bool, optional
        Whether to write a ``cellpose`` group, which marks anatomical mode.

    Returns
    -------
    Path
        The path written.
    """
    n_rois = masks.shape[0]
    coords = np.array(np.nonzero(masks))
    with h5py.File(path, "w") as handle:
        rois = handle.create_group("rois")
        rois["data"] = np.ones(coords.shape[1], dtype=bool)
        rois["coords"] = coords
        rois["shape"] = np.array(masks.shape)
        if with_neuropil_masks:
            rois["neuropil_coords"] = coords
        if with_roi_stats:
            rois["cellpose_soma_probability"] = np.full(n_rois, 0.8)
            for stat in nb.SUITE2P_ROI_STATS:
                rois[stat] = np.full(n_rois, 1.5)
        traces = handle.create_group("traces")
        traces["roi"] = np.zeros((n_rois, n_frames), dtype="float32")
        traces["neuropil"] = np.zeros((n_rois, n_frames), dtype="float32")
        if with_corrected:
            traces["corrected"] = np.zeros((n_rois, n_frames), dtype="float32")
        if with_neuropil_masks:
            traces["neuropil_rcoef"] = np.full(n_rois, 0.7)
        if cellpose:
            handle.create_group("cellpose")["masks"] = np.zeros((HEIGHT, WIDTH), dtype="i2")
    return path


def write_single_dataset(path: Path, key: str, data: np.ndarray) -> Path:
    """Write one root-level dataset, as the dF/F and OASIS steps do.

    Parameters
    ----------
    path : Path
        Destination file.
    key : str
        Dataset name (``"data"`` for dF/F, ``"events"`` for OASIS).
    data : numpy.ndarray
        The array to store.

    Returns
    -------
    Path
        The path written.
    """
    with h5py.File(path, "w") as handle:
        handle[key] = data
    return path


def write_classification(path: Path, n_rois: int = N_ROIS) -> Path:
    """Write a classification HDF5 in the layout aind-ophys-classifier emits.

    ``probabilities`` is ``predict_proba`` output, so it is ``(n_rois,
    n_classes)`` and the builder reads the positive class off the last column.

    Parameters
    ----------
    path : Path
        Destination file.
    n_rois : int, optional
        Number of ROIs; a value disagreeing with the mask count is what the
        positional-pairing guard rejects.

    Returns
    -------
    Path
        The path written.
    """
    probabilities = np.column_stack((np.full(n_rois, 0.1), np.full(n_rois, 0.9)))
    with h5py.File(path, "w") as handle:
        for group in ("soma", "dendrites"):
            created = handle.create_group(group)
            created["predictions"] = np.ones(n_rois, dtype=bool)
            created["probabilities"] = probabilities
    return path


def write_projection(path: Path, rgba: bool = False) -> Path:
    """Write a projection PNG the way motion correction writes one.

    Parameters
    ----------
    path : Path
        Destination file.
    rgba : bool, optional
        Whether to write a 4-channel image instead of the 2-D uint8 one, which
        is the shape :func:`nwb_builders._as_grayscale` collapses.

    Returns
    -------
    Path
        The path written.
    """
    shape = (HEIGHT, WIDTH, 4) if rgba else (HEIGHT, WIDTH)
    iio.imwrite(path, np.zeros(shape, dtype="uint8"))
    return path


def plane_metadata(
    plane_id: str = PLANE_ID,
    index: int = 0,
    coupled: Optional[int] = 0,
) -> Dict[str, Any]:
    """Return one plane's imaging metadata record.

    Parameters
    ----------
    plane_id : str, optional
        Canonical plane id.
    index : int, optional
        Plane index.
    coupled : int, optional
        Coupled-pair index (``plane_group``); ``None`` for metadata that
        omits the coupling.

    Returns
    -------
    dict
        The plane record, in the shape ``load_nwb_metadata`` produces.
    """
    return {
        "plane_id": plane_id,
        "plane_index": index,
        "structure": plane_id.split("_")[0],
        "depth": 150.0 + 50 * index,
        "depth_unit": "micrometer",
        "coupled_plane_index": coupled,
        "plane_group": coupled,
        "scale_factor": SCALE_FACTOR,
        "scale_factor_unit": "micrometer",
        "um_per_pixel": SCALE_FACTOR,
        "width": WIDTH,
        "height": HEIGHT,
    }


def nwb_metadata(core_file: Path, planes: Sequence[Dict[str, Any]]) -> NwbMetadata:
    """Return the acquisition metadata the builders consume.

    Parameters
    ----------
    core_file : Path
        Path recorded as the source core metadata file.
    planes : sequence of dict
        Planes in acquisition order.

    Returns
    -------
    NwbMetadata
        The acquisition record.
    """
    return NwbMetadata(
        schema_version="2.0",
        core_file=core_file,
        planes=tuple(planes),
        frame_rate=FRAME_RATE,
        instrument_id="MESO.1",
        excitation_wavelength=920.0,
        emission_wavelength=520.0,
        indicator="GCaMP6f",
    )


class ExtractionFileCase(unittest.TestCase):
    """Base case owning a temporary directory with one plane's artifacts."""

    def setUp(self) -> None:
        """Write a complete set of upstream artifacts for one plane."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.masks = roi_masks()
        self.extraction = write_extraction(self.root / f"{PLANE_ID}_extraction.h5", self.masks)
        self.dff = write_single_dataset(
            self.root / f"{PLANE_ID}_dff.h5",
            "data",
            np.zeros((N_ROIS, N_FRAMES), dtype="float32"),
        )
        self.events = write_single_dataset(
            self.root / f"{PLANE_ID}_events_oasis.h5",
            "events",
            np.zeros((N_ROIS, N_FRAMES), dtype="float32"),
        )
        self.classification = write_classification(self.root / f"{PLANE_ID}_classification.h5")
        self.average_png = write_projection(self.root / f"{PLANE_ID}_average_projection.png")
        self.maximum_png = write_projection(self.root / f"{PLANE_ID}_maximum_projection.png")

    def artifacts(self, **overrides: Any) -> PlaneArtifacts:
        """Return this plane's resolved artifacts, overridable per test.

        Parameters
        ----------
        **overrides
            Fields to replace, e.g. ``classification_h5=None``.

        Returns
        -------
        PlaneArtifacts
            The artifact record.
        """
        fields = {
            "plane_id": PLANE_ID,
            "extraction_h5": self.extraction,
            "dff_h5": self.dff,
            "events_oasis_h5": self.events,
            "classification_h5": self.classification,
            "average_projection_png": self.average_png,
            "maximum_projection_png": self.maximum_png,
        }
        fields.update(overrides)
        return PlaneArtifacts(**fields)


class TestDatasetReads(ExtractionFileCase):
    """The HDF5 reads every builder is fed from."""

    def test_load_dataset_returns_an_array(self) -> None:
        """A trace read comes back as a numpy array at its stored shape."""
        traces = nb.load_dataset(self.extraction, "roi", "traces")
        self.assertIsInstance(traces, np.ndarray)
        self.assertEqual(traces.shape, (N_ROIS, N_FRAMES))

    def test_load_dataset_reads_the_root_group(self) -> None:
        """dF/F and OASIS write at the root, with no group."""
        self.assertEqual(nb.load_dataset(self.dff, "data").shape, (N_ROIS, N_FRAMES))

    def test_try_load_dataset_returns_none_without_a_file(self) -> None:
        """An upstream stage that never ran yields None, not an exception."""
        self.assertIsNone(nb.try_load_dataset(None, "data"))

    def test_try_load_dataset_logs_a_missing_key(self) -> None:
        """A renamed dataset is named in the warning rather than swallowed."""
        with self.assertLogs(nb.logger, level=logging.WARNING) as logs:
            self.assertIsNone(nb.try_load_dataset(self.extraction, "absent", "traces"))
        self.assertIn("traces/absent", logs.output[0])

    def test_load_roi_masks_expands_the_sparse_footprints(self) -> None:
        """The sparse rois/ group round-trips to the dense mask array."""
        masks = nb.load_roi_masks(self.extraction)
        self.assertEqual(masks.shape, (N_ROIS, HEIGHT, WIDTH))
        np.testing.assert_array_equal(masks, self.masks)

    def test_load_neuropil_masks_and_r_values(self) -> None:
        """Neuropil coordinates expand to a boolean mask of the same shape."""
        masks, r_values = nb.load_neuropil_masks_and_r_values(self.extraction)
        self.assertEqual(masks.shape, (N_ROIS, HEIGHT, WIDTH))
        self.assertEqual(masks.dtype, np.dtype(bool))
        self.assertEqual(r_values.shape, (N_ROIS,))

    def test_segmentation_approach_detects_cellpose(self) -> None:
        """A cellpose group means anatomical mode, its absence activity mode."""
        anatomical = write_extraction(
            self.root / "anatomical_extraction.h5", self.masks, cellpose=True
        )
        self.assertIs(
            nb.segmentation_approach(anatomical),
            nb.SegmentationApproach.SUITE2P_ANATOMICAL,
        )
        self.assertIs(
            nb.segmentation_approach(self.extraction),
            nb.SegmentationApproach.SUITE2P_ACTIVITY,
        )


class TestCollapseMasksToLabelImage(unittest.TestCase):
    """The 3-D mask array to 2-D label image collapse."""

    def test_boolean_masks(self) -> None:
        """Each footprint takes its own 1-based label; background stays 0."""
        label_image = nb.collapse_masks_to_label_image(roi_masks())
        self.assertEqual(label_image.shape, (HEIGHT, WIDTH))
        np.testing.assert_array_equal(np.unique(label_image), np.arange(N_ROIS + 1))

    def test_weighted_masks(self) -> None:
        """Float footprints take the sum-based branch and label identically."""
        label_image = nb.collapse_masks_to_label_image(roi_masks().astype("float32"))
        np.testing.assert_array_equal(np.unique(label_image), np.arange(N_ROIS + 1))

    def test_zero_rois_yields_an_empty_label_image(self) -> None:
        """extraction's CaImAn branch emits no masks; argmax would raise."""
        label_image = nb.collapse_masks_to_label_image(roi_masks(0))
        self.assertEqual(label_image.shape, (HEIGHT, WIDTH))
        self.assertFalse(label_image.any())


class TestRoiTableColumns(ExtractionFileCase):
    """The per-ROI columns, all paired to the masks by positional index."""

    def test_classification_columns_load(self) -> None:
        """All four classifier columns come back as arrays of n_rois."""
        columns = nb._classification_columns(self.artifacts(), N_ROIS)
        self.assertEqual(
            set(columns),
            {"is_soma", "soma_probability", "is_dendrite", "dendrite_probability"},
        )
        for name, values in columns.items():
            with self.subTest(column=name):
                self.assertIsInstance(values, np.ndarray)
                self.assertEqual(len(values), N_ROIS)

    def test_probabilities_are_two_dimensional(self) -> None:
        """The builder reads the positive class as ``[idx][-1]``.

        The classifier writes ``predict_proba`` output, so ``probabilities`` is
        ``(n_rois, n_classes)``. A 1-D array would pass the length check and
        then raise on the trailing index.
        """
        columns = nb._classification_columns(self.artifacts(), N_ROIS)
        self.assertEqual(columns["soma_probability"].ndim, 2)
        self.assertEqual(columns["soma_probability"].shape, (N_ROIS, 2))

    def test_no_classification_file_warns_and_omits(self) -> None:
        """A skipped classifier drops its columns, not the whole plane."""
        with self.assertLogs(nb.logger, level=logging.WARNING):
            self.assertIsNone(nb._classification_columns(self.artifacts(classification_h5=None), 3))

    def test_roi_count_mismatch_is_rejected(self) -> None:
        """Pairing is positional, so a disagreeing count must not truncate."""
        mismatched = write_classification(
            self.root / "mismatch_classification.h5", n_rois=N_ROIS + 1
        )
        with self.assertLogs(nb.logger, level=logging.ERROR) as logs:
            columns = nb._classification_columns(
                self.artifacts(classification_h5=mismatched), N_ROIS
            )
        self.assertIsNone(columns)
        self.assertIn("positional ROI index", logs.output[0])

    def test_missing_classification_group_omits_the_columns(self) -> None:
        """A truncated classification file yields None rather than partial."""
        empty = self.root / "empty_classification.h5"
        with h5py.File(empty, "w") as handle:
            handle.create_group("unrelated")
        with self.assertLogs(nb.logger, level=logging.WARNING):
            self.assertIsNone(
                nb._classification_columns(self.artifacts(classification_h5=empty), N_ROIS)
            )

    def test_extraction_columns_are_always_a_full_set(self) -> None:
        """Every declared column is present when extraction wrote them."""
        columns = nb._extraction_columns(self.artifacts(), N_ROIS)
        self.assertEqual(set(columns), {name for name, _ in nb.EXTRACTION_COLUMN_SPECS})
        self.assertEqual(columns["cellpose_soma_probability"].shape, (N_ROIS,))

    def test_absent_extraction_stats_become_nan_columns(self) -> None:
        """The CaImAn branch writes no stat datasets; the columns stay."""
        bare = write_extraction(self.root / "bare_extraction.h5", self.masks, with_roi_stats=False)
        with self.assertLogs(nb.logger, level=logging.WARNING):
            columns = nb._extraction_columns(self.artifacts(extraction_h5=bare), N_ROIS)
        for name, values in columns.items():
            with self.subTest(column=name):
                self.assertEqual(values.shape, (N_ROIS,))
                self.assertTrue(np.isnan(values).all())


class TestSeriesTiming(unittest.TestCase):
    """Timing kwargs and the grayscale reduction."""

    def test_timestamps_win_when_present(self) -> None:
        """A multiplane plane carries its own sync timestamps."""
        timestamps = np.arange(N_FRAMES, dtype=float) / FRAME_RATE
        kwargs = nb._series_kwargs(timestamps, FRAME_RATE)
        np.testing.assert_array_equal(kwargs["timestamps"], timestamps)
        self.assertNotIn("rate", kwargs)

    def test_frame_rate_fallback_without_a_sync_file(self) -> None:
        """Single-plane sessions have no sync file, so a rate is recorded."""
        kwargs = nb._series_kwargs(None, FRAME_RATE)
        self.assertEqual(kwargs, {"rate": FRAME_RATE, "starting_time": 0.0})

    def test_two_dimensional_image_passes_through(self) -> None:
        """Motion correction writes 2-D uint8, which needs no reduction."""
        image = np.zeros((HEIGHT, WIDTH), dtype="uint8")
        self.assertIs(nb._as_grayscale(image, Path("projection.png")), image)

    def test_colour_image_is_collapsed_with_a_warning(self) -> None:
        """A colour-mapped PNG would abort GrayscaleImage construction."""
        image = np.zeros((HEIGHT, WIDTH, 4), dtype="uint8")
        with self.assertLogs(nb.logger, level=logging.WARNING):
            reduced = nb._as_grayscale(image, Path("projection.png"))
        self.assertEqual(reduced.shape, (HEIGHT, WIDTH))


class TestCreateMicroscope(unittest.TestCase):
    """The device is written from the reported instrument."""

    def _device(self, metadata: NwbMetadata) -> Any:
        """Create the microscope on an empty file and return the device.

        Parameters
        ----------
        metadata : NwbMetadata
            The acquisition metadata.

        Returns
        -------
        pynwb.device.Device
            The created device.
        """
        nwbfile = NWBFile(
            session_description="test",
            identifier="test",
            session_start_time=datetime.now(timezone.utc),
        )
        device, _ = nb.create_microscope(nwbfile, metadata)
        return device

    def test_reported_instrument_is_written(self) -> None:
        """Name, manufacturer and description pass through unchanged."""
        metadata = replace(
            nwb_metadata(Path("/data/acquisition.json"), []),
            instrument=NwbInstrument(
                name="rig-a",
                manufacturer="Bruker",
                description="Resonant 2P",
            ),
        )
        device = self._device(metadata)
        self.assertEqual(
            (device.name, device.manufacturer, device.description),
            ("rig-a", "Bruker", "Resonant 2P"),
        )

    def test_nothing_reported_invents_no_manufacturer(self) -> None:
        """An unnamed instrument falls back to the id, then a placeholder."""
        metadata = replace(
            nwb_metadata(Path("/data/metadata.json"), []), instrument_id=None
        )
        device = self._device(metadata)
        self.assertEqual(device.name, "unknown-instrument")
        self.assertIsNone(device.manufacturer)


class TestAddEpochIntervals(unittest.TestCase):
    """Epochs are named by the reported labels."""

    def _names(self, labels: Optional[Dict[str, str]]) -> List[str]:
        """Write two epochs and return their NWB names.

        Parameters
        ----------
        labels : dict, optional
            Label per TIFF stem.

        Returns
        -------
        list of str
            The epoch table's stimulus names.
        """
        epoch_json = Path(tempfile.mkdtemp()) / "epoch_locations.json"
        epoch_json.write_text(
            json.dumps({"neuron3": [0, 10], "spont": [10, 20]})
        )
        nwbfile = NWBFile(
            session_description="test",
            identifier="test",
            session_start_time=datetime.now(timezone.utc),
        )
        nb.add_epoch_intervals(nwbfile, epoch_json, FRAME_RATE, labels)
        return list(nwbfile.get_time_intervals("epochs")["stimulus_name"][:])

    def test_labels_name_the_epochs(self) -> None:
        """A stem with a label is written under it."""
        self.assertEqual(self._names({"neuron3": "BCI"}), ["BCI", "spont"])

    def test_unlabelled_stems_keep_their_names(self) -> None:
        """Without labels every epoch keeps its stem."""
        self.assertEqual(self._names(None), ["neuron3", "spont"])


class TestSyncTimestampsPerPlane(unittest.TestCase):
    """The multiplane vsync edge train split into per-plane timestamps."""

    def test_coupled_pairs_are_decimated_and_offset(self) -> None:
        """Four planes are two temporal groups, so every second edge."""
        planes = [
            plane_metadata("VISp_0", 0, 0),
            plane_metadata("VISl_1", 1, 0),
            plane_metadata("VISal_2", 2, 1),
            plane_metadata("VISam_3", 3, 1),
        ]
        edges = np.arange(N_FRAMES, dtype=float) / FRAME_RATE
        per_plane = nb.sync_timestamps_per_plane(planes, edges)
        self.assertEqual(set(per_plane), {plane["plane_id"] for plane in planes})
        np.testing.assert_array_equal(per_plane["VISp_0"], edges[0::2])
        np.testing.assert_array_equal(per_plane["VISal_2"], edges[1::2])
        self.assertEqual(len(per_plane["VISl_1"]), N_FRAMES // 2)

    def test_odd_plane_count_raises(self) -> None:
        """Coupled-pair assignment needs an even number of planes."""
        with self.assertRaises(ValueError) as ctx:
            nb.sync_timestamps_per_plane([plane_metadata()], np.arange(10, dtype=float))
        self.assertIn("1 planes", str(ctx.exception))

    def test_coupled_index_disagreeing_with_position_raises(self) -> None:
        """Mis-timestamping every trace is worse than failing the run."""
        planes = [plane_metadata("VISp_0", 0, 0), plane_metadata("VISl_1", 1, 3)]
        with self.assertRaises(ValueError) as ctx:
            nb.sync_timestamps_per_plane(planes, np.arange(10, dtype=float))
        self.assertIn("pair 3", str(ctx.exception))

    def test_unset_coupled_index_is_accepted(self) -> None:
        """v1 metadata can omit the coupled index entirely."""
        planes = [
            plane_metadata("plane_0", 0, coupled=None),
            plane_metadata("plane_1", 1, coupled=None),
        ]
        per_plane = nb.sync_timestamps_per_plane(planes, np.arange(10, dtype=float))
        self.assertEqual(len(per_plane), 2)


class TestAddPlane(ExtractionFileCase):
    """One plane packaged end to end from real upstream payloads."""

    def setUp(self) -> None:
        """Add an empty NWBFile, its device and its optical channel."""
        super().setUp()
        self.plane = plane_metadata()
        self.metadata = nwb_metadata(self.root / "acquisition.json", [self.plane])
        self.nwbfile = NWBFile(
            session_description="test",
            identifier="test",
            session_start_time=datetime.now(timezone.utc),
        )
        self.device, self.optical_channel = nb.create_microscope(self.nwbfile, self.metadata)

    def add(
        self,
        artifacts: Optional[PlaneArtifacts] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, bool], int]:
        """Add this plane and return the builder's result.

        Parameters
        ----------
        artifacts : PlaneArtifacts, optional
            Artifacts to package; the complete set by default.
        timestamps : numpy.ndarray, optional
            Per-plane sync timestamps; omitted for a single-plane session.

        Returns
        -------
        tuple of (dict, int)
            Which containers were written, and the ROI count.
        """
        return nb.add_plane(
            self.nwbfile,
            self.metadata,
            self.plane,
            artifacts if artifacts is not None else self.artifacts(),
            self.device,
            self.optical_channel,
            timestamps=timestamps,
        )

    def test_complete_plane_writes_every_container(self) -> None:
        """A full upstream set yields all traces, images and columns."""
        contents, num_rois = self.add()
        self.assertEqual(num_rois, N_ROIS)
        self.assertTrue(all(contents[name] for name in contents))
        module = self.nwbfile.processing[PLANE_ID]
        self.assertEqual(
            set(module.data_interfaces),
            {
                "raw",
                "neuropil_fluorescence",
                "neuropil_corrected",
                "dff",
                "event",
                "image_segmentation",
                "images",
            },
        )

    def test_roi_table_description_names_the_segmentation(self) -> None:
        """The description separates the method from its description."""
        self.add()
        segmentation = self.nwbfile.processing[PLANE_ID]["image_segmentation"]
        approach = nb.SegmentationApproach.SUITE2P_ACTIVITY.value
        self.assertEqual(
            segmentation.plane_segmentations["roi_table"].description,
            f"{approach['method']}: {approach['description']}",
        )

    def test_traces_are_written_time_major(self) -> None:
        """RoiResponseSeries data is (n_frames, n_rois), the transpose."""
        self.add()
        series = self.nwbfile.processing[PLANE_ID]["neuropil_fluorescence"]
        self.assertEqual(series.data.shape, (N_FRAMES, N_ROIS))

    def test_single_plane_records_a_rate(self) -> None:
        """Without sync timestamps the acquisition frame rate is recorded."""
        self.add()
        series = self.nwbfile.processing[PLANE_ID]["neuropil_fluorescence"]
        self.assertEqual(series.rate, FRAME_RATE)

    def test_multiplane_names_are_suffixed_and_timestamped(self) -> None:
        """Sync timestamps select the v1 ``_timeseries`` container names."""
        timestamps = np.arange(N_FRAMES, dtype=float) / FRAME_RATE
        self.add(timestamps=timestamps)
        module = self.nwbfile.processing[PLANE_ID]
        series = module["neuropil_fluorescence_timeseries"]
        np.testing.assert_array_equal(series.timestamps, timestamps)

    def test_longer_timestamp_vector_is_trimmed_to_the_traces(self) -> None:
        """A debug-truncated movie leaves the sync record longer.

        The trace length is authoritative and every upstream cap is a head
        slice, so the head of the timestamp vector is the matching window. The
        comparison is a length check on two arrays, never a truthiness test.
        """
        timestamps = np.arange(N_FRAMES + 500, dtype=float) / FRAME_RATE
        with self.assertLogs(nb.logger, level=logging.WARNING) as logs:
            self.add(timestamps=timestamps)
        series = self.nwbfile.processing[PLANE_ID]["neuropil_fluorescence_timeseries"]
        self.assertEqual(len(series.timestamps), N_FRAMES)
        np.testing.assert_array_equal(series.timestamps, timestamps[:N_FRAMES])
        self.assertIn("trimming timestamps", logs.output[0])

    def test_trace_roi_count_mismatch_raises(self) -> None:
        """Traces and masks pair positionally, so a mismatch must not pass."""
        wrong = write_single_dataset(
            self.root / "wrong_dff.h5",
            "data",
            np.zeros((N_ROIS + 2, N_FRAMES), dtype="float32"),
        )
        with self.assertRaises(ValueError) as ctx:
            self.add(self.artifacts(dff_h5=wrong))
        self.assertIn("ROI masks", str(ctx.exception))

    def test_missing_traces_group_raises(self) -> None:
        """Without traces/roi the plane has no timeseries to package."""
        bare = self.root / "no_traces_extraction.h5"
        coords = np.array(np.nonzero(self.masks))
        with h5py.File(bare, "w") as handle:
            rois = handle.create_group("rois")
            rois["data"] = np.ones(coords.shape[1], dtype=bool)
            rois["coords"] = coords
            rois["shape"] = np.array(self.masks.shape)
        with self.assertRaises(KeyError):
            self.add(self.artifacts(extraction_h5=bare))

    def test_optional_artifacts_absent_still_packages(self) -> None:
        """A skipped dF/F, OASIS or classifier degrades one container each."""
        with self.assertLogs(nb.logger, level=logging.WARNING):
            contents, num_rois = self.add(
                self.artifacts(dff_h5=None, events_oasis_h5=None, classification_h5=None)
            )
        self.assertEqual(num_rois, N_ROIS)
        self.assertTrue(contents["roi_fluorescence"])
        self.assertFalse(contents["dff"])
        self.assertFalse(contents["events"])
        self.assertFalse(contents["classification"])

    def test_absent_projections_are_reported_from_the_paths(self) -> None:
        """``projections`` must not be derived from the image list, which
        also holds the segmentation and neuropil mask images."""
        with self.assertLogs(nb.logger, level=logging.WARNING):
            contents, _ = self.add(
                self.artifacts(average_projection_png=None, maximum_projection_png=None)
            )
        self.assertFalse(contents["projections"])
        self.assertIn("images", self.nwbfile.processing[PLANE_ID].data_interfaces)

    def test_anatomical_mode_uses_the_cellpose_mask_image(self) -> None:
        """The segmentation image source follows the segmentation approach."""
        anatomical = write_extraction(
            self.root / "cellpose_extraction.h5", self.masks, cellpose=True
        )
        self.add(self.artifacts(extraction_h5=anatomical))
        images = self.nwbfile.processing[PLANE_ID]["images"]
        self.assertIn("segmentation_mask_image", images.images)

    def test_geometry_is_the_pixel_size_in_micrometres(self) -> None:
        """grid_spacing and image resolution come from um_per_pixel.

        A v2 scale is in the acquisition's coordinate-system unit (millimetres
        for the mesoscope), so the raw scale factor must not be written.
        """
        self.plane = {
            **plane_metadata(),
            "scale_factor": SCALE_FACTOR / 1000,
            "scale_factor_unit": "millimeter",
        }
        self.add()
        imaging_plane = self.nwbfile.imaging_planes[PLANE_ID]
        self.assertEqual(list(imaging_plane.grid_spacing), [SCALE_FACTOR, SCALE_FACTOR])
        self.assertEqual(imaging_plane.grid_spacing_unit, "micrometer")
        images = self.nwbfile.processing[PLANE_ID]["images"]
        self.assertEqual(images["average_projection"].resolution, SCALE_FACTOR)

    def test_unknown_pixel_size_omits_the_geometry(self) -> None:
        """No pixel size is written as absent, never as 1.0."""
        self.plane = {**plane_metadata(), "um_per_pixel": None}
        with self.assertLogs(nb.logger, level=logging.WARNING):
            self.add()
        self.assertIsNone(self.nwbfile.imaging_planes[PLANE_ID].grid_spacing)
        images = self.nwbfile.processing[PLANE_ID]["images"]
        self.assertIsNone(images["average_projection"].resolution)

    def test_zero_roi_plane_packages_without_argmax_failing(self) -> None:
        """A plane can legitimately have no ROIs at all, and still writes."""
        empty_masks = roi_masks(0)
        extraction = write_extraction(self.root / "zero_extraction.h5", empty_masks)
        contents, num_rois = self.add(
            self.artifacts(
                extraction_h5=extraction,
                dff_h5=write_single_dataset(
                    self.root / "zero_dff.h5", "data", np.zeros((0, N_FRAMES), dtype="float32")
                ),
                events_oasis_h5=None,
                classification_h5=write_classification(
                    self.root / "zero_classification.h5", n_rois=0
                ),
            )
        )
        self.assertEqual(num_rois, 0)
        self.assertTrue(contents["dff"])
        path = nb.write_nwb_zarr(self.nwbfile, self.root / "out" / "zero.nwb.zarr")
        self.assertEqual(nb.read_nwb_zarr(path), (PLANE_ID,))

    def test_neuropil_masks_absent_omits_the_table(self) -> None:
        """An extraction file without neuropil masks still packages."""
        extraction = write_extraction(
            self.root / "no_neuropil_extraction.h5", self.masks, with_neuropil_masks=False
        )
        with self.assertLogs(nb.logger, level=logging.WARNING):
            self.add(self.artifacts(extraction_h5=extraction))
        segmentation = self.nwbfile.processing[PLANE_ID]["image_segmentation"]
        self.assertNotIn("neuropil_table", segmentation.plane_segmentations)


class TestZarrRoundTrip(ExtractionFileCase):
    """The written store must re-open and report its planes."""

    def test_write_and_read_back(self) -> None:
        """The written store re-opens with its plane module."""
        plane = plane_metadata()
        metadata = nwb_metadata(self.root / "acquisition.json", [plane])
        nwbfile = NWBFile(
            session_description="test",
            identifier="test",
            session_start_time=datetime.now(timezone.utc),
        )
        device, optical_channel = nb.create_microscope(nwbfile, metadata)
        nb.add_plane(nwbfile, metadata, plane, self.artifacts(), device, optical_channel)
        path = nb.write_nwb_zarr(nwbfile, self.root / "out" / "pophys.nwb.zarr")
        self.assertTrue(path.exists())
        self.assertEqual(nb.read_nwb_zarr(path), (PLANE_ID,))


def minimal_nwb_metadata(**session: Any) -> NwbMetadata:
    """Return acquisition metadata read from a minimal document, with the given session fields."""
    return NwbMetadata(
        schema_version="minimal",
        core_file=Path("/data/metadata.json"),
        planes=(),
        frame_rate=FRAME_RATE,
        session=NwbSession(**session),
    )


class TestCreateMinimalBaseNwb(unittest.TestCase):
    """The base NWB file built from a minimal metadata.json."""

    def test_full_session(self) -> None:
        """Every supplied field reaches the file and its subject."""
        nwbfile = nb.create_minimal_base_nwb(
            minimal_nwb_metadata(
                start_time="2026-10-30T09:00:00-04:00",
                dataset_name="single-plane-ophys_0322R_2026-10-30_09-00-00",
                institution="Kempner Institute",
                subject_id="0322R",
                sex="Male",
                species="Mus musculus",
                date_of_birth="2026-01-02",
                genotype="wt/wt",
            )
        )
        self.assertEqual(nwbfile.session_id, "single-plane-ophys_0322R_2026-10-30_09-00-00")
        self.assertEqual(nwbfile.institution, "Kempner Institute")
        self.assertEqual(nwbfile.session_start_time.utcoffset().total_seconds(), -4 * 3600)
        self.assertIn("0322R", nwbfile.session_description)
        self.assertEqual(nwbfile.subject.subject_id, "0322R")
        self.assertEqual(nwbfile.subject.sex, "M")
        self.assertEqual(nwbfile.subject.species, "Mus musculus")
        self.assertEqual(nwbfile.subject.genotype, "wt/wt")
        self.assertEqual(nwbfile.subject.date_of_birth.date().isoformat(), "2026-01-02")

    def test_bare_session(self) -> None:
        """Only a start time: no subject, a generic description, UTC assumed for a Z suffix."""
        metadata = minimal_nwb_metadata(start_time="2026-10-30T09:00:00Z")
        nwbfile = nb.create_minimal_base_nwb(metadata)
        self.assertIsNone(nwbfile.subject)
        self.assertIsNone(nwbfile.session_id)
        self.assertIn("unnamed session", nwbfile.session_description)
        self.assertEqual(nwbfile.session_start_time.tzinfo, timezone.utc)

    def test_naive_start_time_is_assumed_utc(self) -> None:
        """A start time with no offset is read as UTC, with a warning."""
        with self.assertLogs(nb.logger, level=logging.WARNING):
            metadata = minimal_nwb_metadata(start_time="2026-10-30T09:00:00")
            nwbfile = nb.create_minimal_base_nwb(metadata)
        self.assertEqual(nwbfile.session_start_time.tzinfo, timezone.utc)

    def test_unparseable_start_time_names_the_field(self) -> None:
        """A malformed start time fails with the field name."""
        with self.assertRaisesRegex(ValueError, "session_start_time"):
            nb.create_minimal_base_nwb(minimal_nwb_metadata(start_time="yesterday"))

    def test_unknown_sex_maps_to_u(self) -> None:
        """Unrecognised or missing sex values become pynwb's U code."""
        self.assertEqual(nb._nwb_sex(None), "U")
        self.assertEqual(nb._nwb_sex("unknown"), "U")
        self.assertEqual(nb._nwb_sex("female"), "F")


if __name__ == "__main__":
    unittest.main()
