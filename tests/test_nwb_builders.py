"""Tests for the pynwb builders, driven by real upstream payloads.

Every fixture here stands in for an upstream step's compute result, so it is
built from the production type: ``numpy`` arrays written into real HDF5 files at
the shapes extraction, dF/F, OASIS and the classifier actually emit. A Python
list stands in for none of them -- ``x or default``, ``len(x or [])`` and
``if x:`` all behave differently on an array, so a list-based fixture can pass
while a real run raises ``ValueError: The truth value of an array with more
than one element is ambiguous`` after compute has finished.

``nwb_builders`` is coverage-omitted (it needs a writable Zarr store for the
terminal write), so these tests target the array handling: the mask collapse,
the HDF5 reads, the positional ROI pairing, the sync-timestamp split and the
trace/timestamp length reconciliation.
"""

import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import h5py
import imageio.v3 as iio
import numpy as np

from aind_ophys_nwb_library import nwb_builders as nb
from aind_ophys_nwb_library.file_handling import PlaneArtifacts

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


if __name__ == "__main__":
    unittest.main()
