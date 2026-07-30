"""Tests for quality_control.json generation.

The v1 capsule emitted no QC at all, so everything here is net-new against the
v2 output contract: one full ``QualityControl`` per run, every metric tagged
with an ``evaluation`` key matching ``default_grouping``, and every metric
carrying a non-empty ``status_history`` (an empty one raises ``IndexError`` when
``QualityControl`` derives its status).
"""

import json
import tempfile
import unittest
from pathlib import Path

from aind_data_schema.core.quality_control import QualityControl, Status

from aind_ophys_nwb_library import qc

PLANE_IDS = ("VISp_0", "VISl_1")
NWB_REFERENCE = "nwb/pophys.nwb.zarr"

COMPLETE = {name: True for name in qc.PLANE_CONTAINERS}


class TestPlaneContentsMetric(unittest.TestCase):
    """Per-plane record of which NWB containers were written."""

    def test_complete_plane_passes(self):
        """Every container present and ROIs written is a PASS."""
        metric = qc.build_plane_contents_metric("VISp_0", COMPLETE, 42)
        self.assertEqual(metric.status_history[-1].status, Status.PASS)
        self.assertEqual(metric.value["num_rois"], 42)

    def test_missing_containers_are_named_in_the_description(self):
        """A reviewer sees which containers are absent, not just a fail."""
        contents = dict(COMPLETE, dff=False, events=False)
        metric = qc.build_plane_contents_metric("VISp_0", contents, 42)
        self.assertIn("dff", metric.description)
        self.assertIn("events", metric.description)
        self.assertFalse(metric.value["dff"])

    def test_no_traces_fails(self):
        """A plane with no ROI fluorescence is not usable."""
        contents = dict(COMPLETE, roi_fluorescence=False)
        metric = qc.build_plane_contents_metric("VISp_0", contents, 42)
        self.assertEqual(metric.status_history[-1].status, Status.FAIL)

    def test_zero_rois_fails(self):
        """Traces over zero ROIs is a failure, not a pass."""
        metric = qc.build_plane_contents_metric("VISp_0", COMPLETE, 0)
        self.assertEqual(metric.status_history[-1].status, Status.FAIL)

    def test_unknown_container_keys_are_ignored(self):
        """Only the declared container set lands in the value."""
        metric = qc.build_plane_contents_metric("VISp_0", {"surprise": True}, 1)
        self.assertNotIn("surprise", metric.value)
        self.assertEqual(
            set(metric.value) - {"num_rois"},
            set(qc.PLANE_CONTAINERS),
        )

    def test_name_is_scoped_to_the_plane(self):
        """Metric names must be unique within the one session document."""
        metric = qc.build_plane_contents_metric("VISl_1", COMPLETE, 3)
        self.assertTrue(metric.name.startswith("VISl_1"))

    def test_each_required_container_fails_on_its_own(self):
        """Any missing required container fails, not just roi_fluorescence.

        Without this, a plane missing dF/F, events AND classification still
        scored PASS and the overall QualityControl status was identical to a
        healthy run — a silently gutted NWB was indistinguishable from a
        complete one.
        """
        for container in qc.REQUIRED_PLANE_CONTAINERS:
            with self.subTest(missing=container):
                contents = dict(COMPLETE)
                contents[container] = False
                metric = qc.build_plane_contents_metric("VISp_0", contents, 3)
                self.assertEqual(metric.status_history[-1].status, Status.FAIL)
                self.assertIn(container, metric.description)

    def test_optional_container_absence_still_passes(self):
        """A container a full run may legitimately omit does not fail."""
        optional = set(qc.PLANE_CONTAINERS) - set(qc.REQUIRED_PLANE_CONTAINERS)
        self.assertTrue(optional, "expected at least one non-required container")
        for container in optional:
            with self.subTest(missing=container):
                contents = dict(COMPLETE)
                contents[container] = False
                metric = qc.build_plane_contents_metric("VISp_0", contents, 3)
                self.assertEqual(metric.status_history[-1].status, Status.PASS)

    def test_required_containers_are_a_subset_of_all(self):
        """The required set cannot drift out of PLANE_CONTAINERS."""
        self.assertTrue(
            set(qc.REQUIRED_PLANE_CONTAINERS) <= set(qc.PLANE_CONTAINERS),
        )


class TestNwbWrittenMetric(unittest.TestCase):
    """Session-level record of the packaged file."""

    def test_reopened_passes(self):
        """A re-openable store carrying every plane is a PASS."""
        metric = qc.build_nwb_written_metric(PLANE_IDS, NWB_REFERENCE, True)
        self.assertEqual(metric.status_history[-1].status, Status.PASS)
        self.assertEqual(metric.value["num_planes"], 2)
        self.assertEqual(metric.value["planes"], list(PLANE_IDS))
        self.assertEqual(metric.reference, NWB_REFERENCE)

    def test_not_reopened_fails(self):
        """A store that could not be re-opened is a FAIL, not PENDING.

        The round-trip is always attempted, so ``readable=False`` means the
        store did not re-open or a plane is missing from it. PENDING would
        make that indistinguishable from "awaiting human review", which is
        what every other pending metric in the document means.
        """
        metric = qc.build_nwb_written_metric(PLANE_IDS, NWB_REFERENCE, False)
        self.assertEqual(metric.status_history[-1].status, Status.FAIL)
        self.assertFalse(metric.value["reopened"])


class TestNwbReviewMetric(unittest.TestCase):
    """The reviewer dropdown."""

    def test_is_pending_with_both_options(self):
        """The dropdown is unset and offers a pass and a fail option."""
        metric = qc.build_nwb_review_metric(NWB_REFERENCE)
        self.assertEqual(metric.status_history[-1].status, Status.PENDING)
        self.assertEqual(metric.value.options, qc.REVIEW_OPTIONS)
        self.assertEqual(metric.value.value, "")


class TestMetricInvariants(unittest.TestCase):
    """Properties every metric in the document must satisfy."""

    def _all_metrics(self):
        """Return one of every metric this module produces."""
        return [
            qc.build_plane_contents_metric("VISp_0", COMPLETE, 5),
            qc.build_nwb_written_metric(PLANE_IDS, NWB_REFERENCE, True),
            qc.build_nwb_review_metric(NWB_REFERENCE),
        ]

    def test_every_metric_seeds_status_history(self):
        """An empty status_history raises IndexError in QualityControl."""
        for metric in self._all_metrics():
            with self.subTest(metric=metric.name):
                self.assertTrue(metric.status_history)

    def test_every_metric_carries_the_grouping_tags(self):
        """default_grouping is ["evaluation"], so the tag must be present."""
        for metric in self._all_metrics():
            with self.subTest(metric=metric.name):
                self.assertEqual(metric.tags["evaluation"], qc.EVALUATION)
                self.assertEqual(metric.tags["type"], qc.METRIC_TYPE)


class TestQualityControlDocument(unittest.TestCase):
    """The full QualityControl document and its writer."""

    def test_build_uses_project_default_grouping(self):
        """The shared standard grouping is applied."""
        document = qc.build_quality_control([qc.build_nwb_review_metric(NWB_REFERENCE)])
        self.assertIsInstance(document, QualityControl)
        self.assertEqual(document.default_grouping, ["evaluation"])

    def test_write_and_round_trip(self):
        """The written file validates back into a core v2 QualityControl."""
        metrics = [qc.build_plane_contents_metric(plane_id, COMPLETE, 5) for plane_id in PLANE_IDS]
        metrics.append(qc.build_nwb_written_metric(PLANE_IDS, NWB_REFERENCE, True))
        metrics.append(qc.build_nwb_review_metric(NWB_REFERENCE))
        document = qc.build_quality_control(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            written = qc.write_quality_control_json(document, Path(tmp))
            self.assertTrue(written.exists())
            payload = json.loads(written.read_text())
            self.assertEqual(len(payload["metrics"]), 4)
            reloaded = QualityControl.model_validate_json(written.read_text())
        self.assertEqual(reloaded.default_grouping, ["evaluation"])
        self.assertTrue(all(m.status_history for m in reloaded.metrics))


if __name__ == "__main__":
    unittest.main()
