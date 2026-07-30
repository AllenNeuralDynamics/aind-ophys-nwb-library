"""quality_control.json generation for NWB packaging.

Capsule-specific metric content over the shared ``aind_pophys_metadata`` QC
builders. Metric-producing functions *return* metrics; ``job.run`` accumulates
them and writes ONE ``quality_control.json`` for the whole session (the
collector pattern).

The v1 capsule emitted **no** QC at all -- this is net-new. It is session-level
rather than per-plane because ``main.nf`` calls the NWB capsule once with every
upstream channel ``.collect()``ed, so one run produces exactly one NWB file
covering every plane; a per-plane document would have no file of its own to
describe.

Every metric seeds a ``status_history`` entry: an empty one raises
``IndexError`` when ``QualityControl`` derives its status.
"""

from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytz
from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_pophys_metadata import quality_control as shared_qc

_SEATTLE_TZ = pytz.timezone("America/Los_Angeles")

EVALUATION = "NWB Packaging"
METRIC_TYPE = "Operational QC"

REVIEW_OPTIONS = ["Complete", "Incomplete"]
REVIEW_STATUS = [Status.PASS, Status.FAIL]

#: Containers a fully-populated plane contributes to the NWB file. A plane
#: missing any of these still packages, but the metric records the gap.
PLANE_CONTAINERS = (
    "roi_fluorescence",
    "neuropil_fluorescence",
    "dff",
    "events",
    "classification",
    "projections",
)

#: Containers that a full pipeline run always produces, so their absence is a
#: real gap rather than a configuration choice. Gating the metric status on
#: these is what makes a silently gutted NWB distinguishable from a complete
#: one: without it, a plane missing dF/F, events AND classification still
#: scored PASS, and the overall QualityControl status was byte-identical to a
#: healthy run.
REQUIRED_PLANE_CONTAINERS = (
    "roi_fluorescence",
    "dff",
    "events",
    "classification",
)


def _automated_status(status: Status) -> QCStatus:
    """Return an automated QCStatus stamped in US/Pacific.

    Parameters
    ----------
    status : Status
        The derived pass/fail status.

    Returns
    -------
    QCStatus
        The status entry, evaluated by ``"Automated"``.
    """
    return QCStatus(
        evaluator="Automated",
        status=status,
        timestamp=dt.now(_SEATTLE_TZ).isoformat(),
    )


def _metric(
    name: str,
    value,
    description: str,
    status: Status,
    reference: Optional[str] = None,
) -> QCMetric:
    """Build an automatically-derived QCMetric with the standard tags.

    Parameters
    ----------
    name : str
        Metric name, already scoped to the plane or the session.
    value : Any
        The measured value (a scalar or a small dict).
    description : str
        Reviewer-facing description.
    status : Status
        Automated pass/fail status.
    reference : str or None, optional
        Path to a reference artifact, relative to the asset root.

    Returns
    -------
    QCMetric
        The populated metric.
    """
    return QCMetric(
        name=name,
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": EVALUATION, "type": METRIC_TYPE},
        description=description,
        reference=reference,
        status_history=[_automated_status(status)],
        value=value,
    )


def build_plane_contents_metric(
    plane_id: str,
    contents: Dict[str, bool],
    num_rois: int,
) -> QCMetric:
    """Record which NWB containers one plane contributed.

    Fails when the plane contributed no ROIs, or when any of
    :data:`REQUIRED_PLANE_CONTAINERS` is absent. A full pipeline run produces
    all of those, so a gap means an upstream artifact was missing or renamed —
    and the NWB is then silently incomplete for this plane rather than wrong,
    which is the harder failure to notice downstream.

    Parameters
    ----------
    plane_id : str
        Canonical plane id.
    contents : dict of str to bool
        Which of :data:`PLANE_CONTAINERS` were added for this plane.
    num_rois : int
        Number of ROIs written for this plane.

    Returns
    -------
    QCMetric
        The populated metric.
    """
    value = {name: bool(contents.get(name, False)) for name in PLANE_CONTAINERS}
    value["num_rois"] = int(num_rois)
    complete = num_rois > 0 and all(value[name] for name in REQUIRED_PLANE_CONTAINERS)
    status = Status.PASS if complete else Status.FAIL
    missing = sorted(name for name in PLANE_CONTAINERS if not value[name])
    description = (
        f"NWB containers written for {plane_id}"
        if not missing
        else f"NWB containers written for {plane_id}; missing: {', '.join(missing)}"
    )
    return _metric(
        name=f"{plane_id} NWB Contents",
        value=value,
        description=description,
        status=status,
    )


def build_nwb_written_metric(
    plane_ids: Sequence[str],
    nwb_reference: str,
    readable: bool,
) -> QCMetric:
    """Record that the NWB Zarr store was written and can be re-opened.

    Parameters
    ----------
    plane_ids : sequence of str
        Canonical plane ids packaged into the file.
    nwb_reference : str
        Path to the NWB store, relative to the results root.
    readable : bool
        Whether the written store was successfully re-opened. The round-trip
        is always attempted, so ``False`` means the store did not re-open or a
        plane is missing from it — both hard failures, hence ``FAIL`` rather
        than ``PENDING``, which in every other metric here means "awaiting
        human review".

    Returns
    -------
    QCMetric
        The populated metric.
    """
    return _metric(
        name="NWB File Written",
        value={
            "path": nwb_reference,
            "num_planes": len(plane_ids),
            "planes": list(plane_ids),
            "reopened": bool(readable),
        },
        description=("The packaged NWB Zarr store, and whether it was re-opened after " "writing"),
        status=Status.PASS if readable else Status.FAIL,
        reference=nwb_reference,
    )


def build_nwb_review_metric(nwb_reference: str) -> QCMetric:
    """Reviewer dropdown over the packaged NWB file.

    Parameters
    ----------
    nwb_reference : str
        Path to the NWB store, relative to the results root.

    Returns
    -------
    QCMetric
        A pending dropdown metric.
    """
    return shared_qc.dropdown_metric(
        name="NWB Contents Review",
        evaluation=EVALUATION,
        options=REVIEW_OPTIONS,
        status=REVIEW_STATUS,
        reference=nwb_reference,
        description=(
            "Does the packaged NWB contain every expected plane, trace type " "and summary image?"
        ),
        metric_type=METRIC_TYPE,
    )


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap metrics into a full QualityControl with the standard grouping."""
    return shared_qc.build_quality_control(metrics)


def write_quality_control_json(
    quality_control: QualityControl,
    output_dir: Path,
) -> Path:
    """Serialize a QualityControl object to ``quality_control.json``.

    Parameters
    ----------
    quality_control : QualityControl
        The QC document to write.
    output_dir : Path
        Directory into which ``quality_control.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    return shared_qc.write_quality_control_json(quality_control, output_dir)
