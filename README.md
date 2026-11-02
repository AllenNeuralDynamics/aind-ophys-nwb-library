# aind-ophys-nwb-library

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10,<3.13-blue?logo=python)

NWB packaging of AIND planar / population ophys pipeline outputs.

This is the backing library for the `aind-ophys-nwb` Code Ocean capsule: all the
logic lives here with its own tests, and the capsule is a thin wrapper that calls
`aind_ophys_nwb_library.job.run`. Metadata reads and the `Processing` builders
come from the shared
[`aind-pophys-metadata`](https://github.com/AllenNeuralDynamics/aind-pophys-metadata)
package, so they stay consistent across the pipeline's steps.

This is the pipeline's **terminal fan-in**. `main.nf` calls the NWB step once with
every upstream channel `.collect()`ed, so one invocation sees every plane's
outputs at the same time and produces exactly **one** NWB file and one
`processing.json` for the whole session — not one per plane.

## What it does

1. Resolve the `raw` and `processed` mounts under `input_dir` (falling back to
   the mount root when the subdirectory is absent, so a standalone run with
   flat mounts still works).
2. Read the acquisition metadata — aind-data-schema v1 `session.json` or v2
   `acquisition.json`, dispatched on which file is present — into the frozen
   `NwbMetadata` view: canonical plane ids, frame rate, instrument id,
   excitation/emission wavelengths and indicator.
3. Classify the session with `session_type()`: `MULTIPLANE`,
   `SINGLE_PLANE_WITH_EPOCHS`, or `SINGLE_PLANE`. The fork is decided by
   **plane count**, not by `instrument_id` and not by the v1-only
   `data_description.platform`.
4. Locate every upstream artifact for every plane with
   `collect_session_artifacts()`.
5. Build the base NWB file (via `aind-nwb-utils`), then add each plane's
   segmentation, summary images and traces with `add_plane()`.
6. For a multiplane session, split the sync file's 2p vsync rising edges into
   per-plane timestamps with `sync_timestamps_per_plane()`. A single-plane
   session has no sync file and records the acquisition frame rate instead.
7. Write the NWB Zarr store, re-open it to confirm every plane is present, and
   emit `processing.json`.

### Cross-step contracts

Two pairings are **positional**, by suite2p ROI order and by plane order, and
both are checked rather than trusted:

- Trace arrays, ROI masks and the classifier's columns are paired by ROI index.
  A count mismatch raises (traces) or drops the affected columns with an error
  (classification) rather than silently misaligning them.
- The rich per-plane records are paired to the shared layer's canonical plane
  ids by position; a length mismatch raises, because a wrong plane id names the
  wrong output folder.

The classifier writes `predict_proba` output, so `soma/probabilities` and
`dendrites/probabilities` are 2-D `(n_rois, n_classes)` and the positive class
is read off the last column.

## Usage

```python
from aind_ophys_nwb_library.job import run

run()  # settings parsed from CLI arguments / environment
```

Settings can also be constructed explicitly:

```python
from aind_ophys_nwb_library.job import run
from aind_ophys_nwb_library.settings import NwbSettings

run(NwbSettings(skip_lab_metadata=True))
```

### Settings

`NwbSettings` (see `src/aind_ophys_nwb_library/settings.py`) is a
`pydantic-settings` model with no environment-variable prefix. Every field is
addressable as `--name=value`, the only form a Code Ocean app panel emits, and
no field name is one character long (pydantic-settings would render it as a
short option the panel can never produce). The Code Ocean capsule's app panel is generated
from this file with `auto-app-panel`.

| field | default | meaning |
|---|---|---|
| `input_dir` | `/data` | mount root; expected to hold `raw/` and `processed/` |
| `output_dir` | `/results` | results root |
| `raw_subdir` | `raw` | subdirectory holding the core metadata, sync file and movies |
| `processed_subdir` | `processed` | subdirectory holding every upstream step's results |
| `nwb_name` | `pophys.nwb.zarr` | basename of the Zarr store, written inside `output_dir/nwb/` |
| `extension_namespace` | `ndx-aibs-behavior-ophys.namespace.yaml` | namespace YAML defining the per-plane `OphysMetadataSchema` lab metadata |
| `skip_lab_metadata` | `False` | skip that lab metadata; use only when the YAML is unavailable |

`recorded_parameters()` reduces `extension_namespace` to its basename before it
reaches `Code.parameters`: in the pipeline the step is handed the YAML inside
its Nextflow task scratch directory, a path destroyed with the task, so
recording it verbatim would put a dangling absolute path into permanent
metadata.

## Modules

### `job.py`

Orchestration. `run()` is the entry point the runtime wrapper calls: it resolves the
mounts, loads the metadata, configures logging, and runs the packaging inside
the shared `stage_guard` so a failure still leaves a `processing.json` naming
what was attempted and why it stopped.

### `nwb_builders.py`

The pynwb object construction. `add_plane()` adds one plane's imaging plane,
segmentation, summary images and traces, and is shared by the multiplane and
single-plane paths — parameterised by the timing source and the container-name
suffix, so a fix cannot land on one path and miss the other. Also
`create_microscope()`, `instrument_description()`, `segmentation_approach()`,
`collapse_masks_to_label_image()`, `sync_timestamps_per_plane()`,
`add_epoch_intervals()`, `add_plane_lab_metadata()`, `load_pynwb_extension()`,
the HDF5 reads (`load_dataset()`, `try_load_dataset()`, `load_roi_masks()`,
`load_neuropil_masks_and_r_values()`) and the Zarr IO (`write_nwb_zarr()`,
`read_nwb_zarr()`).

### `file_handling.py`

Discovery of the upstream step outputs this library repackages.
`collect_session_artifacts()` resolves one frozen `PlaneArtifacts` per plane
from `find_artifact()` / `find_artifacts()`, which glob recursively, following
symlinked directories, on the `<plane_id>_` filename prefix rather than walking
a fixed directory layout. Only
`extraction_h5` is required; every other artifact degrades to an omitted NWB
container. Also `find_epoch_locations()` and `find_sync_file()`. Kept
import-light — pure path logic, so the upstream contract stays testable without
pynwb or real HDF5 payloads.

### `settings.py`

`NwbSettings`. Kept import-light (pydantic plus stdlib) so it is cheap to import,
free of pynwb / hdmf-zarr / aind-data-schema side effects, and readable by
`auto-app-panel` without the heavy dependencies installed.

### `utils/metadata_utils.py`

The NWB-specific read layer over `aind_pophys_metadata.CoreMetadata`, which owns
file discovery, schema-version dispatch, the canonical plane ids and every
per-plane imaging field. `load_nwb_metadata()` assembles the frozen `NwbMetadata`
view from one `CoreMetadata.load()` call, so no schema version is compared here;
`identified_planes()` sorts the shared plane-record dicts by `plane_index` and
pairs each with its canonical `plane_id`.
`plane_location()` renders a plane's `ImagingPlane.location` string and
`get_injection_material()` reads the indicator out of `procedures.json`, which
the shared layer does not normalize. `build_nwb_code()` builds this step's
`Code` block and `attach_input_data()` records the resolved inputs on it once
they are known. Reads are raw dicts — nothing here calls `model_validate` on an
input file, so messy-but-sufficient metadata still packages.

## Outputs

Everything is written under `<output_dir>/nwb/`, never at the results root:
`main.nf` publishes each step's files at their relative paths, so a root-level
`processing.json` would collide with the run-level one the aggregator writes.

- `pophys.nwb.zarr` (the `nwb_name` setting) — the packaged NWB Zarr store
- `processing.json` — full v2 `Processing`, written by the shared `stage_guard`

Inside the NWB file, each plane gets a processing module named after its
canonical plane id, holding:

- `image_segmentation` — `roi_table` (ROI masks plus the classifier's
  soma/dendrite columns and extraction's own cellpose probability and suite2p
  morphology stats) and `neuropil_table` (neuropil masks and r-values)
- `images` — `average_projection`, `max_projection`,
  `segmentation_mask_image` and `neuropil_mask_image`
- the trace containers — `raw` (ROI fluorescence), `neuropil_fluorescence`,
  `neuropil_corrected`, `dff` and `event`. Multiplane containers carry a
  `_timeseries` suffix and per-plane sync timestamps; single-plane containers
  are unsuffixed and record the frame rate.

A multiplane session also carries the per-plane `ndx-aibs-behavior-ophys` lab
metadata; a single-plane session with `epoch_locations.json` carries an `epochs`
time-intervals table instead.

`processing.json` is emitted through the shared
[`aind-pophys-metadata`](https://github.com/AllenNeuralDynamics/aind-pophys-metadata)
package, so its shape matches every other step in the pipeline.

## Level of Support
Please indicate a level of support:
 - [ ] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using [uv](https://docs.astral.sh/uv/), run
```bash
uv sync
```

## Tests and linting

Tests are `unittest`. The CI gate is:

```bash
flake8 .
interrogate --verbose .
coverage run -m unittest discover
coverage report
```

`flake8` allows 100-character lines; `coverage` and `interrogate` are both gated
at 100%. `nwb_builders.py` and `job.py` are omitted from the coverage
measurement because exercising them fully needs real upstream HDF5 payloads and
a writable Zarr store — a Code Ocean data run. `tests/test_nwb_builders.py` still
covers their array handling against real HDF5 fixtures written at production
shapes and dtypes.
