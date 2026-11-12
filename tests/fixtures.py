"""Minimal v1 and v2 metadata fixtures shared by the tests."""

from typing import Optional

from aind_pophys_metadata.io import object_type_value

FRAME_RATE = 9.48
SUBJECT_ID = "783992"
DATASET_NAME = "multiplane-ophys_783992_2026-07-30_12-00-00"
INSTRUMENT_ID = "MESO.1"
GENOTYPE = "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt"
INJECTION_MATERIAL = "AAV-GCaMP6f"
EXCITATION_NM = 920
EMISSION_NM = 520
SCALE_UM_PER_PX = 0.78
SCALE_MM_PER_PX = SCALE_UM_PER_PX / 1000
FOV_WIDTH = 512
FOV_HEIGHT = 512
SESSION_START_TIME = "2026-07-30T12:00:00-07:00"
STRUCTURES = ("VISp", "VISl")


def _v1_fov(index: int, structure: str, coupled_index: int) -> dict:
    """Return one minimal v1 ``ophys_fovs`` entry.

    Parameters
    ----------
    index : int
        FOV index.
    structure : str
        Targeted-structure acronym.
    coupled_index : int
        Temporal group index (``coupled_fov_index``).

    Returns
    -------
    dict
        The raw FOV dict.
    """
    return {
        "index": index,
        "targeted_structure": {"acronym": structure},
        "imaging_depth": 150 + 50 * index,
        "imaging_depth_unit": "micrometer",
        "frame_rate": FRAME_RATE,
        "fov_scale_factor": SCALE_UM_PER_PX,
        "fov_coordinate_unit": "micrometer",
        "coupled_fov_index": coupled_index,
        "fov_width": FOV_WIDTH,
        "fov_height": FOV_HEIGHT,
    }


def v1_session(plane_count: int = 2, subject_id: Optional[str] = SUBJECT_ID) -> dict:
    """Return a minimal v1 ``session.json`` dict.

    Parameters
    ----------
    plane_count : int, optional
        Number of ophys FOVs; 2 gives one coupled pair.
    subject_id : str, optional
        Subject identifier; pass ``None`` to omit the key.

    Returns
    -------
    dict
        The raw session dict.
    """
    fovs = [
        _v1_fov(index, STRUCTURES[index % len(STRUCTURES)], index // 2)
        for index in range(plane_count)
    ]
    session = {
        "rig_id": INSTRUMENT_ID,
        "data_streams": [
            {
                "light_sources": [{"name": "Laser", "wavelength": EXCITATION_NM}],
                "ophys_fovs": fovs,
            }
        ],
    }
    if subject_id is not None:
        session["subject_id"] = subject_id
    return session


def v1_session_without_frame_rate() -> dict:
    """Return a v1 session whose FOVs declare no frame rate."""
    session = v1_session()
    for fov in session["data_streams"][0]["ophys_fovs"]:
        del fov["frame_rate"]
    return session


def v1_session_without_planes() -> dict:
    """Return a v1 session carrying no ophys FOVs at all."""
    session = v1_session()
    session["data_streams"][0]["ophys_fovs"] = []
    return session


def _v2_plane(index: int, structure: str, coupled_index: Optional[int]) -> dict:
    """Return one minimal v2 ``CoupledPlane`` dict.

    Parameters
    ----------
    index : int
        Plane index.
    structure : str
        Targeted-structure acronym.
    coupled_index : int or None
        The partner plane's index (``coupled_plane_index``).

    Returns
    -------
    dict
        The raw plane dict.
    """
    from aind_data_schema.components.configs import CoupledPlane

    return {
        "object_type": object_type_value(CoupledPlane),
        "plane_index": index,
        "coupled_plane_index": coupled_index,
        "targeted_structure": {"acronym": structure},
        "depth": 150 + 50 * index,
        "depth_unit": "micrometer",
    }


def v2_acquisition(plane_count: int = 2, subject_id: Optional[str] = SUBJECT_ID) -> dict:
    """Return a minimal v2 ``acquisition.json`` dict, shaped as the mesoscope mapper writes it.

    Each plane names its partner, ``index ^ 1``, as its coupled plane. The scale
    is in the acquisition's millimetre coordinate system; ``dimensions_unit``
    is the unit of the pixel dimensions.

    Parameters
    ----------
    plane_count : int, optional
        Number of imaging planes; 2 gives one coupled pair.
    subject_id : str, optional
        Subject identifier; pass ``None`` to omit the key.

    Returns
    -------
    dict
        The raw acquisition dict.
    """
    from aind_data_schema.components.configs import (
        Channel,
        ImagingConfig,
        LaserConfig,
        PlanarImage,
    )
    from aind_data_schema.components.coordinates import Scale

    planes = [
        _v2_plane(
            index, STRUCTURES[index % len(STRUCTURES)], index ^ 1 if plane_count > 1 else None
        )
        for index in range(plane_count)
    ]
    acquisition = {
        "instrument_id": INSTRUMENT_ID,
        "coordinate_system": {"axis_unit": "millimeter"},
        "data_streams": [
            {
                "configurations": [
                    {
                        "object_type": object_type_value(ImagingConfig),
                        "sampling_strategy": {"frame_rate": FRAME_RATE},
                        "channels": [
                            {
                                "object_type": object_type_value(Channel),
                                "channel_name": "green",
                                "emission_wavelength": EMISSION_NM,
                                "light_sources": [
                                    {
                                        "object_type": object_type_value(LaserConfig),
                                        "device_name": "Laser",
                                        "wavelength": EXCITATION_NM,
                                    }
                                ],
                            }
                        ],
                        "images": [
                            {
                                "object_type": object_type_value(PlanarImage),
                                "channel_name": "green",
                                "dimensions_unit": "pixel",
                                "dimensions": {
                                    "object_type": object_type_value(Scale),
                                    "scale": [FOV_WIDTH, FOV_HEIGHT],
                                },
                                "image_to_acquisition_transform": [
                                    {
                                        "object_type": object_type_value(Scale),
                                        "scale": [SCALE_MM_PER_PX, SCALE_MM_PER_PX],
                                    }
                                ],
                                "planes": planes,
                            }
                        ],
                    }
                ]
            }
        ],
    }
    if subject_id is not None:
        acquisition["subject_id"] = subject_id
    return acquisition


def minimal_metadata(plane_count: int = 2) -> dict:
    """Return a flat minimal ``metadata.json`` dict.

    Parameters
    ----------
    plane_count : int, optional
        Number of imaging planes; 2 gives one coupled pair.

    Returns
    -------
    dict
        The raw minimal document.
    """
    return {
        "frame_rate": FRAME_RATE,
        "instrument_id": INSTRUMENT_ID,
        "subject_id": SUBJECT_ID,
        "excitation_nm": EXCITATION_NM,
        "emission_nm": EMISSION_NM,
        "session_start_time": SESSION_START_TIME,
        "planes": [
            {
                "plane_index": index,
                "coupled_plane_index": index // 2,
                "structure": STRUCTURES[index % len(STRUCTURES)],
                "depth": 150 + 50 * index,
                "um_per_pixel": SCALE_UM_PER_PX,
                "width": FOV_WIDTH,
                "height": FOV_HEIGHT,
            }
            for index in range(plane_count)
        ],
    }


def subject(genotype: str = GENOTYPE) -> dict:
    """Return a minimal ``subject.json`` dict."""
    return {"subject_id": SUBJECT_ID, "genotype": genotype}


def procedures() -> dict:
    """Return a minimal v1-shaped ``procedures.json`` dict."""
    return {
        "subject_procedures": [
            {
                "procedures": [
                    {"injection_materials": [{"name": INJECTION_MATERIAL}]},
                ]
            }
        ]
    }


def data_description(name: str = DATASET_NAME) -> dict:
    """Return a minimal ``data_description.json`` dict."""
    return {"name": name}


def platform(frame_rate: float = FRAME_RATE) -> dict:
    """Return a minimal v1 ``platform.json`` dict carrying a frame rate."""
    return {"imaging_plane_groups": [{"acquisition_framerate_Hz": frame_rate}]}
