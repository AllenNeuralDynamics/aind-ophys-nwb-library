# copied from comb

import json

# set up logger
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


MULTIPLANE_FILE_PARTS = {
    "processing_json": "processing.json",
    "params_json": "params.json",
    "registered_metrics_json": "registered_metrics.json",
    "average_projection_png": "average_projection.png",
    "max_projection_png": "maximum_projection.png",
    "motion_transform_csv": "motion_transform.csv",
    "extraction_h5": "extraction.h5",
    "dff_h5": "dff.h5",
    "extract_traces_json": "extract_traces.json",
    "events_oasis_h5": "events_oasis.h5",
    "classifier_h5": "classification.h5",
}


def singleplane_session_data_files(input_path: Union[str, Path]):
    """Find all data files in a single-plane session directory.

    Parameters
    ----------
    input_path : str or Path
        The parent processed directory containing the single-plane data.

    Returns
    -------
    dict
        A dictionary containing the file paths for each data file.
    """
    input_path = Path(input_path)

    # There should be exactly one subfolder inside
    # the single-plane parent folder
    img_path = ""
    subfolders = [f for f in input_path.iterdir() if f.is_dir()]
    if len(subfolders) == 1:
        singleplane_data_path = subfolders[0]
    if len(subfolders) == 0:
        singleplane_data_path = input_path

    for subfolder in subfolders:
        if "capsule" in subfolder.name:
            singleplane_data_path = subfolder
        # newer assets have the motion_correcton in its own folder
        if "MOp2_3_0" in subfolder.name:
            img_path = subfolder / "motion_correction"
        

    # This is the actual data directory

    data_files = {}
    for key, value in MULTIPLANE_FILE_PARTS.items():
        data_files[key] = find_data_file(
            singleplane_data_path, value
        )  # Search in the subfolder

    if img_path != "":
        data_files['max_projection_png'] = find_data_file(
            img_path, "maximum_projection.png"
        )  # Search in the subfolder
        data_files['average_projection_png'] = find_data_file(
            img_path, "average_projection.png"
        )  # Search in the subfolder

    return data_files


def multiplane_session_data_files(input_path: Union[str, Path], plane: str):
    """Find all data files in a multiplane session directory.

    Parameters
    ----------
    input_path : str or Path
        The path to the directory to search.
    plane : str
        The plane name.

    Returns
    -------
    dict
        A dictionary containing the file paths for each data file.
    """
    input_path = Path(input_path)
    data_files = {}
    for key, value in MULTIPLANE_FILE_PARTS.items():
        data_files[key] = find_data_file(input_path, plane + "_" + value)
    return data_files


def find_data_file(input_path, file_part, verbose=False):
    """Find a file in a directory given a partial file name.

    Parameters
    ----------
    input_path : str or Path
        The path to the directory to search.
    file_part : str
        The partial file name to search for.
    """
    input_path = Path(input_path)
    try:
        file = list(input_path.glob(f"**/*{file_part}*"))[0]
    except IndexError:
        if verbose:
            logger.warning(
                f"File with '{file_part}' not found in {input_path}"
            )
        file = None
    return file


def get_file_paths_dict(file_parts_dict, input_path):
    file_paths = {}
    for key, value in file_parts_dict.items():
        file_paths[key] = find_data_file(input_path, value)
    return file_paths


def check_ophys_folder(path):
    """ophys folders can have multiple names, check for all of them"""
    ophys_names = ["ophys", "pophys", "mpophys"]
    ophys_folder = None
    for ophys_name in ophys_names:
        ophys_folder = path / ophys_name
        if ophys_folder.exists():
            break
        else:
            ophys_folder = None

    return ophys_folder


def check_behavior_folder(path):
    behavior_names = ["behavior", "behavior_videos"]
    behavior_folder = None
    for behavior_name in behavior_names:
        behavior_folder = path / behavior_name
        if behavior_folder.exists():
            break
        else:
            behavior_folder = None
    return behavior_folder


def get_sync_file_path(input_path, verbose=False):
    """Find the Sync file"""
    file_parts = {}
    input_path = Path(input_path)
    try:
        # method 1: find sync_file by name
        file_parts = {"sync_h5": "_sync.h5"}
        sync_file_path = find_data_file(
            input_path, file_parts["sync_h5"], verbose=False
        )
    except IndexError:
        if verbose:
            logger.info(
                "file with '*_sync.h5' not found, trying platform json"
            )

    if sync_file_path is None:
        # method 2: load platform json
        # Note: sometimes fails if platform json has incorrect sync_file path
        logging.info(
            f"Trying to find sync file using platform json for {input_path}"
        )
        file_parts = {"platform_json": "_platform.json"}
        platform_path = find_data_file(
            input_path, file_parts["platform_json"]
        )
        with open(platform_path, "r") as f:
            platform_json = json.load(f)
        ophys_folder = check_ophys_folder(input_path)
        sync_file_path = ophys_folder / platform_json["sync_file"]

        if not sync_file_path.exists():
            logger.error(
                f"Unsupported data asset structure, \
                    sync file not found in {sync_file_path}"
            )
            sync_file_path = None
        else:
            logger.info(f"Sync file found in {sync_file_path}")

    return sync_file_path


def plane_paths_from_session(
    session_path: Union[Path, str],
    data_level: str = "raw",
    fovs: List = [],
    single_plane: bool = False,
) -> list:
    """Get plane paths from a session directory,
        supporting both singleplane and multiplane cases.

    Parameters
    ----------
    session_path : Union[Path, str]
        Path to the session directory
    data_level : str, optional
        Data level, by default "raw". Options: "raw", "processed"
    fovs: List, optional
        List of ophys fovs. Should be provided for all processed assets
    single_plane: bool, optional
        Bool indicating whether the sess is single_plane

    Returns
    -------
    list
        List of plane paths
    """
    session_path = (
        Path(session_path) if isinstance(session_path, str) else session_path
    )

    if single_plane:
        return [session_path]  # Singleplane session (no subdirectories)

    if fovs != [] and data_level == "processed":
        planes = []
        for fov in fovs:
            fov_plane = fov["targeted_structure"]
            fov_index = fov["index"]
            fov_pair = fov_plane + "_" + str(fov_index)
            planes.append(fov_pair)
    elif fovs == [] and data_level == "processed":
        raise ValueError("No Ophys fovs")
    if data_level == "raw":
        planes = [
            f for f in (session_path / "pophys").glob("*") if f.is_dir()
        ]
    return planes


def get_ophys_processed_file_paths(processed_path: Union[Path, str]) -> dict:
    """
    Create a dictionary of file paths for a processed ophys session,
        supporting both singleplane and multiplane.

    Parameters
    ----------
    processed_path : Union[Path, str]
        Path to the processed ophys session folder.

    Returns
    -------
    dict
        A dictionary containing file paths for each plane
            (or the session folder in singleplane).
    """
    processed_path = Path(processed_path)

    processed_plane_paths = plane_paths_from_session(
        processed_path, data_level="processed"
    )

    file_paths = {"planes": {}, "processed_path": processed_path}

    for plane_path in processed_plane_paths:
        plane_path = Path(plane_path)
        plane = plane_path.name
        file_paths["planes"][plane] = multiplane_session_data_files(
            plane_path
        )  # Replace with singleplane logic if needed
        file_paths["planes"][plane]["processed_plane_path"] = plane_path

    return file_paths
