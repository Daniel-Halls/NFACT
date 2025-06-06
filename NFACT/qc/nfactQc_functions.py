import os
from pathlib import Path
import re
from ast import literal_eval
import numpy as np
import nibabel as nb
from sklearn.preprocessing import StandardScaler
from glob import glob
import re
from NFACT.base.utils import error_and_exit, colours
from NFACT.base.setup import make_directory
from NFACT.base.imagehandling import (
    get_volume_data,
    get_surface_data,
    get_cifti_data,
    create_surface_brain_masks,
    add_nifti,

)

def extract_seeds_from_log(log_file: str) -> list:
    """
    Function to extract seeds from logs

    Parameters
    ----------
    log_file: str
        str of path to log file
    
    Returns
    --------
    list: list object
        list of seeds
    """
    log_path = Path(log_file).resolve()
    with open(log_path, "r") as log:
        content = log.read()

    match = re.search(r'"seeds":\s*(\[[^\]]*\])', content)
    if not match:
        raise ValueError("No seeds list found in the log file.")
    seeds_str = match.group(1)
    return literal_eval(seeds_str)


def get_latest_log_file(log_dir: str) -> str:
    """
    Function to get the latest
    log file from log directory

    Parameters
    ----------
    log_dir: str
        path to log directory

    Returns
    -------
    path: str
        str of path to latest log file
    """
    log_path = Path(log_dir)
    log_files = list(log_path.glob("*.log"))
    
    if not log_files:
        raise FileNotFoundError(f"No .log files found in {log_dir}")

    return max(log_files, key=lambda fil: fil.stat().st_mtime)
    #return os.path.join(log_dir, latest_log)


def save_gifit(filename: str, meta_data: object, img_data: np.ndarray) -> np.ndarray:
    """
    Function to save gifti file from a
    numpy array

    Parameters
    ---------
    filename: str
        filename of image to save
    meta_data: object
        meat data for img
    img_data: np.ndarray
        data to save in img

    Returns
    -------
    None
    """
    darrays = [
        nb.gifti.GiftiDataArray(
            img_data,
            datatype="NIFTI_TYPE_FLOAT32",
            intent=2001,
            meta=meta_data,
        )
    ]
    nb.gifti.GiftiImage(darrays=darrays, meta=meta_data).to_filename(
        f"{filename}.func.gii"
    )


def save_nifti(filename: str, meta_data: np.array, data: np.array) -> None:
    """
    Function to save nifti file from a
    numpy array

    Parameters
    ----------
    filename: str
        filename of image to save
    meta_data: np.array
        affine of image
    data: np.array
        array of data to save as image

    Returns
    -------
    None
    """
    nb.Nifti1Image(data.astype(np.float32), meta_data).to_filename(filename)


def normalization(img_data: np.array) -> np.array:
    """
    Function to normalise an image

    Parameters
    ----------
    img_data: np.array
        array of data to normalise

    Returns
    -------
    zscores: np.array
        array of zscores reshaped
        to be saved as a imge
    """
    n_voxels = np.prod(img_data.shape[:-1])
    n_vol = img_data.shape[-1]
    reshaped_data = img_data.reshape(n_voxels, n_vol)
    non_zero_mask = ~(reshaped_data == 0).all(axis=1)
    z_scores = np.zeros_like(reshaped_data)
    scaler = StandardScaler()
    z_scores[non_zero_mask] = scaler.fit_transform(reshaped_data[non_zero_mask])
    return z_scores.reshape(img_data.shape)


def hitcount(comp_scores: np.array, threshold: int) -> dict:
    """
    Function to count up the number of times

    Parameters
    ----------
    comp_scores: np.array
       component scores
    threshold: int
        threshold components at

    Return
    ------
    dict: dictionary
        dict of hitcount
        and bin_mask
    """
    binary_masks = np.abs(comp_scores) > threshold
    return {"hitcount": np.sum(binary_masks, axis=-1), "bin_mask": binary_masks}


def binary_mask(binary_masks: np.array) -> np.ndarray:
    """
    Function to create binary mask from
    numpy array

    Parameteres
    -----------
    binary_masks: np.array
         array of

    Returns
    --------
    np.ndarray: np.array
        numpy array of
    """
    return np.any(binary_masks, axis=-1).astype(np.uint8)


def scoring(img_data: np.array, normalize: bool, threshold: int) -> dict:
    """
    Function to zscore hitmap if normlaise given
    else returns the raw values.

    Parameters
    ----------
    img_data: np.array
        imaging data
    normalize: bool
        to normalise or not
    threshold: int
        threshold value

    Returns
    -------
    dict: dictionary
        dict of scores and threshold value
    """
    if normalize:
        return {"scores": normalization(img_data), "threshold": int(threshold)}
    return {"scores": img_data, "threshold": 0}


def volume_hitcount_maps(img_data: np.array, threshold: int, normalize=True) -> dict:
    """
    Function to create a binary coverage mask
    and a hitmap of voxels

    Parameters
    ----------
    img_data: np.array
        Array of image data
    threshold: int
        value to threshold array at

    Returns
    --------
    dictionary: dict
        dictionary of hitcount and bin mask
    """

    comp_scores = scoring(img_data, normalize, threshold)
    return hitcount(comp_scores["scores"], comp_scores["threshold"])


def surface_hitcount_maps(img_data: np.array, threshold: int, normalize=True) -> dict:
    """
    Function to create a binary coverage mask
    and a hitmap of voxels

    Parameters
    ----------
    img_data: np.array
        Array of image data
    threshold: int
        value to threshold array at

    Returns
    --------
    dictionary: dict
        dictionary of hitcount and bin mask
    """

    comp_scores = scoring(img_data, normalize, threshold)
    return np.sum(comp_scores["scores"] > comp_scores["threshold"], axis=0)


def get_data(data_type: str, img_path: str) -> np.ndarray:
    """
    Function to parse data type
    and get corresponding data

    Parameters
    ----------
    data_type: str
        which data type is
        the img
    img_path: str
        path to image

    Returns
    -------
    np.ndarray: array
        array of imaging data

    """
    data_loaders = {
        "gifti": get_surface_data,
        "nifti": get_volume_data,
        "cifti": get_cifti_data,
    }

    return data_loaders[data_type](img_path)


def create_hitmaps(
    img_type: str, img_data: np.ndarray, filename: str, threshold: int, img_to_load: str
) -> None:
    """
    Wrapper to create hitmaps

    Parameters
    ----------
    img_type: str
        str of img type
    img_data: np.ndarray
        array of img data
    filename: str
        filename of imag
    threshold: int
        threshold value
    img_to_load: str
        str of img to load

    Returns
    -------
    None
    """
    hitmap_loaders = {
        "gifti": create_gifti_hitmap,
        "nifti": create_nifti_hitmap,
        "cifti": create_cifti_hitmap,
    }

    img = nb.load(img_to_load)
    meta_data = (
        img.affine
        if img_type == "nifti"
        else (img.darrays[0].meta if img_type == "gifti" else None)
    )
    hitmap_loaders[img_type](img_data, filename, threshold, meta_data)
    threshold_filename = re.sub("nfactQc/", "nfactQc/threshold_", filename)
    hitmap_loaders[img_type](
        img_data, threshold_filename, threshold, meta_data, normalize=True
    )



def cifti_volume_utils(nfact_decomp_dir):
    breakpoint()
    logs_dir = os.path.join(nfact_decomp_dir, "logs")
    coords_dir = os.path.join(nfact_decomp_dir, "group_averages")
    try:
        log_with_seeds = get_latest_log_file(logs_dir)
        seeds = extract_seeds_from_log(log_with_seeds)
        coords = np.loadtxt(os.path.join(coords_dir, "coords_for_fdt_matrix2"), dtype=int)
    except Exception:
        return None
    return {
        "seeds": seeds,
        "coords": coords
    }




def create_cifti_hitmap(
    img_data: dict, filename: str, threshold: int, meta_data: object, normalize=False
): 
    col = colours()
    left_hitmap = surface_hitcount_maps(img_data["L_surf"].T, normalize, threshold)
    right_hitmap = surface_hitcount_maps(img_data["R_surf"].T, normalize, threshold)
    bm = create_surface_brain_masks(left_hitmap, right_hitmap)
    if "vol" in img_data.keys():
        vol_hitmap = volume_hitcount_maps(
            img_data["vol"].get_fdata(), normalize, threshold
        )
        cifti_vol = cifti_volume_utils(os.path.dirname(os.path.dirname(filename)))
        if not cifti_vol:
            print("Unable to save volume part of cifti") 
        breakpoint()


def create_gifti_hitmap(
    img_data: np.ndarray, filename: str, threshold: int, meta_data: object, normalize=False
) -> None:
    """
    Function to create hitmap from
    seed.

    Parameters
    ----------
    seed_path: str
        str of path to seed
    filename: str
        name of file. Does not
        need .func.gii

    Returns
    -------
    None
    """
    hitmap = surface_hitcount_maps(img_data, normalize, threshold)
    save_gifit(filename, meta_data, hitmap)


def create_nifti_hitmap(
    img_data: str, filename: str, threshold: int, meta_data: np.ndarray, normalize=False
) -> None:
    """
    Wrapper function to create a binary coverage mask
    and a hitmap of voxels. Saves images

    Parameters
    ----------
    img_path: str
        path to image
    img_name: str
        name of image
    threshold: int
        value to thres

    Returns
    --------
    float: float
       float of percentage coverage
    """
    maps = volume_hitcount_maps(img_data, threshold, normalize)
    image_name_hitmap = os.path.join(
        os.path.dirname(filename), f"hitmap_{os.path.basename(filename)}.nii.gz"
    )
    save_nifti(image_name_hitmap, meta_data, maps["hitcount"])
    coverage_map_mask = binary_mask(maps["bin_mask"])
    image_name_mask = os.path.join(
        os.path.dirname(filename), f"mask_{os.path.basename(filename)}.nii.gz"
    )
    save_nifti(image_name_mask, meta_data, coverage_map_mask)


def get_images(nfact_directory: str, dim: str, algo: str) -> dict:
    """
    Function to get images

    Parameters
    -----------
    nfact_directory: str
        path to nfact directory
    dim: str
        str of dimensions
    algo: str
        either NMF or ICA

    Returns
    -------
    dict: dictionary
         dict of grey and white images
    """
    # use glob as it also checks the images exist
    return {
        "grey_images": glob(
            os.path.join(
                nfact_directory, "components", algo, "decomp", f"G_{algo}_dim{dim}*"
            )
        ),
        "white_image": glob(
            os.path.join(
                nfact_directory,
                "components",
                algo,
                "decomp",
                f"W_{algo}_dim{dim}.nii.gz",
            )
        ),
    }


def nfactQc_dir(nfactQc_directory: str, overwrite: bool = False) -> None:
    """
    Function to create nfactQc directory.

    Parameters
    -----------
    nfact_directory: str
        path to nfact directory
    dim: str
        str of dimensions
    algo: str
        either NMF or ICA
    overwrite: bool
        overwrite directory

    Returns
    -------
    None
    """

    if os.path.exists(nfactQc_directory) and not overwrite:
        return None
    make_directory(nfactQc_directory, overwrite, ignore_errors=True)


def check_Qc_dir(nfactQc_directory: str, white_name: str) -> None:
    """
    Function to check Qc directory.
    Checks if files already exist and errors out
    if they do.

    Parameters
    ----------
    nfactQc_directory: str
        path to qc directory
    white_name : str
        name of wm image

    Returns
    -------
    None
    """

    if f"hitmap_{white_name}.nii.gz" in os.listdir(nfactQc_directory):
        error_and_exit(
            False, "QC images aleady exist. Please use --overwrite to continue", False
        )


def get_img_name(img: str) -> str:
    """
    Function to get imag name from
    file path.

    Parameters
    ----------
    img: str
        img path

    Returns
    -------
    name: str
        name of img
    """
    try:
        name = os.path.basename(img).split(".")[0]
        return name
    except IndexError:
        error_and_exit(
            False,
            "Unable to find imaging files. Please check nfact_decomp directory",
            False,
        )
