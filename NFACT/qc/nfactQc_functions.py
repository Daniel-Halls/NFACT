import os
import numpy as np
import nibabel as nb
from sklearn.preprocessing import StandardScaler
from glob import glob
from NFACT.base.utils import colours, error_and_exit
from NFACT.base.setup import make_directory
from NFACT.base.imagehandling import get_volume_data, get_surface_data, get_cifti_data


def save_gifit(filename: str, seed: object, surf_data: np.array):
    """
    Function to save gifti file from a
    numpy array

    Parameters
    ---------
    data: np.array
        array of data to save as image
    affine: np.array
        affine of image
    filename: str
        filename of image to save

    Returns
    -------
    None
    """
    darrays = [
        nb.gifti.GiftiDataArray(
            surf_data,
            datatype="NIFTI_TYPE_FLOAT32",
            intent=2001,
            meta=seed.darrays[0].meta,
        )
    ]
    nb.gifti.GiftiImage(darrays=darrays, meta=seed.darrays[0].meta).to_filename(
        f"{filename}.func.gii"
    )


def save_nifti(data: np.array, affine: np.array, filename: str) -> None:
    """
    Function to save nifti file from a
    numpy array

    Parameters
    ---------
    data: np.array
        array of data to save as image
    affine: np.array
        affine of image
    filename: str
        filename of image to save

    Returns
    -------
    None
    """
    nb.Nifti1Image(data.astype(np.float32), affine).to_filename(filename)


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


def nifti_hitcount_maps(img_data: np.array, threshold: int, normalize=True) -> dict:
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
    if data_type == "gifti":
        return get_surface_data(img_path)
    if data_type == "nifti":
        return get_volume_data(img_path)
    if data_type == "cifti":
        return get_cifti_data(img_path)


def hitmap(img_type: str, threshold: int, filename: str, img_to_load: str):
    """
    Wrapper to create hitmaps

    Parameters
    ----------
    """
    create_gifti_hitmap(img_to_load, filename, threshold)
    create_gifti_hitmap(img_to_load, filename, threshold, normalize=True)


def create_gifti_hitmap(
    seed_path: str, filename: str, threshold: int, normalize=True
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
    seed = nb.load(seed_path)

    combinearray = np.array(
        [seed.darrays[idx].data for idx, _ in enumerate(seed.darrays)]
    )

    comp_scores = scoring(combinearray, normalize, threshold)
    hitmap = np.sum(comp_scores["scores"] > comp_scores["threshold"], axis=0)
    save_gifit(filename, seed, hitmap)


def create_nifti_hitmap(
    img_path: str, img_name: str, threshold: int, normalize=True
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
    col = colours()
    img_comp = nb.load(img_path)
    img_data = img_comp.get_fdata()
    maps = nifti_hitcount_maps(img_data, threshold, normalize)
    print(f"{col['pink']}Image:{col['reset']} Saving Hitmap")
    image_name_hitmap = os.path.join(
        os.path.dirname(img_name), f"hitmap_{os.path.basename(img_name)}.nii.gz"
    )
    save_nifti(maps["hitcount"], img_comp.affine, image_name_hitmap)
    coverage_map_mask = binary_mask(maps["bin_mask"])
    image_name_mask = os.path.join(
        os.path.dirname(img_name), f"mask_{os.path.basename(img_name)}.nii.gz"
    )
    print(f"{col['pink']}Image:{col['reset']} Saving Binary Mask")
    save_nifti(coverage_map_mask, img_comp.affine, image_name_mask)


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
