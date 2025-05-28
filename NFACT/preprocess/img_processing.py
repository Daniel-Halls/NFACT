import subprocess
import os
from NFACT.base.utils import error_and_exit


def seeds_to_ascii(surfin: str, roi: str, surfout: str) -> None:
    """
    Function to create seeds from
    surfaces.

    Parameters
    ----------
    surfin: str
        input surface
    roi: str,
        roi to restrict seeding
    surfout: str
        name of output surface.
        Needs to be full path

    Returns
    -------
    None
    """

    try:
        run = subprocess.run(
            [
                "surf2surf",
                "-i",
                surfin,
                "-o",
                surfout,
                f"--values={roi}",
                "--outputtype=ASCII",
            ],
            capture_output=True,
        )

    except subprocess.CalledProcessError as error:
        error_and_exit(False, f"Error in calling surf2surf: {error}")
    except KeyboardInterrupt:
        run.kill()

    if run.returncode != 0:
        error_and_exit(
            False,
            f"FSL surf2surf failure due to {run.stderr}. Unable to create asc surface",
        )


def downsample_target2(
    target_img: str,
    output_dir: str,
    resolution: str,
    reference_img: str,
    interpolation_strategy: str,
) -> None:
    """
    Function to create target2 image

    Parameters
    ----------
    target_img: str
        string to target image
    output: str
        string to output directory
    resolution: str
        resolution of target2
    reference_img: str
        reference input
    interpolation_strategy: str
        interpolation, either
        trilinear,
        nearestneighbour,
        sinc,
        spline

    Returns
    -------
    None
    """

    try:
        run = subprocess.run(
            [
                "flirt",
                "-in",
                target_img,
                "-out",
                output_dir,
                "-applyisoxfm",
                str(resolution),
                "-ref",
                reference_img,
                "-interp",
                interpolation_strategy,
            ],
            capture_output=True,
        )
    except FileNotFoundError:
        error_and_exit(False, "Unable to find reference image. Please check it exists")
    except subprocess.CalledProcessError as error:
        error_and_exit(False, f"Error in calling FSL flirt: {error}")
    except KeyboardInterrupt:
        run.kill()

    if run.returncode != 0:
        error_and_exit(
            False, f"FSL FLIRT failure due to {run.stderr}. Unable to build target2"
        )


def fslmaths_cmd(command: list) -> None:
    """
    Wrapper function around fslmaths

    Parameters
    ----------
    command: list
        fslmaths command

    Returns
    -------
    None
    """
    command.insert(0, "fslmaths")
    try:
        run = subprocess.run(command, capture_output=True)
    except subprocess.CalledProcessError as error:
        error_and_exit(False, f"Error in calling fslmaths: {error}")
    except KeyboardInterrupt:
        run.kill()

    if run.returncode != 0:
        error_and_exit(False, f"fslmaths failed due to {run.stderr}.")


def clean_target2(nfactpp_diretory: str, default_ref: str) -> None:
    """
    Wrapper function around a bunch
    of fslmaths commands to remove
    ventricles from target2 img.

    Parameters
    ----------
    nfactpp_diretory: str
       path to nfact directory
    default_ref: str
        default reference image

    Returns
    -------
    None
    """
    mask = os.path.join(
        os.getenv("FSLDIR"),
        "data",
        "atlases",
        "HarvardOxford",
        "HarvardOxford-sub-maxprob-thr0-2mm.nii.gz",
    )
    # Get ventricle from HarvardOxford
    fslmaths_cmd(
        [
            mask,
            "-thr",
            "14",
            "-uthr",
            "14",
            "-bin",
            f"{nfactpp_diretory}/ventricle_1",
        ]
    )
    # Get other ventricle from HarvardOxford
    fslmaths_cmd(
        [
            mask,
            "-thr",
            "3",
            "-uthr",
            "3",
            "-bin",
            f"{nfactpp_diretory}/ventricle_2",
        ]
    )
    # Add them together
    fslmaths_cmd(
        [
            f"{nfactpp_diretory}/ventricle_1",
            "-add",
            f"{nfactpp_diretory}/ventricle_2",
            "-bin",
            f"{nfactpp_diretory}/ven_mask",
        ]
    )
    # Dilate the mask
    fslmaths_cmd(
        [
            f"{nfactpp_diretory}/ven_mask",
            "-dilM",
            f"{nfactpp_diretory}/ven_mask_dilated",
        ]
    )
    # Invert the mask
    fslmaths_cmd(
        [f"{nfactpp_diretory}/ven_mask_dilated", "-binv", f"{nfactpp_diretory}/ven_inv"]
    )
    # Subtract the maks from the img by multiplication
    fslmaths_cmd(
        [
            default_ref,
            "-mul",
            f"{nfactpp_diretory}/ven_inv",
            f"{nfactpp_diretory}/target2",
        ]
    )
    # Remove all intermediate files
    files_to_delete = [
        "ven_mask_dilated",
        "ventricle_1",
        "ventricle_2",
        "ven_mask",
        "ven_inv",
    ]
    [
        os.remove(os.path.join(nfactpp_diretory, f"{file}.nii.gz"))
        for file in files_to_delete
    ]


def binarise_target2(target2_path: str) -> None:
    """
    Function to binarize target2 mask

    Parameters
    ----------
    target2_path: str
        path to target2 image

    Returns
    --------
    None
    """
    fslmaths_cmd([target2_path, "-thr", "1.0", target2_path])
    fslmaths_cmd([target2_path, "-bin", target2_path])
