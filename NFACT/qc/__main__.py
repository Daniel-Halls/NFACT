from NFACT.qc.nfactQc_args import nfact_qc_args
from NFACT.qc.nfactQc_functions import (
    nfactQc_dir,
    check_Qc_dir,
    get_images,
    create_nifti_hitmap,
    create_gifti_hitmap,
    get_data,
)
from NFACT.base.setup import check_arguments, check_algo, process_dim
from NFACT.base.utils import error_and_exit, colours
from NFACT.base.imagehandling import imaging_type
from NFACT.base.signithandler import Signit_handler
import os


def nfactQc_main(args: dict = None) -> None:
    """
    Main nfactQc function

    Parameters
    ----------
    arg: dict
        Set of command line arguments
        from nfact_pipeline
        Default is None

    Returns
    -------
    None
    """
    Signit_handler()
    to_exit = False
    if not args:
        args = nfact_qc_args()
        to_exit = True
    col = colours()
    check_arguments(args, ["nfact_folder", "dim", "algo"])
    error_and_exit(
        os.path.exists(args["nfact_folder"]),
        "NFACT decomp directory does not exist.",
        False,
    )

    args["algo"] = check_algo(args["algo"]).upper()
    args["dim"] = process_dim(args["dim"])
    nfactQc_directory = os.path.join(args["nfact_folder"], "nfactQc")
    images = get_images(args["nfact_folder"], args["dim"], args["algo"])

    try:
        white_name = os.path.basename(images["white_image"][0]).split(".")[0]
    except IndexError:
        error_and_exit(
            False,
            "Unable to find imaging files. Please check nfact_decomp directory",
            False,
        )
    print(f"{col['plum']}nfactQC directory:{col['reset']} {nfactQc_directory}")
    nfactQc_dir(nfactQc_directory, args["overwrite"])
    check_Qc_dir(nfactQc_directory, white_name)
    print("\nQC")
    print("-" * 100)
    print(f"{col['pink']}QC:{col['reset']} WM")
    img_type = imaging_type(images["white_image"][0])
    w_data = get_data(img_type, images["white_image"][0])
    create_nifti_hitmap(
        images["white_image"][0],
        os.path.join(nfactQc_directory, white_name),
        args["threshold"],
    )

    create_nifti_hitmap(
        images["white_image"][0],
        os.path.join(nfactQc_directory, f"{white_name}_raw"),
        args["threshold"],
        normalize=False,
    )
    print(f"{col['pink']}QC:{col['reset']} GM")
    for grey_img in images["grey_images"]:
        grey_name = os.path.basename(grey_img).split(".")[0]
        img_type = imaging_type(grey_img)
        if img_type == "gifti":
            create_gifti_hitmap(
                grey_img, os.path.join(nfactQc_directory, grey_name), args["threshold"]
            )
            create_gifti_hitmap(
                grey_img,
                os.path.join(nfactQc_directory, f"{grey_name}_raw"),
                args["threshold"],
                normalize=False,
            )
        if img_type == "nifti":
            print(f"{col['pink']}QC GM Volume:{col['reset']} Zscoring")
            create_nifti_hitmap(
                grey_img, os.path.join(nfactQc_directory, grey_name), args["threshold"]
            )
            print(f"{col['pink']}QC GM Volume:{col['reset']} No thresholding")
            create_nifti_hitmap(
                grey_img,
                os.path.join(nfactQc_directory, f"{grey_name}_raw"),
                args["threshold"],
            )

    if to_exit:
        exit(0)


if __name__ == "__main__":
    nfact_qc_args()
    exit(0)
