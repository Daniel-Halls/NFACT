from NFACT.stats.stats_args import nfact_stats_args
from NFACT.stats.stats_component_loadings import Component_loading, save_components
from NFACT.base.signithandler import Signit_handler
from NFACT.base.utils import colours, nprint
from NFACT.base.setup import (
    check_algo,
    get_subjects,
    check_arguments,
    get_paths,
    check_nfact_decomp_directory,
)
from NFACT.base.filesystem import delete_folder, make_directory
import os
import glob


def nfactstats_main(args: dict = None) -> None:
    """
    Main function for nfact stats

    Parameters
    ----------
    arg: dict
        Set of command line arguments
        Default is None

    Returns
    -------
    None
    """
    Signit_handler()
    to_exit = False
    if not args:
        args = nfact_stats_args()
        to_exit = True
    col = colours()
    check_arguments(args, ["list_of_subjects", "nfact_folder", "outdir"])
    check_algo(args["algo"])
    args["nfact_decomp_dir"] = args.pop("nfact_folder")
    paths = get_paths(args)
    stats_dir = os.path.join(args["outdir"], "nfact_stats")
    if args["overwrite"]:
        delete_folder(stats_dir)

    args = get_subjects(args, key_name="dr_output")
    check_nfact_decomp_directory(paths["component_path"], paths["group_average_path"])
    args["group_white"] = os.path.join(
        paths["component_path"], f"W_{args['algo']}_dim{args['dim']}.nii.gz"
    )
    args["group_grey"] = glob.glob(
        os.path.join(paths["component_path"], f"G_{args['algo']}_dim{args['dim']}*")
    )
    del paths
    nprint(f"{col['plum']}Number of subject:{col['reset']} {len(args['dr_output'])}")
    nprint(f"{col['plum']}Stats Directory:{col['reset']} {stats_dir}")
    make_directory(stats_dir)
    nprint(f"\n{col['pink']}Running:{col['reset']} Component loadings")
    nprint("-" * 100)
    loadings = Component_loading(args["group_white"], args["group_grey"], args["dim"])
    component_loadings = loadings.run(args["dr_output"])
    nprint(f"\n{col['pink']}Saving:{col['reset']} Component loadings")
    nprint("-" * 100)
    save_components(
        component_loadings["white_correlations"],
        "W_component_loadings",
        stats_dir,
        args["dr_output"],
        args["no_csv"],
    )
    save_components(
        component_loadings["grey_correlations"],
        "G_component_loadings",
        stats_dir,
        args["dr_output"],
        args["no_csv"],
    )
    nprint("-" * 100)
    nprint(f"\n{col['plum']}Finished:{col['reset']}")
    if to_exit:
        exit(0)


if __name__ == "__main__":
    args = nfact_stats_args()
    nfact_stats_args(args)
    exit(0)
