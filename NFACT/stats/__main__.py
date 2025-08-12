from NFACT.stats.stats_args import nfact_stats_args
from NFACT.stats.stats_functions import split_component_type
from NFACT.base.signithandler import Signit_handler
from NFACT.base.utils import colours, nprint
from NFACT.base.setup import (
    check_algo,
    get_subjects,
    check_subject_exist,
    check_arguments,
    get_paths,
    check_nfact_decomp_directory,
)
from NFACT.base.filesystem import delete_folder, make_directory
import os
import glob


def nfactstats_main(args: dict = None):
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

    # check subjects exist
    args = get_subjects(args, key_name="dr_output")
    check_subject_exist(args["dr_output"])
    check_nfact_decomp_directory(paths["component_path"], paths["group_average_path"])
    component_files = split_component_type(args["dr_output"])
    component_files["group_white"] = os.path.join(
        paths["component_path"], f"W_{args['algo']}_dim{args['dim']}.nii.gz"
    )
    component_files["group_grey"] = glob.glob(
        os.path.join(paths["component_path"], f"G_{args['algo']}_dim{args['dim']}*")
    )
    del (args["dr_output"], paths)
    nprint(
        f"{col['plum']}Number of subject:{col['reset']} {len(component_files['dr_grey'])}"
    )
    nprint(f"{col['plum']}Stats Directory:{col['reset']} {stats_dir}")
    make_directory(stats_dir)

    if to_exit:
        exit(0)


if __name__ == "__main__":
    args = nfact_stats_args()
    nfact_stats_args(args)
    exit(0)
