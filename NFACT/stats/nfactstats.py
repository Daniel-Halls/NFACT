from NFACT.base.filesystem import delete_folder, make_directory
from NFACT.base.setup import (
    check_algo,
    get_subjects,
    check_arguments,
    get_paths,
    check_nfact_decomp_directory,
)
import os
import glob


class NFACTStats:
    """
    Main class of qpipeline

    Determines what part of the pipeline
    should be ran.

    Usage
    -----
    pipeline = Qpipeline()
    pipeline.qpipeline_handler(args['command'], args)
    """

    def __init__(self):
        pass

    def loadings(self, *kwargs):
        from NFACT.stats.stats_component_loadings import component_loadings_main

        component_loadings_main(kwargs)

    def statsmap(self, *kwargs):
        pass

    def nfactstats_module(self, command: str, args: dict):
        """
        Method to determine which nfact stats module to be ran
        based on user input
        """
        func = getattr(self, command)
        func(**{key: value for key, value in args.items() if key != "command"})


def create_nfactstats_folder(stats_dir, overwrite=False):
    if overwrite:
        delete_folder(stats_dir)
    make_directory(stats_dir)


def get_group_level_decomp(args, paths):
    check_nfact_decomp_directory(paths["component_path"], paths["group_average_path"])
    args["group_white"] = os.path.join(
        paths["component_path"], f"W_{args['algo']}_dim{args['dim']}.nii.gz"
    )
    args["group_grey"] = glob.glob(
        os.path.join(paths["component_path"], f"G_{args['algo']}_dim{args['dim']}*")
    )
    del paths
    return args


def process_nfactstats_args(args):
    """
    TODO: nfact statsmap may take only group level components
    so subject list is redudant
    """
    check_arguments(args, ["list_of_subjects", "nfact_folder", "outdir"])
    check_algo(args["algo"])
    args["nfact_decomp_dir"] = args.pop("nfact_folder")
    paths = get_paths(args)
    args = get_subjects(args, key_name="dr_output")
    args = get_group_level_decomp(args, paths)
    return args
