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

    def loadings(self, **kwargs):
        """
        entry method into calculating
        component loadings
        """
        from NFACT.stats.stats_component_loadings import component_loadings_main

        component_loadings_main(kwargs)

    def statsmap(self, **kwargs):
        from NFACT.stats.statsmap import statsmap_main

        statsmap_main(kwargs)

    def nfactstats_module(self, command: str, args: dict):
        """
        Method to determine which nfact stats module to be ran
        based on user input
        """
        func = getattr(self, command)
        func(**{key: value for key, value in args.items() if key != "command"})


def get_group_level_decomp(args: dict, paths: dict) -> dict:
    """
    Function to get group level components

    Parameters
    ----------
    args: dict
        dict of cmdline args
    paths: dict
        dictionary of paths

    Returns
    -------
    args: dict
        cmdline args with
        group level components
    """
    check_nfact_decomp_directory(paths["component_path"], paths["group_average_path"])
    args["group_white"] = os.path.join(
        paths["component_path"], f"W_{args['algo']}_dim{args['dim']}.nii.gz"
    )
    args["group_grey"] = glob.glob(
        os.path.join(paths["component_path"], f"G_{args['algo']}_dim{args['dim']}*")
    )
    del paths
    return args


def process_nfactstats_args(args: dict) -> dict:
    """
    Function to process nfact stats args

    Parameters
    ----------
    args: dict
        cmdline args dictionary

    Returns
    -------
    args: dict
        processed cmdline args
    """
    args_to_check = ["list_of_subjects", "nfact_folder", "outdir"]
    if "group-only" in args:
        if args["group-only"]:
            args_to_check.remove("list_of_subjects")

    check_arguments(args, args_to_check)
    check_algo(args["algo"])
    args["nfact_decomp_dir"] = args.pop("nfact_folder")
    args["stats_dir"] = os.path.join(args["outdir"], "nfact_stats")
    paths = get_paths(args)
    if "list_of_subjects" in args_to_check:
        args = get_subjects(args, key_name="dr_output")
    args = get_group_level_decomp(args, paths)
    return args
