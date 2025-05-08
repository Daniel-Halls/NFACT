from NFACT.base.utils import error_and_exit
import os


def non_compulsory_arguments(additional_args: list = []) -> list:
    """
    Function to return what are non
    complusory arguments.

    Parameters
    ----------
    additional_args: list
        a list of additional arguments
        to make non complusory

    Returns
    -------
    list: list object
        List of non
        compulsory arguments
    """
    return [
        "target2",
        "config",
        "pp_skip",
        "dr_skip",
        "file_tree",
        "overwrite",
        "qc_skip",
    ] + additional_args


def get_compulsory_arguments(args):
    """
    Function to return non compulsory
    arguments

    Parameters
    ----------
    args: dict
        command line arguments

    Returns
    -------
    None

    """
    non_complusory_commands = [key for key in args["cluster"].keys()] + [
        "roi",
        "seedref",
        "target",
    ]
    if args["input"]["pp_skip"]:
        return non_compulsory_arguments(
            non_complusory_commands.extend(["bpx_path", "warps"])
        )
    if args["pre_process"]["file_tree"]:
        return non_compulsory_arguments(
            non_complusory_commands.extend["bpx_path", "warps", "seed"]
        )
    return non_compulsory_arguments(non_complusory_commands)


def pipeline_args_check(args: dict):
    """
    Function to check that compulsory
    args are given.

    Parameters
    ----------
    args: dict
        command line arguments

    Returns
    -------
    None
    """

    non_complusory = get_compulsory_arguments(args)
    for val in args.keys():
        [
            error_and_exit(
                args[val][arg], f"{arg} is not defined. Please define with --{arg}"
            )
            for arg in args[val].keys()
            if arg not in non_complusory
        ]


def build_args(args_dict: dict, module_dict: dict) -> dict:
    """
    Fuction to build out arguments
    from args dict to module dict

    Parameters
    ----------
    args_dict: dict
        dictionary of command line
        arguments
    module_dict: dict
        dict of module arguments

    Returns
    -------
    module_dict: dict
        dictionary of updated module
        arguments
    """
    for key in module_dict:
        if key in args_dict:
            module_dict[key] = args_dict[key]
    return module_dict


def build_module_arguments(module_dict: dict, args: dict, key: str):
    """
    Function to build out a module command line
    arguments.

    Parameters
    ----------
    module_dict: dict
        dict of module arguments
    args_dict: dict
        dictionary of command line
        arguments
    key: str
        str of key for argument dict
        to build out module dictionary

    Returns
    -------
    dict: dictionary
        dictionary of module arguments

    """
    module_dict = build_args(args["input"], module_dict)
    return build_args(args[key], module_dict)


def compulsory_args_for_config(args: dict) -> None:
    """
    Function to check for required
    arguments in nfact config file

    Parameters
    ----------
    args: dict
       arguments for NFACT

    Returns
    ------
    None
    """

    if ("Required" in args["global_input"]["seed"][0]) and (
        not args["nfact_pp"]["file_tree"]
    ):
        error_and_exit(False, "config file either needs seeds or file_tree argument")
    if args["nfact_pp"]["file_tree"]:
        args["global_input"]["seed"] = []
    [
        error_and_exit(False, f"{key} not given in config file. Please provide")
        for _, sub_dict in args.items()
        for key, value in sub_dict.items()
        if value == "Required"
    ]


def assign_nfactpp(args: dict) -> None:
    """
    Function to assign arguments in
    place for nfact_pp

    Parameters
    ----------
    args: dict
        dict of command line arguments

    Returns
    -------
    None
    """
    args["nfact_pp"]["seed"] = args["global_input"]["seed"]
    args["nfact_pp"]["list_of_subjects"] = args["global_input"]["list_of_subjects"]
    args["nfact_pp"]["overwrite"] = args["global_input"]["overwrite"]
    args["nfact_pp"].update(args["cluster"])


def assign_outdir(args: dict) -> None:
    """
    Function to assign outputdir in
    place for nfact modules

    Parameters
    ----------
    args: dict
        dict of command line arguments

    Returns
    -------
    None
    """
    for module in ["nfact_pp", "nfact_decomp", "nfact_dr"]:
        args[module]["outdir"] = os.path.join(
            args["global_input"]["outdir"], args["global_input"]["folder_name"]
        )


def assign_nfact_decomp(args: dict) -> None:
    """
    Function to assign outputdir in
    place for nfact_dr

    Parameters
    ----------
    args: dict
        dict of command line arguments

    Returns
    -------
    None
    """

    args["nfact_decomp"]["overwrite"] = args["global_input"]["overwrite"]
    args["nfact_decomp"]["list_of_subjects"] = os.path.join(
        args["nfact_pp"]["outdir"],
        "nfact_pp",
        "nfact_config.sublist",
    )
    args["nfact_decomp"]["seeds"] = os.path.join(
        args["nfact_pp"]["outdir"], "nfact_pp", "seeds_for_decomp.txt"
    )


def assign_nfact_dr(args: dict) -> None:
    """
    Function to assign outputdir in
    place for nfact_dr

    Parameters
    ----------
    args: dict
        dict of command line arguments

    Returns
    -------
    None
    """
    args["nfact_dr"]["nfact_decomp_dir"] = os.path.join(
        args["global_input"]["outdir"],
        args["global_input"]["folder_name"],
        "nfact_decomp",
    )
    args["nfact_dr"]["algo"] = args["nfact_decomp"]["algo"]
    args["nfact_dr"]["overwrite"] = args["global_input"]["overwrite"]
    args["nfact_dr"]["seeds"] = args["nfact_decomp"]["seeds"]
    args["nfact_dr"]["list_of_subjects"] = args["nfact_decomp"]["list_of_subjects"]
    args["nfact_dr"].update(args["cluster"])


def roi_file(args: dict) -> None:
    """
    Function to assign roi in
    place for nfact_ecomp/nfact_dr

    Parameters
    ----------
    args: dict
        dict of command line arguments

    Returns
    -------
    None
    """
    args["nfact_dr"]["roi"] = False
    if args["nfact_pp"]["roi"] or args["nfact_pp"]["file_tree"]:
        path = os.path.join(
            args["global_input"]["outdir"],
            args["global_input"]["folder_name"],
            "nfact_pp",
            "roi_for_decomp.txt",
        )
        args["nfact_decomp"]["roi"] = path
        args["nfact_dr"]["roi"] = path
    if args["nfact_decomp"]["roi"]:
        args["nfact_dr"]["roi"] = args["nfact_decomp"]["roi"]


def update_nfact_args(args: dict) -> None:
    """
    Function to update arguments
    in place for nfact modules.

    Parameters
    ----------
    args: dict
        dict of command line arguments

    Returns
    -------
    None
    """
    assign_outdir(args)
    assign_nfactpp(args)
    assign_nfact_decomp(args)
    assign_nfact_dr(args)
