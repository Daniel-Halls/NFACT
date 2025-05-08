from NFACT.pipeline.nfact_pipeline_args import nfact_args
from NFACT.pipeline.nfact_pipeline_functions import (
    pipeline_args_check,
    build_module_arguments,
    compulsory_args_for_config,
    update_nfact_args,
    roi_file,
)
from NFACT.base.config import get_nfact_arguments, process_dictionary_arguments
from NFACT.base.utils import error_and_exit, colours, Timer
from NFACT.base.filesystem import make_directory, load_json, read_file_to_list
from NFACT.base.setup import does_list_of_subjects_exist
from NFACT.preprocess.__main__ import nfact_pp_main
from NFACT.preprocess.nfactpp_args import nfact_pp_splash
from NFACT.decomp.setup.args import nfact_decomp_splash
from NFACT.decomp.__main__ import nfact_decomp_main
from NFACT.dual_reg.nfact_dr_args import nfact_dr_splash
from NFACT.dual_reg.__main__ import nfact_dr_main
from NFACT.qc.__main__ import nfactQc_main
from NFACT.qc.nfactQc_args import nfact_Qc_splash
import os


def nfact_pipeline_main() -> None:
    """
    Main function for NFACT pipeline

    Parameters
    ----------
    None

    Returns
    ------
    None
    """

    time = Timer()
    time.tic()
    col = colours()
    args = nfact_args()

    # Build out command line argument input
    if not args["input"]["config"]:
        print(f"{col['plum']}NFACT input:{col['reset']} Command line")
        pipeline_args_check(args)
        global_arguments = get_nfact_arguments()
        global_arguments["global_input"] = args["input"]
        nfact_pp_args = build_module_arguments(
            global_arguments["nfact_pp"], args, "pre_process"
        )
        nfact_pp_args["n_cores"] = None
        global_arguments["nfact_decomp"]["algo"] = args["decomp"]["algo"]
        nfact_decomp_args = build_module_arguments(
            global_arguments["nfact_decomp"], args, "decomp"
        )
        nfact_dr_args = build_module_arguments(
            global_arguments["nfact_dr"], args, "decomp"
        )
        nfact_qc_args = build_module_arguments(global_arguments["nfact_qc"], args, "qc")

    # Build out arguments from config file
    if args["input"]["config"]:
        print(f"{col['plum']}NFACT input:{col['reset']} Config File")
        global_arguments = load_json(args["input"]["config"])
        nfact_pp_args = global_arguments["nfact_pp"]
        nfact_decomp_args = global_arguments["nfact_decomp"]
        nfact_dr_args = global_arguments["nfact_dr"]
        nfact_qc_args = global_arguments["nfact_qc"]
        compulsory_args_for_config(global_arguments)

    update_nfact_args(global_arguments)
    roi_file(global_arguments)
    print(f"{col['plum']}NFACT directory:{col['reset']} {nfact_pp_args['outdir']}")
    make_directory(nfact_pp_args["outdir"], ignore_errors=True)

    # Clean str instances of bool to actual bool type
    global_arguments = process_dictionary_arguments(global_arguments)
    nfact_pp_args = global_arguments["nfact_pp"]
    nfact_decomp_args = global_arguments["nfact_decomp"]
    nfact_dr_args = global_arguments["nfact_dr"]
    # Create tmp directory and decomposition subject list
    error_and_exit(
        does_list_of_subjects_exist(nfact_pp_args["list_of_subjects"]),
        "List of subjects doesn't exist. Used in this mode NFACT PP cannot get subjects from folder.",
    )

    # Run NFACT_PP
    if not global_arguments["global_input"]["pp_skip"]:
        print(f"{col['plum']}Running:{col['reset']} NFACT PP")
        print("-" * 100)
        print(nfact_pp_splash())
        nfact_pp_main(nfact_pp_args)

        print(f"{col['pink']}\nFinished running NFACT_PP{col['reset']}")
        print("-" * 100)
    else:
        print(f"\n{col['plum']}Skipping:{col['reset']} NFACT_PP")
        nfact_pp_args["list_of_subjects"] = read_file_to_list(
            nfact_pp_args["list_of_subjects"]
        )

    # Run NFACT_decomp
    print(f"{col['plum']}Running:{col['reset']} NFACT Decomp")
    print("-" * 100)
    error_and_exit(
        os.path.exists(os.path.join(nfact_pp_args["outdir"], "nfact_pp")),
        "NFACT PP has not been ran which NFACT needs. If pre-processing is seperate from decomposition please run the modules individually",
    )

    print(nfact_decomp_splash())
    nfact_decomp_main(nfact_decomp_args)
    print(f"{col['plum']}\nFinished:{col['reset']} NFACT Decomp")
    print("-" * 100)

    if not global_arguments["global_input"]["qc_skip"]:
        nfact_qc_args["nfact_folder"] = nfact_dr_args["nfact_decomp_dir"]
        nfact_qc_args["dim"] = nfact_decomp_args["dim"]
        nfact_qc_args["algo"] = nfact_decomp_args["algo"]
        nfact_qc_args["overwrite"] = False
        nfact_qc_args["threshold"] = (
            nfact_qc_args["threshold"] if not nfact_qc_args["threshold"] else 2
        )
        print(f"{col['plum']}Running:{col['reset']} NFACT Qc")
        print("-" * 100)
        print(nfact_Qc_splash())
        nfactQc_main(nfact_qc_args)
        print(f"{col['plum']}\nFinished:{col['reset']} NFACT Qc")
        print("-" * 100)
    else:
        print(f"{col['plum']}Skipping: {col['reset']} NFACT Qc")

    # Run NFACT_DR
    if (len(nfact_pp_args["list_of_subjects"]) > 1) and (
        not global_arguments["global_input"]["dr_skip"]
    ):
        print(f"\n\n{col['plum']}Running: {col['reset']} NFACT DR")
        print("-" * 100)
        print(nfact_dr_splash())
        nfact_dr_main(nfact_dr_args)
        print(f"{col['plum']}\nFinished:{col['reset']} NFACT DR")
        print("-" * 100)
    else:
        print(f"{col['plum']}Skipping: {col['reset']} NFACT DR")

    # Exit
    print(f"Decomposition pipeline took {time.how_long()}")
    print("Finished")
    exit(0)


if __name__ == "__main__":
    nfact_pipeline_main()
