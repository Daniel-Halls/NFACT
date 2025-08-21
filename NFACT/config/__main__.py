from NFACT.config.nfact_config_functions import (
    nfact_config_args,
    create_combined_algo_dict,
    save_to_json,
    create_subject_list,
    check_arguments,
    create_config,
)
from NFACT.base.utils import error_and_exit


def nfact_config_main() -> None:
    """
    Main nfact config function.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    args = nfact_config_args()
    check_arguments(args)
    if args["subject_list"]:
        print(f"Saving subject list to {args['output_dir']}\n")
        create_subject_list(args["subject_list"], args["output_dir"], args["file_name"])
        exit(0)

    if args["decomp_only"]:
        arguments = create_combined_algo_dict()
        file_name = f"{args['file_name']}.decomp"
    if args["config"]:
        arguments = create_config()
        file_name = f"{args['file_name']}.pipeline"

    print(f"Saving {file_name}.config to {args['output_dir']}\n")
    try:
        save_to_json(args["output_dir"], arguments, file_name)
    except Exception as e:
        error_and_exit(False, f"Unable to save file due to {e}", False)
    exit(0)


if __name__ == "__main__":
    nfact_config_main()
