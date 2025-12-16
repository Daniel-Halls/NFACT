from NFACT.stats.stats_args import nfact_stats_args
from NFACT.base.filesystem import make_directory
from NFACT.base.signithandler import Signit_handler
from NFACT.base.utils import colours, nprint
from NFACT.stats.nfactstats import (
    process_nfactstats_args,
    NFACTStats,
)


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
    args = process_nfactstats_args(args)
    try:
        num_subjects = len(args["dr_output"])
    except KeyError:
        num_subjects = 1
    nprint(f"{col['plum']}Number of subject:{col['reset']} {num_subjects}")
    nprint(f"{col['plum']}Stats Directory:{col['reset']} {args['stats_dir']}")
    make_directory(args["stats_dir"], args["overwrite"], ignore_errors=True)
    nprint("-" * 100)
    nfactstats = NFACTStats()
    nfactstats.nfactstats_module(args["command"], args)
    nprint(f"\n{col['plum']}Finished:{col['reset']}")
    if to_exit:
        exit(0)


if __name__ == "__main__":
    args = nfact_stats_args()
    nfactstats_main(args)
    exit(0)
