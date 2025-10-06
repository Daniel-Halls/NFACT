from NFACT.stats.stats_args import nfact_stats_args
from NFACT.base.signithandler import Signit_handler
from NFACT.base.utils import colours, nprint
from NFACT.stats.nfactstats import (
    create_nfactstats_folder,
    process_nfactstats_args,
    NFACTStats,
)
import os


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
    nprint(f"{col['plum']}Number of subject:{col['reset']} {len(args['dr_output'])}")
    stats_dir = os.path.join(args["outdir"], "nfact_stats"), args["overwrite"]
    nprint(f"{col['plum']}Stats Directory:{col['reset']} {stats_dir}")
    create_nfactstats_folder(stats_dir, args["overwrite"])
    nprint("-" * 100)
    nfactstats = NFACTStats()
    nfactstats.nfactstats_module(args["command"], args)
    nprint(f"\n{col['plum']}Finished:{col['reset']}")
    if to_exit:
        exit(0)


if __name__ == "__main__":
    args = nfact_stats_args()
    nfact_stats_args(args)
    exit(0)
