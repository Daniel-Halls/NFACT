from NFACT.stats.stats_args import nfact_stats_args
from NFACT.base.signithandler import Signit_handler
from NFACT.base.utils import colours
from NFACT.base.setup import check_arguments, check_algo


def nfactstats_main(args: dict = None):
    Signit_handler()
    to_exit = False
    if not args:
        args = nfact_stats_args()
        to_exit = True
    col = colours()
    check_arguments(args, ["list_of_subjects", "nfact_folder", "output"])
    check_algo(args["algo"])

    if to_exit:
        exit(0)


if __name__ == "__main__":
    args = nfact_stats_args()
    nfact_stats_args(args)
    exit(0)
