from NFACT.dual_reg.nfact_dr_functions import get_group_level_components
from NFACT.base.matrix_handling import load_fdt_matrix
from NFACT.dual_reg.dual_regression import (
    nmf_dual_regression,
    ica_dual_regression,
    run_decomp,
)
from NFACT.dual_reg.nfact_dr_functions import save_dual_regression_images
from NFACT.base.utils import colours, error_and_exit, nprint
import argparse
import os
import numpy as np


def cluster_mode_args() -> dict:
    """
    Function to pass cmd
    arguements to the dual_regression_pipeline
    pipeline if ran directly.

    Parameters
    ----------
    None

    Returns
    -------
    dict: dictionary
        dict of cmd options
    """
    parser = argparse.ArgumentParser(description="Run Dual Regression")
    parser.add_argument(
        "--fdt_path", required=True, help="Directory to individual subject fdt path"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the output components."
    )
    parser.add_argument(
        "--component_path", required=True, help="Directory to components path."
    )
    parser.add_argument(
        "--group_average_path", required=True, help="Path to group averages."
    )
    parser.add_argument("--algo", required=True, help="Which algo has been run")
    parser.add_argument("--seeds", required=True, nargs="+", help="Path to seed(s).")
    parser.add_argument("--id", required=True, help="Subject ID.")
    parser.add_argument("--roi", nargs="+", default=False, help="Path to roi(s).")
    parser.add_argument(
        "--parallel", default=1, type=int, help="Number of cores to parallel with"
    )
    return vars(parser.parse_args())


def dual_regression_pipeline(
    fdt_path: str,
    output_dir: str,
    component_path: str,
    group_average_path: str,
    algo: str,
    seeds: str,
    sub_id: str,
    roi: str,
    parallel: int = 1,
    components: np.ndarray = False,
) -> None:
    """
    The dual regression pipeline function.
    This function either runs the pipeline
    locally or can be submitted directly
    to the cluster

    Parameters
    ----------
    fdt_path: str
        path to fdt matrix
    output_dir: str
        output directory
    component_path: str,
        path to group components
    group_average_path: str,
    algo: str,
    seeds: str,
    sub_id: str,
    roi: str,
    parallel: int = 1,

    components: np.ndarray
        group components.
        Can be False (default)
        and pipeline will get group
        components

    Returns
    -------
    None
    """
    col = colours()
    nprint(
        f"{col['plum']}Number of cores{col['reset']}: ",
        parallel,
        to_flush=True,
    )
    nprint("-" * 100)
    if not components:
        nprint(
            f"{col['pink']}Obtaining{col['reset']}: Group Level Components",
            to_flush=True,
        )

        try:
            components = get_group_level_components(
                component_path,
                group_average_path,
                seeds,
                roi,
            )
        except Exception:
            error_and_exit(False, "Unable to find components")

    nprint(f"{col['plum']}Subject ID{col['reset']}: {sub_id}", to_flush=True)
    nprint(f"{col['pink']}Obtaining{col['reset']}: FDT Matrix")

    try:
        matrix = load_fdt_matrix(os.path.join(fdt_path, "fdt_matrix2.dot"))
    except Exception:
        error_and_exit(False, f"Unable to load {sub_id} fdt matrix")

    method = nmf_dual_regression if algo.lower() == "nmf" else ica_dual_regression
    nprint(f"{col['pink']}DR Method:{col['reset']} {method}")

    nprint(f"{col['pink']}Running{col['reset']}: Dual Regression", to_flush=True)
    try:
        dr_results = run_decomp(method, components, matrix, parallel)
    except Exception as e:
        error_and_exit(False, f"Dual regression failed due to {e}")

    nprint(f"{col['pink']}Saving{col['reset']}: Components", flush=True)

    try:
        save_dual_regression_images(
            dr_results,
            output_dir,
            seeds,
            algo.upper(),
            dr_results["white_components"].shape[0],
            sub_id,
            fdt_path,
            roi,
        )
    except Exception as e:
        error_and_exit(False, f"Unable to save images due to {e}")

    nprint(f"{col['pink']}Completed{col['reset']}: {sub_id}", to_flush=True)
    return None


if __name__ == "__main__":
    args = cluster_mode_args()
    dual_regression_pipeline(
        fdt_path=args["fdt_path"],
        output_dir=args["output_dir"],
        component_path=args["component_path"],
        group_average_path=args["group_average_path"],
        algo=args["algo"],
        seeds=args["seeds"],
        sub_id=args["id"],
        roi=args["roi"],
        parallel=args["parallel"],
    )
