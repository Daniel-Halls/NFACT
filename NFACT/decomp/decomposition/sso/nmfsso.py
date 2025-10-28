from NFACT.decomp.decomposition.sso.sso_functions import (
    compute_similairty_matrix,
    sim2dis,
    clustering_components,
    compute_r_index,
    idx2centrotype,
    projection,
    cluster_scores,
)
from NFACT.decomp.decomposition.sso.sso_plotting import (
    plot_matrix,
    plot_rindex,
    plot_cluster_stats,
    plot_network,
)
from NFACT.base.utils import error_and_exit
from sklearn.decomposition import NMF
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy as np
from multiprocessing import shared_memory
from joblib import Parallel, delayed
import os

warnings.filterwarnings("ignore")


@ignore_warnings(category=ConvergenceWarning)
def nmf_decomp(parameters: dict, fdt_matrix: np.ndarray) -> dict:
    """
    Function to perform NFM.

    Parameters
    ----------
    parameters: dict
        dictionary of hyperparameters
    fdt_matrix: np.ndarray
        matrix to perform decomposition
        on

    Returns
    -------
    dict: dictionary
        dictionary of grey and white matter
        components
    """
    decomp = NMF(**parameters)
    try:
        grey_matter = decomp.fit_transform(fdt_matrix)
    except Exception as e:
        error_and_exit(False, f"Unable to perform NMF due to {e}")
    return {"grey_components": grey_matter, "white_components": decomp.components_}


class NMFsso:
    """
    NMFsso class. Creates can run with and
    without parallelization. This is based on
    ICAsso but paired down to suit NFACT

    Usage
    -----
    est = NMFsso(
        fdtmat=fdt_mat,
        num_int=15,
        nmf_params=nmf_params,
        n_jobs=5
    )

    components = est.run()

    """

    def __init__(
        self,
        fdt_mat: np.ndarray,
        num_int: int,
        nmf_params: dict,
        n_jobs: int,
    ) -> None:
        self.num_int = num_int
        self.nmf_params = nmf_params.copy()
        self.n_jobs = n_jobs
        self.fdt_mat = fdt_mat
        self.nmf_params["init"] = "random"
        self.nmf_params["random_state"] = None

    def _results(self):
        return {"grey": [], "white": []}

    def _run_single_shared(self, shm_name, shape, dtype, nmf_params):
        """
        Worker function: attach to shared memory and run one NMF decomposition.
        """
        shm = shared_memory.SharedMemory(name=shm_name)
        fdt_mat = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        nmf_state = nmf_decomp(nmf_params, fdt_mat)
        shm.close()  # detach, do NOT unlink here
        return nmf_state["grey_components"], nmf_state["white_components"]

    def _parallel_run(self):
        self.shared_shape = self.fdt_mat.shape
        self.shared_dtype = self.fdt_mat.dtype
        self.shm = shared_memory.SharedMemory(create=True, size=self.fdt_mat.nbytes)
        np.ndarray(self.shared_shape, dtype=self.shared_dtype, buffer=self.shm.buf)[
            :
        ] = self.fdt_mat
        self.shm_name = self.shm.name
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        print(
            f"Running {self.num_int} NMF decompositions in parallel (n_jobs={self.n_jobs})..."
        )
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_single_shared)(
                iterat,
                self.shm_name,
                self.shared_shape,
                self.shared_dtype,
                self.nmf_params,
            )
            for iterat in range(self.num_int)
        )

        # Collect results
        nmf_sso_results = self._results()
        for grey, white in results:
            nmf_sso_results["grey"].append(grey)
            nmf_sso_results["white"].append(white)

        # Clean up shared memory
        self.shm.close()
        self.shm.unlink()

        return nmf_sso_results

    def _single_run(self) -> dict:
        """
        Single run NMF-sso method

        Parameters
        ----------
        None

        Returns
        -------
        nmf_sso_results: dict
            dict of grey and white matter_components
        """

        nmf_sso_results = self._results()
        for iterat in range(self.num_int):
            print(f"NMF: run {iterat+1}/{self.num_int}")
            nmf_state = nmf_decomp(self.nmf_params, self.fdt_mat)
            nmf_sso_results["grey"].append(nmf_state["grey_components"])
            nmf_sso_results["white"].append(nmf_state["white_components"])

        return nmf_sso_results

    def run(self) -> dict:
        """
        Run method for NMF sso. Will run either parallel
        or single run depending on number of jobs given

        Parameters
        -----------
        None

        Returns
        -------
        nmf_sso_results: dict
            dict of grey and white matter_components
        """

        if self.n_jobs > 1:
            nmf_sso_results = self._parallel_run()
        else:
            nmf_sso_results = self._single_run()

        return nmf_sso_results


def nmf_sso_plotting_wrapper(
    output_dir: str,
    sim: np.ndarray,
    dis: np.ndarray,
    rindex: np.ndarray,
    partitions: np.ndarray,
    centroids: np.ndarray,
) -> None:
    """
    Function wrapper around plotting different
    measures

    Parameters
    ----------
    output_dir: str
        output directory of nfact
    sim: np.ndarray
        similairty matrix
    dis: np.ndarray
        dis-similairty matrix
    rindex: np.ndarray
        r index scores
    partitions: np.ndarray
        partition of cluster labels
    centroids: np.ndarray
        array of centroids

    Returns
    -------
    None
    """
    plotting_output = os.path.join(output_dir, "sso_output")
    coords = projection(dis)
    clust_score = cluster_scores(sim, partitions)
    plot_matrix(
        os.path.join(plotting_output, "similarity_matrix.tiff"),
        sim,
        "Similarity Matrix",
    )
    plot_matrix(
        os.path.join(plotting_output, "dissimilarity_matrix.tiff"),
        dis,
        "Dis-Similarity Matrix",
    )
    plot_rindex(rindex, os.path.join(plotting_output, "rindex.tiff"))
    plot_cluster_stats(
        clust_score["N"],
        clust_score["clusternumber"],
        clust_score["score"],
        clust_score["order"],
        os.path.join(plotting_output, "cluster_stats.tiff"),
    )
    plot_network(
        coords,
        partitions,
        sim,
        centroids,
        os.path.join(plotting_output, "cluster_network.tiff"),
    )


def nmf_sso(fdt_matrix: np.ndarray, parameters: dict, args: dict) -> dict:
    """
    Function to run nmf, either sso run
    or singular run.

    Parameters
    ----------
    fdt_matrix: np.ndarray
        fdt_matrix to decompose
    parameters: dict
        NMF parameters
    args: dict
        cmd arguments

    Returns
    -------
    dict: dictionary
        dictionary of grey and white matter
        components
    """
    if args["no_sso"]:
        return nmf_decomp(parameters, fdt_matrix)

    nmfsso_est = NMFsso(fdt_matrix, args["iterations"], parameters, args["n_cores"])
    results_of_comp = nmfsso_est.run()
    w_components = np.vstack(results_of_comp["white"])
    g_components = np.hstack(results_of_comp["grey"])

    sim = compute_similairty_matrix(w_components)
    dis = sim2dis(sim)
    partitions = clustering_components(dis)
    rindex = compute_r_index(dis, partitions)
    number_of_clusters = int(np.where(~np.isnan(rindex))[0].max())
    centroids = idx2centrotype(sim, partitions[number_of_clusters])
    nmf_sso_plotting_wrapper(
        args["outdir"], sim, dis, rindex, partitions[number_of_clusters], centroids
    )
    return {
        "grey_components": g_components[:, centroids],
        "white_components": w_components[centroids, :],
    }
