from NFACT.decomposition.decomp import normalise_components
from NFACT.decomposition.matrix_handling import load_fdt_matrix

import numpy as np
from scipy.optimize import nnls
import os


def dualreg(
    connectivity_matrix: np.array,
    components: dict,
    algo: str,
    normalise: str,
    glm: bool = False,
):
    """
    Wrapper function to perform dual regression
    and save results

    Parameters
    ----------
    connectivity_matrix: np.array
        connectivity matrix of a single subject
    component: dict
        dictionary of components
    """

    if algo == "ica":
        dual_reg = ICA_dual_regression(connectivity_matrix, components)
    if algo == "nfm":
        return
    if normalise:
        demean = True if algo == "ica" else False
        normalise_components(dual_reg["grey_matter"], dual_reg["white_matter"], demean)

    if glm:
        return dual_reg


class Dual_regression:
    def __init__(
        self,
        algo: str,
        normalise: str,
        parallel: str,
        list_of_files: list,
        component: dict,
        glm: bool,
    ) -> None:
        self.algo = algo
        self.normalise = normalise
        self.parallel = parallel
        self.list_of_file = list_of_files
        self.component = component
        self.glm = glm

        if glm:
            self.glm_data = {
                "dualreg_on_G": [],
                "dualreg_on_W": [],
            }

    def fit(self) -> None:
        if self.parallel:
            print("Not implemented yet")
            return None
        if not self.parallel:
            self.__run_dual_regress_single()

    def __run_dual_regress_single(self) -> None:
        decomp = (
            self.__ICA_dual_regression
            if self.algo == "ica"
            else self.__nfm_dual_regression
        )
        for subject in self.list_of_file:
            self.connectivity_matrix = load_fdt_matrix(
                os.path.join(subject, "fdt_matrix2.dot")
            )
            decomp()

    def __ICA_dual_regression(
        self,
    ) -> None:
        """
        Dual regression method for ICA.
        Regresses the invidiual connectivity matrix
        onto the group components.

        If white matter component then regresses
        grey matter map onto connectivity matrix and vice versa.

        """
        self.wm_component_grey_map = (
            np.linalg.pinv(self.component["white_components"].T)
            @ self.connectivity_matrix.T
        ).T
        self.wm_component_white_map = (
            np.linalg.pinv(self.wm_component_grey_map) @ self.connectivity_matrix
        )
        self.gm_component_white_map = (
            np.linalg.pinv(self.component["grey_components"]) @ self.connectivity_matrix
        )
        self.gm_component_grey_map = (
            np.linalg.pinv(self.gm_component_white_map.T) @ self.connectivity_matrix.T
        ).T

        if self.glm:
            self.glm_data["dualreg_on_G"].append(self.gm_component_grey_map)
            self.glm_data["dualreg_on_W"].append(self.wm_component_white_map)

    def __nfm_dual_regression(self) -> None:
        """
        Dual regression method for NFM.
        """
        self.gm_component_grey_map = np.array(
            [
                nnls(
                    self.component["grey_components"], self.connectivity_matrix[:, col]
                )[0]
                for col in range(self.connectivity_matrix.shape[1])
            ]
        ).T
        self.wm_component_white_map = np.array(
            [
                nnls(
                    self.component["white_components"], self.connectivity_matrix[:, col]
                )[0]
                for col in range(self.connectivity_matrix.shape[0])
            ]
        ).T

        if self.glm:
            self.glm_data["dualreg_on_G"].append(self.gm_component_grey_map)
            self.glm_data["dualreg_on_W"].append(self.wm_component_white_map)

    def return_data_for_glm(self) -> dict:
        return self.glm_data


def ICA_dual_regression(
    connectivity_matrix: np.array,
    component: dict,
) -> dict:
    """
    Dual regression function for ICA.
    Regresses the invidiual connectivity matrix
    onto the group components.

    If white matter component then regresses
    grey matter map onto connectivity matrix and vice versa.

    Parameters
    ----------
    connectivity_matrix: np.array
        connectivity matrix of a single subject
    component: dict
        dictionary of components

    Returns
    -------
    dictionary: dict
        dictionary of grey and white matter components


    """
    wm_component_grey_map = (
        np.linalg.pinv(component["white_components"].T) @ connectivity_matrix.T
    ).T
    wm_component_white_map = np.linalg.pinv(wm_component_grey_map) @ connectivity_matrix
    gm_component_white_map = (
        np.linalg.pinv(component["grey_components"]) @ connectivity_matrix
    )
    gm_component_grey_map = (
        np.linalg.pinv(gm_component_white_map.T) @ connectivity_matrix.T
    ).T

    return {
        "white_matter": wm_component_white_map,
        "grey_matter": gm_component_grey_map,
    }
