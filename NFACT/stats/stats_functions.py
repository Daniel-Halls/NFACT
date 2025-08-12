import numpy as np
import nibabel as nb
import os


def split_component_type(dr_output: list) -> dict:
    """
    Function to split the dual regression output
    into white matter and grey matter components

    Parameters
    -----------
    dr_output: list
        list of dr output

    Returns
    -------
    dict: dictionary
        dict of grey and white matter
        components
    """
    return {
        "dr_white": [
            w_file for w_file in dr_output if os.path.basename(w_file).startswith("W")
        ],
        "dr_grey": [
            w_file for w_file in dr_output if os.path.basename(w_file).startswith("G")
        ],
    }


class Component_loading:
    def __init__(
        self,
        white_group_component_path,
        grey_group_component_path,
        white_dr_paths,
        grey_dr_paths,
    ):
        self.white_group_component_path = white_group_component_path
        self.grey_group_component_path = grey_group_component_path
        self.white_dr_paths = white_dr_paths
        self.grey_dr_paths = grey_dr_paths
        self.grey_type = self.__grey_type()

    def __load_group_components(self):
        self.group_white = nb.load(self.white_group_component_path).get_fdata()
        self.group_grey

    def __grey_type(self):
        if "dscalar" in self.grey_group_component_path:
            return "cifti"
        if "gii" in self.grey_group_component_path:
            return "gifti"
        if "nii" in self.grey_group_component_path:
            return "nifti"


def get_component_data(
    component_map: np.ndarray, component_number: np.ndarray, surface: bool = False
) -> np.ndarray:
    """
    Function to get component
    values from component map

    Parameters
    ----------
    component_map: np.ndarray
        component volume or
        surface map
    component_number: np.ndarray
        which component to obtain
    surface: bool
        is component data a surface

    Returns
    -------
    component_map: np.ndarray
         flattened 1D array of
         component data
    """

    if surface:
        return component_map[..., component_number].flatten()
    return component_map[:, component_number].flatten()


def component_correlation(
    group_map: np.ndarray,
    subject_map: np.ndarray,
    component_number: int,
    surface: bool = False,
) -> float:
    """
    Function to calculate
    the correlation between
    subject and group component maps

    Parameters
    ----------
    group_map: np.ndarray
        group level component volume
        or surface map
    subject_map: np.ndarray
        subject level component volume
        or surface map
    component_number: np.ndarray
        which component to obtain
    surface: bool
        is component data a surface

    Returns
    -------
    float: correlation value
        correlation value between group
        and subject map
    """

    group_component = get_component_data(group_map, component_number, surface)
    subject_component = get_component_data(subject_map, component_number, surface)
    return np.corrcoef(group_component, subject_component)[0, 1]


def calculate_component_loading(
    group_map: np.ndarray, subject_map: np.ndarray, surface: bool = False
) -> np.ndarray:
    """
    Function to calculate
    component loading between
    subject and group components

    Parameters
    ----------
    group_map: np.ndarray
        group level component volume
        or surface map
    subject_map: np.ndarray
        subject level component volume
        or surface map
    surface: bool
        is component data a surface

    Returns
    -------
    np.ndarray: np.array
        array of component loadings
    """

    return np.array(
        [
            component_correlation(group_map, subject_map, comp, surface)
            for comp in range(group_map.shape[-1])
        ]
    )


def get_loadings(subjects: list):
    return np.vstack([get_subject_loadings() for sub in subjects])


def get_subject_loadings(component_list, group_component_path):
    group_comp = get_component_data(group_component_path)
    return None
