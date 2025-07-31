import numpy as np

def get_component_data(
        component_map: np.ndarray, 
        component_number: np.ndarray, 
        surface: bool=False
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
        surface: bool=False
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
        group_map: np.ndarray, 
        subject_map: np.ndarray, 
        surface: bool=False
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

    return np.array([
            component_correlation(group_map, subject_map, comp, surface) 
            for comp in range(group_map.shape[-1])
        ])
