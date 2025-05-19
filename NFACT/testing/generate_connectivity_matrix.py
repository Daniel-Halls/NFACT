import numpy as np
from pathlib import Path
import random


def generate_fdt_matrix2(
    seeds: np.array,
    targets: np.array,
    strength_range: tuple = (50, 500),
    num_connections: int = 100,
) -> np.ndarray:
    """
    Function to generate a random fdt matrix from
    a given list of "seeds" and "target"

    Parameters
    ----------
    seeds: np.array
       np.array of seeds
    targets: np.array
        np.array of targets
    strength_range: tuple
        strength range for connectivity matrix
    number_connections: int
        the number of connections

    Returns
    -------
    fdt_matrix: np.ndarray
        n by 3 fdt like connectivity matrix
    """

    seed = np.random.choice(seeds, num_connections)
    target = np.random.choice(targets, num_connections)
    weights = np.random.randint(
        strength_range[0], strength_range[1], size=num_connections
    )
    fdt_matrix = np.vstack((seed, target, weights)).T
    return fdt_matrix[np.lexsort((fdt_matrix[:, 1], fdt_matrix[:, 0]))]


def remove_duplicates(matrix: np.array) -> np.ndarray:
    """
    Function to remove duplicates from fdt matrix

    Parameters
    ----------
    matrix: np.array
        fdt matrix like

    Returns
    -------
    np.array: np.ndarray
        n by 3 fdt like connectivity matrix
        with duplicates removed
    """
    index = np.unique(matrix[:, 0:2], axis=0, return_index=True)[1]
    return np.delete(matrix, index, axis=0)


def fdt_matrix2(
    seeds: np.array,
    targets: np.array,
    strength_range: tuple = (50, 500),
    num_connections: int = 100,
) -> np.ndarray:
    """
    Function to generate a random fdt matrix from
    a given list of "seeds" and "target"

    Parameters
    ----------
    seeds: np.array
       np.array of seeds
    targets: np.array
        np.array of targets
    strength_range: tuple
        strength range for connectivity matrix
    number_connections: int
        the number of connections

    Returns
    -------
    fdt_matrix: np.ndarray
        n by 3 fdt like connectivity matrix
    """

    fdt_matrix = generate_fdt_matrix2(seeds, targets, strength_range, num_connections)
    return remove_duplicates(fdt_matrix).astype(int)


def waytotal(sub_dir: str) -> None:
    """
    Test function to generate
    a random waytotal file

    Parameters
    ----------
    sub_dir: str
        sub directory

    Returns
    -------
    None
    """
    waytotal = random.randint(10_000_000, 22_555_000)
    with open(f"{sub_dir}/waytotal", "w") as waytot_file:
        waytot_file.write(str(waytotal))


if __name__ == "__main__":
    print("Building random connectivity matricies")
    seeds = np.array(range(1, 100))
    targets = np.array(range(101, 200))
    for sub in range(1, 4):
        mat = fdt_matrix2(seeds, targets, num_connections=20000)
        new_row = np.array((seeds[-1], targets[-1], np.random.randint(50, 500)))
        mat = np.vstack([mat, new_row])
        current_working_dir = Path(__file__).parent
        np.savetxt(
            f"{current_working_dir}/test_data/sub-{sub}/fdt_matrix2.dot", mat, fmt="%i"
        )
        waytotal(f"{current_working_dir}/test_data/sub-{sub}")
