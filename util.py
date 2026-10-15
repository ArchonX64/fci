from numba import njit
import numpy as np

@njit
def spatial(orbital: int) -> int:
    """Gets the spatial orbital label from an orbital label

    Args:
        orbital (int): The orbital label

    Returns:
        int: The corresponding orbital label
    """
    return (orbital - 1) // 2

@njit
def spin(orbital: int) -> int:
    """Gets the spin of an electron label, being either 0 or 1
    
        Args:
            orbital (int): The orbital label
    
        Returns:
            int: The corresponding electron spin, either 0 or 1
        """
    return (orbital - 1) % 2

@njit
def oribtal_differences(o1: np.array, o2: np.array) -> list[np.array, np.array]:
    """Determines the number of differences between two orbital configuration

    Args:
        o1 (np.array): An array of electron label integers
        o2 (np.array): An array of electron label integers

    Returns:
        np.array: An array containing the electron labels with no counterpart in o1
        np.array: An array containing the electron labels with no counterpart in o2
    """
    o1_diffs = np.setdiff1d(o1, o2)
    o2_diffs = np.setdiff1d(o2, o1)
    return o1_diffs, o2_diffs

@njit
def get_phase(configs: np.ndarray, o1: int, o1_diffs: np.array, o2: int, o2_diffs: np.array) -> float:
    '''Determines the sign of each hamiltonian element, which is based on how
       many electron swaps occured.
       
    Args:
        configs (np.ndarray): The array of electron configurations
        o1: The index of the first configuration being comapred
        o2: The index of the second configuration being compared
        o1_diffs: The determined differences in o1 against o2
        o2_diffs: The determined differences in o2 against o1
        
    Returns:
        float: Either ~1 or -1, depending on the calculated phase
        
    '''
    pf = 0.
    for orbital, diffs in [(o1, o1_diffs), (o2, o2_diffs)]:
        for d in diffs:
            pf += np.searchsorted(configs[orbital], d)
    return (-1) ** pf