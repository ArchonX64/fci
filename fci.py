import itertools

import numpy as np
from numba import njit

from util import *
from slater import SlaterCondon
from diag import numpy_diag, david_diag

# -- PARAMETERS -------------------------------------
# Molecular System Params
electrons: int = 6
orbitals: int = 6

# Hartree-Fock Electron Energies
h1e: np.ndarray = np.load("resources/h1e.npy")
h2e: np.ndarray = np.load("resources/h2e.npy")
# --------------------------------------------------

configs = np.array(list(itertools.combinations(range(orbitals * 2), electrons))) + 1  # Each value represents the i'th electron, can't have zero

@njit
def process():
    '''Sets up 2-D hamiltonian matrix, which gives the energy for each determinant
    and their interaction energies. Solves for the energies of each element using
    Slater-Condon rules.
    '''
    
    sc = SlaterCondon(configs, h1e, h2e)
    
    h = np.zeros(shape=(len(configs), len(configs)))
    
    for i in range(len(configs)):
        for j in range(len(configs)):
            
            # Get orbitals that are different between elements
            o1_diffs, o2_diffs = oribtal_differences(configs[i], configs[j])
            diffs_len = len(o1_diffs)  # len(o1_diffs) == len(o2_diffs)
            
            # Calculate sign for element
            phase = get_phase(configs, i, o1_diffs, j, o2_diffs)
            
            match diffs_len:
                case 0:
                    h[i, j] = phase * (sc.e1_kk(i) + sc.e2_kk(i))
                case 1:
                    h[i, j] = phase * (sc.e1_kl1(o1_diffs[0], o2_diffs[0]) + sc.e2_kl1(i, o1_diffs[0], o2_diffs[0]))
                case 2:
                    h[i, j] = phase * sc.e2_kl2(o1_diffs[0], o1_diffs[1], o2_diffs[0], o2_diffs[1])
                case _:
                    pass
    
    print(np.min(numpy_diag(h)))
    print(np.min(david_diag(h, guess_vec_len=2, eig=1, tol=1e-9)))
    
process()