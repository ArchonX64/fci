import itertools

import numpy as np

from slater import SlaterCondon


# -- PARAMETERS -------------------------------------
# Molecular System Params
electrons: int = 6
orbitals: int = 6

# Hartree-Fock Electron Energies
h1e: np.ndarray = np.load("resources/h1e.npy")
h2e: np.ndarray = np.load("resources/h2e.npy")
# --------------------------------------------------

configs = np.array(list(itertools.combinations(range(orbitals * 2), electrons))) + 1  # Each value represents the i'th electron, can't have zero

sc = SlaterCondon(configs, h1e, h2e)

