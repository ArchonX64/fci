import itertools

import numpy as np
from numba import njit

# Define System
# Example: H2 -> 2 electrons 4 spin orbitals
electrons = 2
orbitals = 4

# Spin-orbital/electron combinations
n_values = np.array(list(itertools.combinations(range(orbitals), electrons))) + 1  # Each value represents the i'th electron, can't have zero
n = len(n_values)

# Load integrals
# 1-electon integrals: 2-dim (m, n) and 6 elements (electrons)
# 2-electron integrals: 4 dim (m, n, p, q) and 6 elements (electrons)
ei1 = np.load("resources/h1e.npy")
ei2 = np.load("resources/h2e.npy")

# Slater-Condon Rule Notation
# K and L are used to show whether the used wavefunctions are the same or different


# Write out Slate-Condon Rules
@njit
def e1_kk(m: int) -> float:
    e = 0.

    for mi in range(m):
        e += ei1[mi, mi]
    return e

@njit
def e2_kk(m: int, n: int, p: int, q: int) -> float:
    e = 0.

    for mi in range(m):
        for ni in range(n):
            e += ei2[mi, ni, mi, ni]
    return e

@njit
def e1_kl1(m: int, p: int) -> float:
    return ei1[m, p]

@njit
def e2_kl1(m: int, n: int, p: int, q: int) -> float:
    e = 0

    for ni in range(n):
        e += ei2[m, ni, p, ni]
    return e

@njit
def e1_kl2(m: int, n: int) -> float:
    return 0

@njit
def e2_kl2(m: int, n: int, p: int, q: int) -> float:
    return ei2[m, n, p, q]


# Set up matrix
@njit
def process():
    hamiltonian = np.zeros(shape=(n, n))

    # Set up 
    for i in range(n):
        for j in range(n):

            # KK case
            if i == j:
                hamiltonian[i, j] = e1_kk(mp[i])

            # 

process()
