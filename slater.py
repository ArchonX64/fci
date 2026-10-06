import itertools

import numpy as np

from numba import njit

# Define System
# Example: H2
#   2 electrons
#   2 orbitals * 2 spins = 4 spin orbitals
electrons = 6
orbitals = 6

# Orbital/electron combinations
# Each top element is a possible configuration, inner element is what orbital each are in
# [[1 2] [1 3] [1 4]]
configs = np.array(list(itertools.combinations(range(orbitals * 2), electrons))) + 1  # Each value represents the i'th electron, can't have zero

# We are treating 2 and 1 for example as the same orbital but with different spin
@njit
# Spin orbitals are 1-indexed (1, 2 -> spatial 0; 3, 4 -> spatial 1; ...) to match the integral arrays
def no_spin(orbital: int) -> int:
    return (orbital - 1) // 2

# Odd spin orbitals are alpha (0), even are beta (1)
@njit
def spin(orbital: int) -> int:
    return (orbital - 1) % 2

# Checking how many changes there are between two configurations
@njit
def oribtal_differences(o1: np.array, o2: np.array) -> list[np.array, np.array]:
    o1_diffs = np.setdiff1d(o1, o2)
    o2_diffs = np.setdiff1d(o2, o1)
    return o1_diffs, o2_diffs
            

# Load integrals
# 1-electon integrals: 2-dim (m, n) and 6 elements (electrons)
# 2-electron integrals: 4 dim (m, n, p, q) and 6 elements (electrons)
ei1 = np.load("resources/h1e.npy")
ei2 = np.load("resources/h2e.npy")

# Slater-Condon Rule Notation
# K and L are used to show whether the used wavefunctions are the same or different

# Write out Slate-Condon Rules
# ei2 is in chemist's notation: (pq|rs) = <pr|qs>, so <mn|pq> = ei2[m, p, n, q]
# Each integral is zero unless the spins it pairs up match
@njit
def e1_kk(m: int) -> float:
    m = no_spin(m)
    return ei1[m, m]

# <mn||mn> = (mm|nn) - (mn|nm)
@njit
def e2_kk(m: int, n: int) -> float:
    sm, sn = no_spin(m), no_spin(n)
    coulomb = ei2[sm, sm, sn, sn]
    exchange = ei2[sm, sn, sn, sm] if spin(m) == spin(n) else 0.0
    return coulomb - exchange

# <m|h|p>
@njit
def e1_kl1(m: int, p: int) -> float:
    if spin(m) != spin(p):
        return 0.0
    return ei1[no_spin(m), no_spin(p)]

# <mn||pn> = (mp|nn) - (mn|np)
@njit
def e2_kl1(m: int, n: int, p: int) -> float:
    if spin(m) != spin(p):
        return 0.0
    sm, sn, sp = no_spin(m), no_spin(n), no_spin(p)
    coulomb = ei2[sm, sp, sn, sn]
    exchange = ei2[sm, sn, sn, sp] if spin(m) == spin(n) else 0.0
    return coulomb - exchange

# <mn||pq> = (mp|nq) - (mq|np), with m, n occupied in K and p, q occupied in L
@njit
def e2_kl2(m: int, n: int, p: int, q: int) -> float:
    sm, sn, sp, sq = no_spin(m), no_spin(n), no_spin(p), no_spin(q)
    coulomb = ei2[sm, sp, sn, sq] if spin(m) == spin(p) and spin(n) == spin(q) else 0.0
    exchange = ei2[sm, sq, sn, sp] if spin(m) == spin(q) and spin(n) == spin(p) else 0.0
    return coulomb - exchange


# Set up matrix
@njit
def process():
    hamiltonian = np.zeros(shape=(len(configs), len(configs)))

    # Set up iteration through matrix elements
    for i in range(len(configs)):
        for j in range(len(configs)):

            o1_diffs, o2_diffs = oribtal_differences(configs[i], configs[j])

            # KK case (diagnal)
            if len(o1_diffs) == 0:
                # 1 electron interactions
                e1 = 0.0
                for m in configs[i, :]:
                    e1 += e1_kk(m)

                # 2-electron interactions
                e2 = 0.0
                for m in configs[i, :]:
                    for n in configs[i, :]:
                        e2 += e2_kk(m, n)
                hamiltonian[i, j] = e1 + 0.5 * e2

            # KL 1-step case
            elif len(o1_diffs) == 1:
                # 1-electron interactions
                e1 = e1_kl1(o1_diffs[0], o2_diffs[0])

                # 2-electron interactions
                e2 = 0.0
                m = o1_diffs[0]
                p = o2_diffs[0]
                for n in configs[i, :]:
                    e2 += e2_kl1(m, n, p)

                # Determine sign
                sign = (-1) ** (np.searchsorted(configs[i], m) + np.searchsorted(configs[j], p))
                hamiltonian[i, j] = sign * (e1 + e2)

            # KL 2-step case
            elif len(o1_diffs) == 2:
                # 2-electron interactions
                e2 = e2_kl2(o1_diffs[0], o1_diffs[1], o2_diffs[0], o2_diffs[1])

                # Sign
                sign = (-1) ** (np.searchsorted(configs[i], o1_diffs[0]) + np.searchsorted(configs[i], o1_diffs[1])
                        + np.searchsorted(configs[j], o2_diffs[0]) + np.searchsorted(configs[j], o2_diffs[1]))
                hamiltonian[i, j] = sign * e2

            # Anything past has no interaction
            else:
                pass

    # Diagnolize matrix to find eigenvalues
    evals, evecs = np.linalg.eigh(hamiltonian)

    print(np.min(evals))


process()
