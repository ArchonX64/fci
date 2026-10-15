import numpy as np

def numpy_diag(h: np.ndarray) -> np.array:
    evals, evecs = np.linalg.eigh(h)
    return evals

def david_diag(h: np.ndarray, guess_vec_len: int, eig: int, tol: float) -> np.array:
    '''Davidson's algorithm implementation, which works quickly
       for diagnol-dominant matrix, such as FCI hamiltonians.
       
        Args:
            h (np.ndarray): 2-Dim Hamiltonian to be solved
            k (int): Number of initial guess vectors
            eig (int): Number of desired eigenvalues
            tol (float): Convergence tolerance
            
        Returns:
            np.array: The eigenvalues (we don't care about eigenvectors)'''
    
    n = len(h)
    unit_guess = np.eye(n, guess_vec_len)  # Set of k unit vectors as guesses
    guesses = np.zeros((n, n))  # Array of zeroes to hold guess vec
    iden = np.eye(n)  # Identity matrix
    diag_h = np.diag(h).copy()  # Diagonal of h, used by the preconditioner
    max_iter = n // 2  # Maximum number of iterations
    theta_old = np.ones(eig)
    
    for m in range(guess_vec_len, max_iter, guess_vec_len):
        if m <= guess_vec_len:  # Only runs on first try
            # Sets up the unit guess vectors into storage
            for j in range(0, guess_vec_len):
                guesses[:, j] = unit_guess[:, j] / np.linalg.norm(unit_guess[:, j])
        
        # Orthogonalize vectors
        guesses[:, :m], R = np.linalg.qr(guesses[:, :m])
        
        # Project into subspace, making a "shadow" of h.
        # Forms T = V^(T)AV
        v = np.ascontiguousarray(guesses[:, :m])  # Contiguous copy for faster np.dot
        projected = np.dot(v.T, np.dot(h, v))
        
        # Diagnolize small matrix
        eval, evec = np.linalg.eigh(projected)
        idx = eval.argsort()
        theta = eval[idx]
        evecs = evec[:, idx]
        
        # Compute correction vectors
        for j in range(0, guess_vec_len):
            # Calculate residual: How far h * vector deviates from eigenvalue
            w = np.dot((h - theta[j] * iden), np.dot(v, np.ascontiguousarray(evecs[:, j])))

            # Davidson preconditioner
            # Divides each residual component by (theta - h_ii), which gives us a pretty good correction
            denom = theta[j] - diag_h
            denom[np.abs(denom) < 1e-8] = 1e-8  # Avoid 0/0 where h_ii == theta
            q = w / denom
            
            # Apply conditioning factor
            guesses[:, (m + j)] = q
        norm = np.linalg.norm(theta[:eig] - theta_old)
        if norm < tol:
            break
        
        theta_old = theta[:eig].copy()
    return theta[:eig]
            
            
    