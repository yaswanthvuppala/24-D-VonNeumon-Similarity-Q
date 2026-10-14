import numpy as np

def build_intensity_density_matrix(neighborhood: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Method A: Intensity-only diagonal density matrix.
    
    IMPORTANT: We do NOT use a pure state |ψ⟩ = Σ sqrt(qi)|i⟩ followed by ρ = |ψ⟩⟨ψ| 
    because that produces a rank-1 pure state with S(ρ)=0 always, which provides no 
    information discrimination for keypoint detection.
    
    Args:
        neighborhood: Extracted neighborhood patch.
        epsilon: Small constant for stability.
        
    Returns:
        np.ndarray: 9x9 diagonal density matrix.
    """
    v = neighborhood.flatten()
    q = (v + epsilon) / np.sum(v + epsilon)
    return np.diag(q)

def build_covariance_density_matrix(feature_vectors: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Method B: Covariance-based density matrix.
    
    Args:
        feature_vectors: (N, d) array of feature vectors.
        epsilon: Stability constant.
        
    Returns:
        np.ndarray: dxd covariance-based density matrix.
    """
    N = feature_vectors.shape[0]
    mu = np.mean(feature_vectors, axis=0)
    C = (feature_vectors - mu).T @ (feature_vectors - mu) / N
    C += epsilon * np.eye(C.shape[0])
    return C / np.trace(C)

def verify_density_matrix(rho: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Verify that rho is a valid density matrix.
    
    Args:
        rho: Density matrix to verify.
        tol: Tolerance for numeric checks.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    is_hermitian = np.allclose(rho, rho.T.conj(), atol=tol)
    if not is_hermitian:
        print("Warning: Matrix is not Hermitian.")
        
    eigenvalues = np.linalg.eigvalsh(rho)
    is_psd = np.all(eigenvalues >= -tol)
    if not is_psd:
        print("Warning: Matrix is not positive semidefinite.")
        
    is_trace_one = np.abs(np.trace(rho) - 1.0) < tol
    if not is_trace_one:
        print("Warning: Matrix trace is not 1.")
        
    return is_hermitian and is_psd and is_trace_one
