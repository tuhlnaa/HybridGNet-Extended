import scipy.sparse as sp

from utils.utils import scipy_to_torch_sparse, genMatrixesLH

def load_sparse_matrices(device: str):
    """Load and convert sparse matrices for model initialization."""
    A, AD, D, U = genMatrixesLH()  # Assuming this function exists in utils
    
    matrices = {
        'A': sp.csc_matrix(A).tocoo(),
        'AD': sp.csc_matrix(AD).tocoo(),
        'D': sp.csc_matrix(D).tocoo(),
        'U': sp.csc_matrix(U).tocoo()
    }
    
    D_ = [matrices['D'].copy()]
    U_ = [matrices['U'].copy()]
    A_ = [matrices['A'].copy()] * 3 + [matrices['AD'].copy()] * 3
    
    return tuple([scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_))