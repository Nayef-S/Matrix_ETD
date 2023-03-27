import torch

def matrix_mult_w_identity_pad(A, B):
    """
    A is the large mxm matrix 
    B is the smaller nxn matrix that is padded to be mxm using the identity matrix
    """
    small_A = A[:, :B.shape[0]]
    prod_small_A_B = torch.matmul(small_A, B)

    # Create the final product matrix by combining the result of the multiplication
    # and the remaining columns of A
    prod = torch.zeros_like(A)
    prod[:, :B.shape[0]] = prod_small_A_B
    prod[:, B.shape[0]:] = A[:, B.shape[0]:]

    return prod



def ETD1_BCH1(L,R,dt):
    """
    Returns the operators for matrix ETD1
    Parameters
    ----------
    L : (M, M) array_like
        LHS operator acting on Q
        
    R : (N, N) array_like
       RHS operator acting on Q
        
    dt : float
        Time step 

    Returns
    -------
    m1_L : (M, M) array_like
        M1[L] acting on LHS of Q_n.
        
    m1_R : (N, N) array_like
        M1[R] acting on RHS of Q_n.
        
    m2 : (M, M) array_like
        M2[L+R] acting on the non-linear operator.
    """
    assert L.shape[0] >= R.shape[0], "L matrix should have equal or larger dimensions than R matrix."

    m1_L = torch.linalg.matrix_exp(dt*L) # matrix exp for mxm matrix
    m1_R = torch.linalg.matrix_exp(dt*R) # matrix exp for nxn matrix 

    R_padded = torch.zeros_like(L).cuda()
    R_padded[:R.shape[0], :R.shape[1]] = R # matrix exp pads with identity 

    prod = matrix_mult_w_identity_pad(m1_L, m1_R) # compute m1_L @ padded m1_R
    
    m2_b =  prod - torch.eye(L.shape[0]).cuda()

    regularised_L = L + 1e-6 * torch.eye(L.shape[0], dtype=L.dtype).cuda() # regularisation

    # eigenvalues, eigenvectors = torch.linalg.eigh(regularised_L)

    # smallest_eigenvalue = eigenvalues[0]
    # largest_eigenvalue = eigenvalues[-1]

                 
    m2 = torch.linalg.solve((regularised_L+R_padded), m2_b)
    
    return m1_L, m1_R, m2


def ETD1_BCH2(L, R, dt):
    """
    Returns the operators for matrix ETD1 using second-order BCH formula when L and R have different sizes.

    Parameters
    ----------
    L : (M, M) array_like
        LHS operator acting on Q.
        
    R : (N, N) array_like
        RHS operator acting on Q. N <= M
       
    dt : float
        Time step.

    Returns
    -------
    m1_L : (M, M) array_like
        M1[L] acting on LHS of Q_n.
        
    m1_R : (M, M) array_like
        M1[R] acting on RHS of Q_n.
        
    m2 : (M, M) array_like
        M2[L+R] acting on the non-linear operator.
    """
    assert L.shape[0] >= R.shape[0], "L matrix should have equal or larger dimensions than R matrix."
    
    R_padded = torch.zeros_like(L).cuda()
    R_padded[:R.shape[0], :R.shape[1]] = R

    commutator = L @ R_padded - R_padded @ L

    m1_L = torch.linalg.matrix_exp(dt * L)
    m1_R = torch.linalg.matrix_exp(dt * R)

    L_plus_R = L + R_padded + 0.5 * dt * commutator + 1e-6 * torch.eye(L.shape[0], dtype=L.dtype).cuda() # regularisation

    m2_b = torch.linalg.matrix_exp(dt * L_plus_R) - torch.eye(L.shape[0]).cuda()

    m2 = torch.linalg.solve(L_plus_R, m2_b)
    
    return m1_L, m1_R, m2


def ETD2(L,R,N,dt):
    """
    
    Parameters
    ----------
    L : (M, M) array_like
        LHS operator acting on Q
        
    R : (M, M) array_like
       RHS operator acting on Q
       
    N : (M, N) array_like
        Non-linear operator acting on Q
        
    dt : float
        Time step
    Returns
    -------
    m1_L : (M, M) array_like
        M1[L] acting on LHS of Q_n
    m1_R : (M, M) array_like
        M1[R] acting on RHS of Q_n
    m2 : TYPE
        DESCRIPTION.
    m3 : TYPE
        DESCRIPTION.
    """

    m1_L = torch.linalg.matrix_exp(dt*L) 
    m1_R = torch.linalg.matrix_exp(dt*R) 

    breakpoint()
    
    m_lr = m1_L @ m1_R
    
    m2 = (1/dt)* torch.linalg.matrix_power((L+R),-2) @ (  (dt*(L+R)+torch.eye(L.shape[0]) ) @ (m_lr) - 2*dt*(L+R) -   torch.eye(L.shape[0]) )  
    
    m3 = (1/dt)* torch.linalg.matrix_power((L+R),-2) @ ( torch.eye(L.shape[0]) + dt*(L+R)  -    (m_lr)   )
    
    return m1_L , m1_R , m2 , m3 