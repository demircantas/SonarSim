import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

def getParam_Sonar(Nx, Nz, Lx, Lz, UseSparseMatrices=True):
    """
    Defines the parameters for 2D acoustic wave equation for sonar propagation.
    Returns matrices for the linear system representation dx/dt = p.A x + p.B u
    where state x = [p_1, ..., p_N, v_1, ..., v_N]^T (pressure and velocity)

    INPUT:
    Nx         number of grid points in x direction
    Nz         number of grid points in z direction
    Lx         total length in x direction 
    Lz         total length in z direction 

    OUPUTS:
    p.A         system matrix (2Nx2N)
    p.B         input matrix (2Nx1)
    p.c         speed of sound
    p.rho       density of the medium
    p.alpha     absorption coefficient
    p.dx        spatial step in x direction
    p.dz        spatial step in z direction
    p.sonar_ix  sonar source grid index in x direction
    p.sonar_iz  sonar source grid index in z direction

    x_start     initial state vector
    t_start     initial time
    t_stop      simulation end time
    max_dt_FE   maximum stable timestep for Forward Euler

    EXAMPLE:
    [p,x_start,t_start,t_stop,max_dt_FE] = getParam_Sonar(Nx, Nz, Lx, Lz);
    """

    p = {
        'c': 1500.0,        # (m/s)
        'alpha': 0.001,       # (1/s) -- should be 0.001
        'Nx': Nx,           # (grid points)
        'Nz': Nz,           # (grid points)
        'Lx': Lx,           # domain size (m)
        'Lz': Lz,           # domain size (m)
        'sonar_ix': 5,      # (index)
        'sonar_iz': Nz//2   # (index)
    }

    n_phones = 5
    spacing = max(1, (Nx-10)//(n_phones-1))

    p['hydrophones'] = {
        'z_pos': Nz//2,  # fixed depth 
        'x_indices': [5 + i*spacing for i in range(n_phones)],  # horizontal array 
        'n_phones': n_phones
    }

    p['dx'] = Lx / (Nx - 1) # spatial discretiation
    p['dz'] = Lz / (Nz - 1)

    N = Nx * Nz # total grid points

    # build Lapacian matrix
    if UseSparseMatrices:
        L = sp.lil_matrix((N, N), dtype=float)
    else:
        L = np.zeros((N, N))

    # map 2D indices to 1D
    def idx(i, j):
        return i * Nz + j
    
    c2_dx2 = (p['c']**2) / (p['dx']**2)
    c2_dz2 = (p['c']**2) / (p['dz']**2)

    for i in range(Nx):
        for j in range(Nz):
            k = idx(i, j)

            # interior points
            if 1 <= i < Nx - 1 and 1 <= j < Nz - 1:
                L[k,k] = -2 * (c2_dx2 + c2_dz2)
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j-1)] = c2_dz2
                L[k, idx(i, j+1)] = c2_dz2

            # left boundary (x=0): absorbing 
            elif i == 0 and 1 <= j < Nz - 1:
                L[k, k] = -c2_dx2 - 2*c2_dz2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j-1)] = c2_dz2
                L[k, idx(i, j+1)] = c2_dz2

            # right boundary (x=Lx): absorbing
            elif i == Nx - 1 and 1 <= j < Nz - 1:
                L[k, k] = -c2_dx2 - 2*c2_dz2
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i, j-1)] = c2_dz2
                L[k, idx(i, j+1)] = c2_dz2

            # top boundary (z=0): pressure-release (sea surface)
            elif j == 0 and 1 <= i < Nx - 1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j+1)] = 2*c2_dz2  # pressure-release
           
            # bottom boundary (z=Lz): rigid (seafloor)
            elif j == Nz - 1 and 1 <= i < Nx - 1:
                L[k, k] = -2*c2_dx2 - c2_dz2
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j-1)] = c2_dz2

            # corners - only use available neighbors
            elif i == 0 and j == 0:
                L[k, k] = -c2_dx2 - 2*c2_dz2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j+1)] = 2*c2_dz2
            elif i == Nx-1 and j == 0:
                L[k, k] = -c2_dx2 - 2*c2_dz2
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i, j+1)] = 2*c2_dz2
            elif i == 0 and j == Nz-1:
                L[k, k] = -c2_dx2 - c2_dz2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j-1)] = c2_dz2
            elif i == Nx-1 and j == Nz-1:
                L[k, k] = -c2_dx2 - c2_dz2
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i, j-1)] = c2_dz2

    if UseSparseMatrices:
        p['A'] = sp.bmat([[sp.csr_matrix((N, N)), sp.eye(N)],
                          [L, -p['alpha']*sp.eye(N)]]).tocsr()
        p['B'] = sp.lil_matrix((2*N, 1), dtype=float)
    else:
        p['A'] = np.block([[np.zeros((N, N)), np.eye(N)],
                          [L, -p['alpha']*np.eye(N)]])
        p['B'] = np.zeros((2*N, 1))

    # source location
    source_idx = idx(p['sonar_ix'], p['sonar_iz'])
    p['B'][N + source_idx, 0] = 1.0 / (p['dx'] * p['dz'])

    # define also some example parameters for a typical transient simulation     
    x_start = np.zeros((2*N,1))
    x_start[:N] = np.random.randn(N, 1) * 1e-10

    t_start = 0
    t_cross = max(Lx, Lz) / p['c'] # time for sound to cross domain
    t_stop = 2.0 * t_cross # CHANGE THIS TO ADJUST SIMULATION TIME 

    # p['A'] not symmetric

    max_dt_FE = min(p['dx'], p['dz']) / (np.sqrt(2) * p['c']) * 0.5

    return p, x_start, t_start, t_stop, max_dt_FE
