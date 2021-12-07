import numpy as np
from warnings import warn
from scipy.sparse import csr_matrix, isspmatrix_csr, SparseEfficiencyWarning
import math
from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
import torch
import torch.nn as nn
import scipy as sp
import pyamg
import time
from helper import *

#This file contains the functions used to produce the results in the paper

def geometric_solver(A, prolongation_function,
                     presmoother=('gauss_seidel', {'sweep': 'forward'}),
                     postsmoother=('gauss_seidel', {'sweep': 'forward'}),
                     max_levels=5, max_coarse=10,coarse_solver='splu',**kwargs):
   
    levels = [multilevel_solver.level()]

    # convert A to csr
    if not isspmatrix_csr(A):
        try:
            A = csr_matrix(A)
            warn("Implicit conversion of A to CSR",
                 SparseEfficiencyWarning)
        except BaseException:
            raise TypeError('Argument A must have type csr_matrix, \
                             or be convertible to csr_matrix')
    # preprocess A
    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels[-1].A = A

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        extend_hierarchy(levels, prolongation_function)

    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


# internal function
def extend_hierarchy(levels, prolongation_fn):
    """Extend the multigrid hierarchy."""

    A = levels[-1].A

    N=A.shape[0]
    n = int(math.sqrt(N))
    R = prolongation_fn(n)


    P=R.T.tocsr()*4
    levels[-1].P = P  # prolongation operator
    levels[-1].R = R  # restriction operator

    levels.append(multilevel_solver.level())

    # Form next level through Galerkin product
    A = R * A * P
    A = A.astype(np.float64)  # convert from complex numbers, should have A.imag==0
    levels[-1].A = A
    
    
def multigrid_solver(A,size,args):
    if args['smoother'] == 'a-CNN':
        solver = geometric_solver(A, prolongation_fn,max_levels=5,coarse_solver='splu')
        ada_no_jac_nets = get_neural_smoothers(args)
        M0 = neural_smoother(ada_no_jac_nets[0],size)
        M1 = neural_smoother(ada_no_jac_nets[1],size//2)
        M2 = neural_smoother(ada_no_jac_nets[2],size//2//2)
        M3 = neural_smoother(ada_no_jac_nets[3],size//2//2//2)
        def new_relax0(A,x,b):
            x[:] += M0*(b-A*x)
        solver.levels[0].presmoother = new_relax0
        solver.levels[0].postsmoother = new_relax0
        def new_relax1(A,x,b):
            x[:] += M1*(b-A*x)
        solver.levels[1].presmoother = new_relax1
        solver.levels[1].postsmoother = new_relax1
        def new_relax2(A,x,b):
            x[:] += M2*(b-A*x)
        solver.levels[2].presmoother = new_relax2
        solver.levels[2].postsmoother = new_relax2
        def new_relax3(A,x,b):
            x[:] += M3*(b-A*x)
        solver.levels[3].presmoother = new_relax3
        solver.levels[3].postsmoother = new_relax3
    elif args['smoother'] == 'w-jacobi':
        solver = geometric_solver(A, prolongation_fn,max_levels=5,coarse_solver='splu')
        D0 = sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(solver.levels[0].A)).tocsr()
        D1 = sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(solver.levels[1].A)).tocsr()
        D2 = sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(solver.levels[2].A)).tocsr()
        D3 = sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(solver.levels[3].A)).tocsr()
        def new_relax0(A,x,b):
            x[:] += 2/3*D0*(b-A*x)
        solver.levels[0].presmoother = new_relax0
        solver.levels[0].postsmoother = new_relax0
        def new_relax1(A,x,b):
            x[:] += 2/3*D1*(b-A*x)
        solver.levels[1].presmoother = new_relax1
        solver.levels[1].postsmoother = new_relax1
        def new_relax2(A,x,b):
            x[:] += 2/3*D2*(b-A*x)
        solver.levels[2].presmoother = new_relax2
        solver.levels[2].postsmoother = new_relax2
        def new_relax3(A,x,b):
            x[:] += 2/3*D3*(b-A*x)
        solver.levels[3].presmoother = new_relax3
        solver.levels[3].postsmoother = new_relax3
    elif args['smoother'] == 'gs':
        solver = geometric_solver(A, prolongation_fn,max_levels=5,coarse_solver='splu')
        D0 = sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(solver.levels[0].A)).tocsr()
        D1 = sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(solver.levels[1].A)).tocsr()
        D2 = sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(solver.levels[2].A)).tocsr()
        D3 = sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(solver.levels[3].A)).tocsr()
        def new_relax0(A,x,b):
            x[:] += 2/3*D0*(b-A*x)
        solver.levels[0].presmoother = new_relax0
        solver.levels[0].postsmoother = new_relax0
        def new_relax1(A,x,b):
            x[:] += 2/3*D1*(b-A*x)
        solver.levels[1].presmoother = new_relax1
        solver.levels[1].postsmoother = new_relax1
        def new_relax2(A,x,b):
            x[:] += 2/3*D2*(b-A*x)
        solver.levels[2].presmoother = new_relax2
        solver.levels[2].postsmoother = new_relax2
        def new_relax3(A,x,b):
            x[:] += 2/3*D3*(b-A*x)
        solver.levels[3].presmoother = new_relax3
        solver.levels[3].postsmoother = new_relax3
    elif args['smoother']=='mixed':
        solver = geometric_solver(A, prolongation_fn,max_levels=5,coarse_solver='splu')
        mixed_nets = get_mixed_smoothers()
        M0 = neural_smoother(mixed_nets[0],size,mixed = 1)
        M1 = neural_smoother(mixed_nets[1],size//2,mixed = 1)
        M2 = neural_smoother(mixed_nets[2],size//2//2,mixed = 1)
        M3 = neural_smoother(mixed_nets[3],size//2//2//2,mixed = 1)
        def new_relax0(A,x,b):
            x[:] += M0*(b-A*x)
        solver.levels[0].presmoother = new_relax0
        solver.levels[0].postsmoother = new_relax0
        def new_relax1(A,x,b):
            x[:] += M1*(b-A*x)
        solver.levels[1].presmoother = new_relax1
        solver.levels[1].postsmoother = new_relax1
        def new_relax2(A,x,b):
            x[:] += M2*(b-A*x)
        solver.levels[2].presmoother = new_relax2
        solver.levels[2].postsmoother = new_relax2
        def new_relax3(A,x,b):
            x[:] += M3*(b-A*x)
        solver.levels[3].presmoother = new_relax3
        solver.levels[3].postsmoother = new_relax3
    return solver


# def compute_spectral_radius(stencil,N,args):
#     if args['smoother'] == 'a-CNN':
#         ada_no_jac_nets = get_neural_smoothers(args)
#     P=pro_matrix(N,N//2)
#     R=P.T/4
#     level = 5
#     mat_sten = torch.from_numpy(stencil)
#     A = pyamg.gallery.stencil_grid(stencil,(N,N),format='csr').toarray()
#     A = torch.from_numpy(A)
#     matrix_stencils = []
#     for i in range(level-1):
#         matrix_stencils = [mat_sten]+matrix_stencils
#         mat_sten = project_stencil(mat_sten)
#     NN = int((N+1)/2**(level-1)-1)
#     A = pyamg.gallery.stencil_grid(mat_sten,(NN,NN),format='csr').toarray()
#     A = torch.from_numpy(A)
#     A = torch.inverse(A)
#     M=A
#     n=int((N+1)/2**(level-2)-1)

#     for i in range(level-1):
#         jac1 = pyamg.gallery.stencil_grid(matrix_stencils[i].numpy(),(n,n),format='csr').toarray()
#         jac1 = torch.from_numpy(jac1).double()
#         D = torch.diag(torch.diag(jac1))
#         D = torch.inverse(D)
#         if args['smoother'] =='w-jacobi':
#             jac = torch.eye(n*n)-2/3*jac1/matrix_stencils[i][1,1]
#         elif args['smoother'] == 'a-CNN':
#             jac = torch.eye(n*n).double()-torch.mm(torch.from_numpy(neural_smoother(ada_no_jac_nets[level-2-i],n).toarray()),jac1)
#         P=pro_matrix(n,n//2)
#         R=P.T/4
#         M=operator(n,jac,jac1,M,P,R)
#         n=n*2+1
#     M = torch.eye(N*N)-torch.mm(M,jac1)
#     return torch.max(torch.eig(M)[0])
def compute_bound_E_helper(M,A):
    MM= M.T+M-A
    MM=np.linalg.inv(MM)
    M=np.matmul(M.T,np.matmul(MM,M))
    M=np.linalg.inv(M)
    M=np.matmul(M,A)
    w,v = np.linalg.eig(M)
    return math.sqrt(1-min(w))

def compute_bound_E(stencil,size,args):
    A = pyamg.gallery.stencil_grid(stencil,(size,size),format='csr')
    if args['smoother'] == 'a-CNN':
        ada_no_jac_nets = get_neural_smoothers(args)
        M = neural_smoother(ada_no_jac_nets[0],size).toarray()
        M = np.linalg.inv(M)
        return compute_bound_E_helper(M,A.toarray())
    elif args['smoother'] == 'w-jacobi':
        D = 2/3*sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(A)).toarray()
        D = np.linalg.inv(D)
        return compute_bound_E_helper(D,A.toarray())
    elif args['smoother'] == 'gs':
        L = sp.sparse.tril(A.toarray()).toarray()
#         L = np.linalg.inv(L)
        return compute_bound_E_helper(L,A.toarray())

def compute_spectral(stencil,size,args):
    A = pyamg.gallery.stencil_grid(stencil,(size,size),format='csr')
    I = np.identity(size*size)
    if args['smoother'] == 'a-CNN':
        ada_no_jac_nets = get_neural_smoothers(args)
        M = neural_smoother(ada_no_jac_nets[0],size).toarray()
        G = I-np.matmul(M,A.toarray())
    elif args['smoother'] == 'w-jacobi':
        D = 2/3*sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(A)).toarray()
        G = I-np.matmul(D,A.toarray())
    elif args['smoother'] == 'gs':
        L = sp.sparse.tril(A.toarray()).toarray()
        L = np.linalg.inv(L)
        G = I-np.matmul(L,A.toarray())
    return max(np.linalg.eig(G)[0])

def compute_damping_factor(A,n):
    vector = np.linalg.eig(A.toarray())
    ada_no_jac_nets = get_neural_smoothers({'smoother':'a-CNN','eps':100,'theta':75})
    net = get_CNN_smoothers()
    M1 = neural_smoother(ada_no_jac_nets[0],n)
    M0 = neural_smoother2(net,n)
    D0 = sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(A)).tocsr()
    L = sp.sparse.tril(A.toarray())
    L = np.linalg.inv(L.toarray())
    idx = vector[0].argsort()[::-1]   
    eigenValues = vector[0][idx]
    eigenVectors = vector[1][:,idx]
    v = eigenVectors
    original = []
    w_jacobi = []
    CNN = []
    a_CNN = []
    gs = []
    t = []
    for i in range(n*n):
        if i%10==0:
            vv = v[:,i]
            t.append(i+1)
            damping_vector0 = vv-2/3*D0*(A*vv)
            damping_vector1 = damping_vector0-2/3*M0*(D0*(A*vv))
            damping_vector2 = vv-M1*(A*vv)
            damping_vector3 = vv-np.matmul(L,(A*vv))
            original.append(np.linalg.norm(vv))
            w_jacobi.append(np.linalg.norm(damping_vector0))
            CNN.append(np.linalg.norm(damping_vector1))
            a_CNN.append(np.linalg.norm(damping_vector2))
            gs.append(np.linalg.norm(damping_vector3))
    return t,w_jacobi,CNN,a_CNN,gs
    
def solve_systems(A,solver):
    m=A.shape[0]
    num_test = 10
    b = np.random.rand(A.shape[0],num_test)
    x0 = np.ones((m,1))
    t1=time.time()
    num_iter = []
    for i in range(num_test):
        res=[]
        x = solver.solve(b[:,i],x0=x0,maxiter=1000, tol=1e-6,residuals=res)
        num_iter.append(len(res))
    print(res[-1])
    t2=time.time()
    if res[-1]>1e-2 or np.isnan(res[-1]):
        print('fail to converge')
    return np.mean(num_iter),(t2-t1)/num_test
