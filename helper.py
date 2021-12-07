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
from model import ConvNet,_ConvNet_

#This file contains all the functions used in Deep_solver.py



def prolongation_fn(grid_size):
    res_stencil = np.double(np.zeros((3,3)))
    k=16
    res_stencil[0,0] = 1/k
    res_stencil[0,1] = 2/k
    res_stencil[0,2] = 1/k
    res_stencil[1,0] = 2/k
    res_stencil[1,1] = 4/k
    res_stencil[1,2] = 2/k
    res_stencil[2,0] = 1/k
    res_stencil[2,1] = 2/k
    res_stencil[2,2] = 1/k
    P_stencils= np.zeros((grid_size//2,grid_size//2,3,3))
    for i in range(grid_size//2):
        for j in range(grid_size//2):
            P_stencils[i,j,:,:]=res_stencil
    return compute_p2(P_stencils, grid_size).astype(np.double)  # imaginary part should be zero


def diffusion_stencil_2d(epsilon=1.0, theta=0.0, type='FE'):
    eps = float(epsilon)  # for brevity
    theta = float(theta)

    C = np.cos(theta)
    S = np.sin(theta)
    CS = C*S
    CC = C**2
    SS = S**2

    if(type == 'FE'):
        a = (-1*eps - 1)*CC + (-1*eps - 1)*SS + (3*eps - 3)*CS
        b = (2*eps - 4)*CC + (-4*eps + 2)*SS
        c = (-1*eps - 1)*CC + (-1*eps - 1)*SS + (-3*eps + 3)*CS
        d = (-4*eps + 2)*CC + (2*eps - 4)*SS
        e = (8*eps + 8)*CC + (8*eps + 8)*SS

        stencil = np.array([[a, b, c],
                            [d, e, d],
                            [c, b, a]]) / 6.0

    elif type == 'FD':

        a = -0.5*(eps - 1)*CS
        b = -(eps*SS + CC)
        c = -a
        d = -(eps*CC + SS)
        e = 2.0*(eps + 1)

        stencil = np.array([[a+c, d-2*c, 2*c],
                            [b-2*c, e+4*c, b-2*c],
                            [2*c, d-2*c, a+c]])

        
    return stencil


def neural_smoother(nets,size,mixed = 0):
#     size = int(math.sqrt(size))
    if mixed==1:
        I = sp.sparse.identity(size*size)
        x0=I
        for net in nets.convLayers1:
            M = pyamg.gallery.stencil_grid(net.weight.view(3,3).detach().numpy(),(size,size),dtype=np.double,format='csr')
            x0 = M*x0
        return x0
    I = sp.sparse.identity(size*size)
    x0=I
    for net in nets.convLayers1:
        M = pyamg.gallery.stencil_grid(net.weight.view(3,3).detach().numpy(),(size,size),dtype=np.double,format='csr')
        x0 = M*x0
    M = pyamg.gallery.stencil_grid(nets.convLayers2[0].weight.view(3,3).detach().numpy(),(size,size),dtype=np.double,format='csr')
    y=x0+2/3*M
    
    return y 



def operator(N,jac,jac1,A,P,R):
    #return torch.mm(jac,torch.mm((torch.eye(N*N)-torch.mm(torch.mm(torch.mm(P,A),R),jac1)),jac))
    temp = torch.mm(jac,torch.mm((torch.eye(N*N)-torch.mm(torch.mm(torch.mm(P,A),R),jac1)),jac))
    return torch.mm((torch.eye(N*N)-temp),torch.inverse(jac1))
def pro_matrix(m,n):
    matrix = torch.zeros(m,n).double()
    for i in range(n):
        matrix[2*i,i]=1
        if 2*i+1<m:
            matrix[2*i+1,i]=2
        if 2*i+2<m:
            matrix[2*i+2,i]=1
    matrix=matrix/2
    return kronecker(matrix, matrix)
def kronecker(matrix1, matrix2):
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1)).double()


def get_neural_smoothers(args):
        ada_level1_no_jac = torch.load('./models/theta_'+str(args['theta'])+'_eps_'+str(args['eps'])+'/theta_'+str(args['theta'])+'_eps_'+str(args['eps'])+'_level1')
        ada_level2_no_jac = torch.load('./models/theta_'+str(args['theta'])+'_eps_'+str(args['eps'])+'/theta_'+str(args['theta'])+'_eps_'+str(args['eps'])+'_level2')
        ada_level3_no_jac = torch.load('./models/theta_'+str(args['theta'])+'_eps_'+str(args['eps'])+'/theta_'+str(args['theta'])+'_eps_'+str(args['eps'])+'_level3')
        ada_level4_no_jac = torch.load('./models/theta_'+str(args['theta'])+'_eps_'+str(args['eps'])+'/theta_'+str(args['theta'])+'_eps_'+str(args['eps'])+'_level4')
        no_jac_net1 = ConvNet(5,0)
        no_jac_net1.load_state_dict(ada_level1_no_jac)
        no_jac_net2 = ConvNet(5,0)
        no_jac_net2.load_state_dict(ada_level2_no_jac)
        no_jac_net3 = ConvNet(5,0)
        no_jac_net3.load_state_dict(ada_level3_no_jac)
        no_jac_net4 = ConvNet(5,0)
        no_jac_net4.load_state_dict(ada_level4_no_jac)
        for param in no_jac_net1.convLayers1.parameters():
            param.requires_grad = False    
        for param in no_jac_net2.convLayers1.parameters():
            param.requires_grad = False    
        for param in no_jac_net3.convLayers1.parameters():
            param.requires_grad = False    
        for param in no_jac_net4.convLayers1.parameters():
            param.requires_grad = False   
        ada_no_jac_nets = [no_jac_net1,no_jac_net2,no_jac_net3,no_jac_net4]
        return ada_no_jac_nets

def get_mixed_smoothers():
    nets1 = _ConvNet_(5)
    nets1.load_state_dict(torch.load('mixed_level_1'))
    nets2 = _ConvNet_(5)
    nets2.load_state_dict(torch.load('mixed_level_2'))
    nets3 = _ConvNet_(5)
    nets3.load_state_dict(torch.load('mixed_level_3'))
    nets4 = _ConvNet_(5)
    nets4.load_state_dict(torch.load('mixed_level_4'))
    net = [nets4,nets3,nets2,nets1]
    return net


def project_stencil(matrix_stencil):
    a11 = matrix_stencil[0,0]
    a12 = matrix_stencil[0,1]
    a13 = matrix_stencil[0,2]
    a21 = matrix_stencil[1,0]
    a22 = matrix_stencil[1,1]
    a23 = matrix_stencil[1,2]
    a31 = matrix_stencil[2,0]
    a32 = matrix_stencil[2,1]
    a33 = matrix_stencil[2,2]
    pro_stencil = torch.zeros(3,3).double()
    pro_stencil[0,0] = 1/16*a11+1/64*a12         +1/64*a21+1/256*a22
    pro_stencil[0,1] = 1/16*a11+6/64*a12+1/16*a13+1/64*a21+3/128*a22+1/64*a23
    pro_stencil[0,2] =          1/64*a12+1/16*a13+         1/256*a22+1/64*a23 
    pro_stencil[1,0] = 1/16*a11+1/64*a12+        +6/64*a21+3/128*a22         +1/16*a31+1/64*a32
    pro_stencil[1,1] = 1/16*a11+6/64*a12+1/16*a13+6/64*a21+ 9/64*a22+6/64*a23+1/16*a31+6/64*a32+1/16*a33 
    pro_stencil[1,2] =          1/64*a12+1/16*a13+        +3/128*a22+6/64*a23+        +1/64*a32+1/16*a33
    pro_stencil[2,0] =                           +1/64*a21+1/256*a22+        +1/16*a31+1/64*a32
    pro_stencil[2,1] =                           +1/64*a21+3/128*a22+1/64*a23+1/16*a31+6/64*a32+1/16*a33
    pro_stencil[2,2] =                                     1/256*a22+1/64*a23+        +1/64*a32+1/16*a33
    return pro_stencil*4

def get_CNN_smoothers():
    level = _ConvNet_(5)
    level.load_state_dict(torch.load('./models/level'))
    return level
def neural_smoother2(nets,size):
#     size = int(math.sqrt(size))
    I = sp.sparse.identity(size*size)
    x0=I
    for net in nets.convLayers:
        M = pyamg.gallery.stencil_grid(net.weight.view(3,3).detach().numpy(),(size,size),dtype=np.double,format='csr')
        x0 = M*x0
    y=x0
    return y
def map_2_to_1(grid_size=8):
    k = np.zeros((grid_size, grid_size, 3, 3))
    M = np.reshape(np.arange(grid_size ** 2), (grid_size, grid_size)).T
    M = np.concatenate([M, M], 0)
    M = np.concatenate([M, M], 1)
    for i in range(3):
        I = (i - 1) % grid_size
        for j in range(3):
            J = (j - 1) % grid_size
            k[:, :, i, j] = M[I:I + grid_size, J:J + grid_size]
    return k
def get_p_matrix_indices_one(grid_size):
    K = map_2_to_1(grid_size=grid_size)
    indices = []
    for ic in range(grid_size // 2):
        i = 2 * ic + 1
        for jc in range(grid_size // 2):
            j = 2 * jc + 1
            J = int(grid_size // 2 * jc + ic)
            for k in range(3):
                for m in range(3):
                    I = int(K[i, j, k, m])
                    indices.append([I, J])

    return np.array(indices)


def compute_p2(P_stencil, grid_size):
    indexes = get_p_matrix_indices_one(grid_size)
    P = csr_matrix(arg1=(P_stencil.reshape(-1), (indexes[:, 1], indexes[:, 0])),
                   shape=((grid_size//2) ** 2, (grid_size) ** 2))

    return P
