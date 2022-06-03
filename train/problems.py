import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import scipy
device='cpu'

class Problem:

    def __init__(self,
                 f=None,
                 k=20,
                 k_ground_truth=1000,
                 initial_ground_truth=None,
                 initial_u=None,
                 levels=0,
                 net_trained=0,
                 mxl=0
                ):
        N = levels[0]['N']
        l = levels[0]['l']
        if f is None:
            f = torch.zeros((l,1))
        # Initialize parameters to compute ground truth solution
        if initial_ground_truth is None:
            self.ground_truth = torch.rand(l,1).double().to(device)
        else:
            self.ground_truth = initial_ground_truth
        

        if initial_u is None:
            self.initial_u = torch.rand(l,1).double().to(device)
        else:
            self.initial_u = initial_u

        self.k = k
        self.N = N

        self.levels=levels
        self.mxl = mxl
        self.net_trained = net_trained
        if net_trained!=0:
            for net in self.net_trained:
                for param in net.parameters():
                    param.requires_grad = False    
        A = self.levels[0]['A']
        self.f = torch.sparse.mm(A,self.ground_truth)
    def compute_solution(self, nets):
        u_H = Solver_red_black(nets,self.net_trained,self.levels,self.f,self.initial_u,self.mxl,self.k)
        return u_H

def Solver_red_black(net,nets_trained,levels,f,initial_u,mxl,k):
    def step(res,net):
        _u_n = net(res)
        return _u_n
    u = initial_u
    if nets_trained!=0:
        nets = [net]+nets_trained
    else:
        nets = [net]
    for i in range(k):
        u = V_cycle_red_black(levels,f,u,0,mxl-1,step,nets)
    return u

def V_cycle_red_black(levels,f,initial_u,lvl,mxl,step,nets):
    u = initial_u
    level = levels[lvl]
    A = level['A']
    N = level['N']
    if lvl==mxl:
        return torch.mm(torch.inverse(A.to_dense()),f)
    P = level['P']
    R = level['R']
    if level['square']:
        r = f-torch.sparse.mm(A,u)
        u = u+step(r.view(1,1,N,N),nets[lvl]).view(-1,1)
    else:
        idx_x,idx_y,idx_a,idx_b = level['rotate_idx']
        r = f-torch.sparse.mm(A,u)
        rr = torch.zeros(N*N,1).double()
        rr[0:N*N:2,:] = r.clone()    
        rr = rr.view(N,N)
        uu = torch.zeros(N,N).double()
        uu[idx_x,idx_y] = rr[idx_a,idx_b]
        uu = step(uu.view(1,1,N,N),nets[lvl]).view(N,N)
        BB = torch.zeros(N,N).double()
        BB[idx_a,idx_b] = uu[idx_x,idx_y]
        BB = BB.view(-1,1)
        u = u+BB[0:N*N:2,:]
    r = f-torch.sparse.mm(A,u)
    r = torch.sparse.mm(R,r)

    u0 = torch.zeros(r.shape).double().to(device)
    delta = V_cycle_red_black(levels,r,u0,lvl+1,mxl,step,nets)
    u = torch.sparse.mm(P,delta)+u
    if level['square']:
        r = f-torch.sparse.mm(A,u)
        u = u+step(r.view(1,1,N,N),nets[lvl]).view(-1,1)
    else:
        idx_x,idx_y,idx_a,idx_b = level['rotate_idx']
        r = f-torch.sparse.mm(A,u)
        rr = torch.zeros(N*N,1).double()
        rr[0:N*N:2,:] = r.clone()    
        rr = rr.view(N,N)
        uu = torch.zeros(N,N).double()
        uu[idx_x,idx_y] = rr[idx_a,idx_b]
        uu = step(uu.view(1,1,N,N),nets[lvl]).view(N,N)
        BB = torch.zeros(N,N).double()
        BB[idx_a,idx_b] = uu[idx_x,idx_y]
        BB = BB.view(-1,1)
        u = u+BB[0:N*N:2,:]
    return u
