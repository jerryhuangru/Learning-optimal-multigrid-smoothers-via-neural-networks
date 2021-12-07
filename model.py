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
class ConvNet(nn.Module):
    def __init__(self, nb_layers,initial):
        super(ConvNet, self).__init__()

        self.convLayers1 = nn.ModuleList([nn.Conv2d(1, 1, 3, padding=1, bias=False).double()
                                          for _ in range(5)])
        self.convLayers2 = nn.ModuleList([nn.Conv2d(1, 1, 3, padding=1, bias=False).double()
                                         for _ in range(1)])                         


    def forward(self, x):
        y1 = x
        for net in self.convLayers1:
            y1 = torch.relu(net(y1))
        y2 = x
        for net in self.convLayers2:
            y2 = torch.relu(net(y2))
        
        return y1+2/3*y2
    
    
class _ConvNet_(nn.Module):
    def __init__(self, nb_layers):
        super(_ConvNet_, self).__init__()

        self.convLayers = nn.ModuleList([nn.Conv2d(1, 1, 3, padding=1, bias=False).double()
                                         for _ in range(nb_layers)])


    def forward(self, x):
        for net in self.convLayers:
            x = torch.relu(net(x))
        return x