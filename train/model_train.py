import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import time
import math
import scipy.sparse as sp
import scipy
from torch.autograd import Variable



device = 'cpu'

class _ConvNet_(nn.Module):
    def __init__(self,k,kernel_size,initial_kernel):
        super(_ConvNet_, self).__init__()

        self.convLayers1 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False).double()
                                          for _ in range(5)])
        self.convLayers2 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False).double()
                                         for _ in range(2)])                         
        initial_weights = torch.zeros(1,1,kernel_size,kernel_size).double()
        initial_weights[0,0,kernel_size//2,kernel_size//2] = initial_kernel

        for net in self.convLayers1:
            net.weight = nn.Parameter(initial_weights)
        for net in self.convLayers2:
            net.weight = nn.Parameter(initial_weights)

    def forward(self, x):
        y1 = x
        y2 = x
        for net in self.convLayers1:
            y1 = torch.tanh(net(y1))

        for net in self.convLayers2:
            y2 = torch.tanh(net(y2))
        
        return y1+2/3*y2

def compute_loss(net,problem_instances):

    loss = torch.zeros(1).to(device)

    for problem_instance in problem_instances:
        # Compute solution
        u = problem_instance.compute_solution(net)
        loss += torch.norm(u-problem_instance.ground_truth)

    return loss

class alphaCNN:

    def __init__(self,
                 nets=None,
                 batch_size=1,
                 learning_rate=1e-6,
                 max_epochs=1000,
                 nb_layers=3,
                 tol=1e-6,
                 stable_count=50,
                 N=16,
                 optimizer='SGD',
                 check_spectral_radius=False,
                 random_seed=None,initial_nets = None,initial = 5,kernel_size=3,initial_kernel=0.1):

        if random_seed is not None:
            set_seed(random_seed)

        if nets is None:
            self.nb_layers = nb_layers
            self.net = _ConvNet_(initial,kernel_size,initial_kernel).to(device)
        else:
            self.net = net

        self.learning_rate = learning_rate
        if optimizer == 'Adadelta':
            self.optim = torch.optim.Adadelta(list(self.net.parameters()),lr=1)
        elif optimizer == 'Adam':
            self.optim = torch.optim.Adam(list(self.net.parameters()),lr=learning_rate)
        else:
            self.optim = torch.optim.SGD(
                self.net.parameters(), lr=learning_rate)

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tol = tol
        self.stable_count = stable_count

        # self.T = helpers.build_T(N)
        # self.H = None
        # self.N = N

    def _optimization_step_(self, problem_instances):

        shuffled_problem_instances = np.random.permutation(problem_instances)

        for problem_chunk in chunks(shuffled_problem_instances, self.batch_size):
            self.net.zero_grad()
            t1= time.time()
            loss = compute_loss(self.net, problem_chunk)
            t2 =time.time()
            loss.backward(retain_graph=True)
            t3 = time.time()
            self.optim.step()

    def fit(self, problem_instances):
        # Initialization
        losses = []
        prev_total_loss = compute_loss(self.net, problem_instances).item()
        convergence_counter = 0
        for n_epoch in range(self.max_epochs):

            self._optimization_step_(problem_instances)

            total_loss = compute_loss(
                self.net, problem_instances).item()

            losses.append(total_loss)

            if np.abs(total_loss - prev_total_loss) < self.tol:
                convergence_counter += 1
                if convergence_counter > self.stable_count:
                    break
            else:
                convergence_counter = 0

            prev_total_loss = total_loss

            if n_epoch % 100 == 0:
                print('Epoch: '+str(n_epoch) +' total loss '+str(prev_total_loss))

        self.losses = losses
        print(str(n_epoch)+ ' total loss: '+str(total_loss))

        return self



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


