# Learning optimal multigrid smoothers via neural networks

by Ru Huang, Ruipeng Li, Yuanzhe Xi

This paper has been accepted by SIAM Journal on Scientific Computing (SISC) which be found on https://arxiv.org/abs/2102.12071.
>Implementation of training a-CNN smoothers in Python. It is an efficent adaptive learning algorithm for learning smoothers formed by convolutional neural networks. 
<img src="https://github.com/jerryhuangru/Learning-optimal-multigrid-smoothers-via-neural-networks/blob/main/figures/framework.png" width = 400>

# Abstract

Multigrid methods are one of the most efficient techniques for solving large sparse linear systems arising from Partial Differential Equations (PDEs) and graph Laplacians from machine learning applications. One of the key components of multigrid is smoothing, which aims at reducing high-frequency errors on each grid level. However, finding optimal smoothing algorithms is problem-dependent and can impose challenges for many problems. In this paper, we propose an efficient adaptive framework for learning optimized smoothers from operator stencils in the form of convolutional neural networks (CNNs). The CNNs are trained on small-scale problems from a given type of PDEs based on a supervised loss function derived from multigrid convergence theories, and can be applied to large-scale problems of the same class of PDEs. Numerical results on anisotropic rotated Laplacian problems and variable coefficient diffusion problems demonstrate improved convergence rates and solution time compared with classical hand-crafted relaxation methods.


# Requirements

* pytorch

* pyamg

# Software implementation

# Experiments

# Citations

Please cite our paper if you use this code in your own work:
```
@article{huang2021learning,
  title={Learning optimal multigrid smoothers via neural networks},
  author={Huang, Ru and Li, Ruipeng and Xi, Yuanzhe},
  journal={arXiv preprint arXiv:2102.12071},
  year={2021}
}
