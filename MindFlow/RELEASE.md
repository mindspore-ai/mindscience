# MindFlow Release Notes

## MindFlow 0.1.0a0 Release Notes

### Major Features and Improvements

#### Network Cells

- [STABLE] Support various networks, including fully connected networks, residual networks, Fourier neural operators, Vision Transformer (ViT), PDE-Net.

#### PDE

- [STABLE] Solve partial differential equations (PDE) based on physics informed neural networks (PINNs). Users can define PDE by sympy. Commenly used basic equations are also supported.

#### Geometry

- [STABLE] The definition of basic geometrics, time domains and their operations. Support sampling within the geometric domain and boundaries.

#### Data

- [STABLE] Support combining multiple datasets. Loading existed npy files are also supported.

#### Learning Rate

- [STABLE] Support polynomial decay laerning rate, warmup-cosine-annealing learning rate and multi-step learning rate.

#### Loss

- [STABLE] Support relative RMSE loss, multi-level wavelet loss and weighted multi-task loss.

#### Operators

- [STABLE] Calculate the Hessian and Jacobian matrix of network output to input.

#### Solver

- [STABLE] Train and evaluate neural networks.

#### CFD

- [STABLE] An end-to-end differentiable compressible computational fluid dynamics (CFD) solver, MindFlow-CFD, is introduced. WENO5 reconstruction, Rusanov flux, Runge-Kutta integrator are supported. Symmetry, periodic, non-slip wall and Neumann boundary conditions are supported.

## Contributors

Thanks goes to these wonderful people:

wangzidong, liuhongsheng, dengzhiwen, zhangyi, zhouhongye, libokai, yangge, liulei, longzichao

