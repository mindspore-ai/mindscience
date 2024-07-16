# Dataset Files

## Pretraining Dataset

* Equation Form:

$$
\begin{split}
u_t+f_0(u)+s(x)+(f_1(u)-\kappa(x)u_x)_x&=0 , \quad (t,x) \in [0,1] \times [-1,1], \\
u(0,x) &= g(x), \quad x \in [-1,1]
\end{split}
$$

where $f_i(u) = c_{i1}u+c_{i2}u^2+c_{i3}u^3$, $i=0,1$.

* Boundary Conditions: Periodic + Non-periodic. For the non-periodic case, it is randomly chosen from Dirichlet, Neumann, Robin independently on the left and right boundaries.
* Download link：[PKU Disk](https://disk.pku.edu.cn/anyshare/en-us/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/434EE9473D90449A8B1E4847065BCA89)
* Filename format: `custom_v4.23_sinus0_[#bc]_c[#c]_k[#k]_seed[#r].hdf5`
    * For example: `custom_v4.23_sinus0_robin_cU3_k1e-03_1_seed1.hdf5`
    * `[#bc]`: Boundary condition type (`circ` for periodic, `robin` for non-periodic including Dirichlet, Neumann and Robin).
    * `[#c]`: Non-zero coefficient distribution (default: `U3` for $[-3,3]$).
    * `[#k]`: Range of Positive Coefficient $\kappa(x)$ (default: `k1e-03` for $[10^{-3},1]$).
    * `[#r]`: Random seed, with positive integer value.
* Data generation code: [../data_generation/custom_sinus.py](../data_generation/custom_sinus.py).

## Inverse Problem Dataset with Same Data Distribution as the Pretraining Set

* The equation form, boundary condition are same as the pretraining dataset.
    Every data file contains 100 PDEs, and every PDE has 100 samples. Different samples have different initial conditions $g(x)$.
* Download Link: [PKU Disk](https://disk.pku.edu.cn/anyshare/en-us/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/CDBCBEF0F0D4459C893F3CBBE62F521E)

* File Name Format: `custom_v4.23_inv_sinus0_[#bc]_f[#f]_c[#c]_k[#k]_seed[#r].hdf5`
    * For example：`custom_v4.23_inv_sinus0_circ_fS_cU3_k1e-03_1_seed1.hdf5`
    * `[#f]`: In the form of the PDE, $s(x)$ and $\kappa(x)$ are terms that specify whether these two are spatially dependent (i.e., a coefficient field). For example, `S` indicates that $s(x)$ in all equations within the dataset is a coefficient field, while $\kappa(x)=\kappa$ is a scalar coefficient. `SK` indicates that both are coefficient fields. Leaving it empty indicates that both are scalar coefficients.
    * The meanings of the remaining variables are the same as in the pre-trained dataset.
* Data generation code: [../data_generation/inverse_sinus.py](../data_generation/inverse_sinus.py).

## Dataset with Trigonometric Function Terms

* Equation form and boundary conditions: Similar to the pre-training dataset, the only difference being the introduction of trigonometric function terms
    $f_i(u) = \sum_{k=1}^3c_{i0k}u^k + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$，$i=0,1$,
    where $h_{ij}\in\{\sin,\cos\}$ is chosen with equal probability, $J_0+J_1=J$, and $J_0\in\{0,1,\dots,J\}$ is randomly chosen.
* Download Link: [PKU Disk](https://disk.pku.edu.cn/anyshare/en-us/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/F682DE84519241489C2F6822476A8DFD)
* File name format：`custom_v4.23_sinus[#J]_[#bc]_c[#c]_k[#k]_seed[#r].hdf5`
    * For example：`custom_v4.23_sinus4_circ_cU3_k1e-03_1_seed1.hdf5`
    * `[#J]`：(Integer) The total number of trigonometric function terms $J$ in the PDE.
    * The meanings of the remaining variables are the same as in the pre-trained dataset.
* Data generation code：[../data_generation/custom_sinus.py](../data_generation/custom_sinus.py)。

## Wave Equation Dataset

* Equation Form: $u_{tt}+\mu u_t+Lu+bu_x+f(u)+s_T(t)s_X(x)=0$，$(t,x)\in[0,1]\times[-1,1]$,
    Initial Condition: $u(0,x)=g(x),u_t(0,x)=h(x)$.
    Where $f(u) = c_{1}u+c_{2}u^2+c_{3}u^3$,
    $Lu$ is chosen from the following with equal probability: $Lu=-c(x)^2u_{xx}$, $Lu=-c(x)(c(x)u_x)_x$, $Lu=-(c(x)^2u_x)_x$.
    The source term $s_T(t)s_X(x)$ is chosen from the following cases with equal probability: zero, scalar coefficient, space-dependent coefficient, time-dependent coefficient, space-time-dependent coefficient.
* Boundary Conditions: Periodic + Non-periodic. For the non-periodic case, it is randomly chosen from Dirichlet, Neumann, Robin and Mur (absorbing boundary condition) independently on the left and right boundaries.
* Download link：[PKU Disk](https://disk.pku.edu.cn/anyshare/en-us/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/4A89FAE282C5404D9866EE2E27D98B56)

* Filename format: `custom_v4.2_wave_[#bc]_c[#c]_k[#k]_seed[#r].hdf5`
    * For example: `custom_v4.2_wave_interval_cU3_k1e-02_4_seed1.hdf5`
    * `[#bc]`: Boundary condition type, `circ` for periodic, `interval` for non-periodic.
    * `[#k]`: Range of Positive Coefficient $\kappa(x)$ (default: `k1e-02_4` for $[10^{-2},4]$).
    * Other parameters are the same as the pretraining dataset.
* Data generation code: [../data_generation/custom_wave.py](../data_generation/custom_wave.py).
* Note that the generated wave equation dataset has not been sufficiently filtered, so some samples may contain non-physical modes generated by the solver (such as high-frequency checkerboard artifact).
    We use this to demonstrate that the training process of PDEformer can tolerate some abnormal data samples, but if you wish to use this dataset for other purposes, you may need to do some additional post-processing on your own.

## Wave Function Inverse Problem Dataset

* Equation Form and Boundary Conditions: Basically the same as the wave equation dataset, but the source term $s_T(t)s_X(x)$ is set to be time-space-dependent coefficient.
    For the non-periodic case, the boundary conditions are set as homogeneous Mur boundary (outgoing wave) at the left endpoint and homogeneous Neumann boundary (stress-free) at the right endpoint.
    Every data file contains 100 PDEs, and every PDE has 100 samples.
    The initial conditions $g(x),h(x)$ and source term $s_T(t)s_X(x)$ varies in each sample.
* Download link：[PKU Disk](https://disk.pku.edu.cn/anyshare/en-us/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/CDBCBEF0F0D4459C893F3CBBE62F521E)
* Filename format: `custom_v4.2_inv_wave_[#bc]_c[#c]_k[#k]_seed[#r].hdf5`
    * For example: `custom_v4.2_inv_wave_interval_cU3_k1e-02_4_seed1.hdf5`
    * The meaning of `[#bc]`, `[#c]`, `[#k]`, `[#r]` is the same as the wave equation dataset.
* Data generation code: [../data_generation/inverse_wave.py](../data_generation/inverse_wave.py).

## Multi-Component Equation Dataset

* Equation Form: $\partial_tu_i + \sum_jc_{ij}u_j + s_i +\partial_x(\sum_ja_{ij}u_j + \sum_{j,k}b_{ijk}u_ju_k - \kappa_i\partial_xu_i) = 0$,
    Where $0 \le i,j,k \le d-1$, $j \le k$, $(t,x)\in[0,1]\times[-1,1]$, $d$ represents the number of components.
    The coefficients $c_{ij},a_{ij},b_{ijk}$ are all sparse matrices/ tensor, whose number of non-zero elements are independently sampled from $\{0,1,\dots,2d\}$.
* Boundary Conditions: In consideration of the complexity of the multi-component equation, we only provide periodic boundary conditions.
* Download link：[PKU Disk](https://disk.pku.edu.cn/anyshare/en-us/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/5EB8C7AF3CD34BA28869DBD6944871E1)
* Filename format: `custom_v4.2_compn[#d]_c[#c]_k[#k]_seed[#r].hdf5`
    * For example: `custom_v4.2_compn2_cU3_k1e-03_1_seed1.hdf5`
    * `[#d]`: Number of components.
    * The meaning of `[#c]`, `[#k]`, `[#r]` is the same as the pretraining dataset.
* Data generation code: [../data_generation/custom_multi_component.py](../data_generation/custom_multi_component.py).

## PDEBench Dataset

The data loading module supports a subset of 1D equations from the PDEBench dataset, including the Burgers equation, Advection equation, and Reaction-Diffusion equation.
PDEBench updated the Burgers equation dataset during our development, and we did not modify the data loading module to support this update.
Therefore, it is necessary to download the old version of Burgers equation dataset.
For the forms of the equations and initial boundary conditions, please refer to [PDEBench paper](https://arxiv.org/abs/2210.07182).
We present the download method and usage for these datasets in the following text.

### Burgers Equation

* Based on the pre-update version of the PDEBench dataset.
* Filename format: `1D_Burgers_Sols_Nu[#f].hdf5`
    * For example: `1D_Burgers_Sols_Nu0.1.hdf5`
    * `[#f]`: viscosity coefficient, a float number.

Configuration file setting：

```yaml
# ...
model:
    # ...
    load_ckpt: path/to/your/downloaded/model-L_3M_pretrained.ckpt  # pretrained model weights
# ...
data:
    path: ../data_download  # dataset path
    num_samples_per_file:
        train: 9000  # number of samples in each training dataset file
        test: 1000  # number of samples in each test dataset file
    single_pde:
        param_name: burgers_nu2  # parameter name
        train: [0.1]  # viscosity
        test: [0.1]  # viscosity
    # ...
# ...
```

#### Download Links

* Viscosity coefficient 0.001: [https://darus.uni-stuttgart.de/api/access/datafile/133133](https://darus.uni-stuttgart.de/api/access/datafile/133133).
* Viscosity coefficient 0.01: [https://darus.uni-stuttgart.de/api/access/datafile/133136](https://darus.uni-stuttgart.de/api/access/datafile/133136).
* Viscosity coefficient 0.1: [https://darus.uni-stuttgart.de/api/access/datafile/133139](https://darus.uni-stuttgart.de/api/access/datafile/133139).

#### Using Command Line to Download

```bash
# Viscosity coefficient 0.001
wget https://darus.uni-stuttgart.de/api/access/datafile/133133 -O 1D_Burgers_Sols_Nu0.001.hdf5
```

```bash
# Viscosity coefficient 0.01
wget https://darus.uni-stuttgart.de/api/access/datafile/133136 -O 1D_Burgers_Sols_Nu0.01.hdf5
```

```bash
# Viscosity coefficient 0.1
wget https://darus.uni-stuttgart.de/api/access/datafile/133139 -O 1D_Burgers_Sols_Nu0.1.hdf5
```

### Advection Equation

* Based on the current version of the PDEBench dataset.
* PDEBench dataset: [https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)
* Filename format: `1D_Advection_Sols_beta[#f].hdf5`
    * For example: `1D_Advection_Sols_beta1.0.hdf5`
    * `[#f]`: advection coefficient, a float number.
* Configuration file setting: Different from the Burgers configuration in the following lines

```yaml
# ...
data:
    # ...
    single_pde:
        param_name: adv_beta  # parameter name
        train: [1.0]  # advection speed
        test: [1.0]  # advection speed
    # ...
# ...
```

### Reaction-Diffusion Equation

* Based on the current version of the PDEBench dataset.
* PDEBench dataset: [https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)
* Filename format: `ReacDiff_Nu[#f1]_Rho[#f2].hdf5`
    * For example: `ReacDiff_Nu1.0_Rho1.0.hdf5`
    * `[#f1]`: diffusion coefficient, a float number
    * `[#f2]`: reaction coefficient, a float number
* Configuration file setting: Different from the Burgers configuration in the following lines

```yaml
# ...
data:
    # ...
    single_pde:
        param_name: reacdiff_nu_rho
        train: [[1.0,1.0]]  # diffusion coefficient, reaction coefficient
        test: [[1.0,1.0]]  # diffusion coefficient, reaction coefficient
    # ...
# ...
```
