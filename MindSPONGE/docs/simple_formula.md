[TOC]


# Restrain

## RestrainForce

计算 Restrain 产生的回复力：

$$
\boldsymbol{F}_{\mathrm{restrain}} = -2k(\boldsymbol{r})
$$

$$
\boldsymbol{F}_{\mathrm{restrain}} = -2k(\boldsymbol{r} - \boldsymbol{r}_{\mathrm{ref}})
$$

## RestrainEnergy

计算 Restrain 产生的能量：
$$
E_{\mathrm{restrain}} = k(\boldsymbol{r}-\boldsymbol{r}_{\mathrm{ref}})^2
$$

## RestrainForceWithAtomEnergyAndVirial

计算 Restrain 产生的力、能量和维里，计算维里的统一公式为：
$$
\Xi = -\frac{1}{2}\sum_{i < j}^N \boldsymbol{r}_{ij} \otimes \boldsymbol{F}_{ij}
$$
下面涉及到维里的部分将不再重复给出。

# Constrain

在正常的 MD 迭代步骤（如 Leap Frog）后，将进行多次 constrain 迭代，每个 constrain 迭代步骤将按照下式迭代坐标：
$$
\boldsymbol{r}_i' = \boldsymbol{r}_i + \sum_{j} k \frac{m_j}{m_i + m_j}\Delta  t^2\left(1-\frac{r_0}{r_{ij}}\right)\boldsymbol{r}_{ij ,0}
$$
其中 $\boldsymbol{r}_i$​​​ ​是原子 $i$​​​ ​的坐标，$\sum_j$​​ ​​代表对所有和 $i$​​​ ​原子有约束的原子进行求和，$\boldsymbol{r}_{ij,0}$ ​​​​代表正常迭代前的 $\boldsymbol{r}_i-\boldsymbol{r}_j$ ，$\Delta t$​​​​​为步长。

在每次正常迭代之前，将会调用 lastcrdtodr，记录下 $\boldsymbol{r}_{ij,0}$​，然后进行正常的迭代，之后进行多次 constrain 迭代，每次 constrain 迭代中将调用 constrainforcecycle 计算力（如果需要计算 virial 的话调用 constrainforcecyclewithvirial），再调用 refreshuintcrd 更新坐标，使总的更新结果与上面给出的式子相符，在所有迭代完成之后，调用 refreshcrdvel 更新坐标和速度。

# Crd Molecular Map

## CalculateNowrapCrd

计算每个原子在周期性盒子内的周期性映射，即：
$$
\boldsymbol{r}_i' = \boldsymbol{r}_i + (T_x, T_y, T_z)^\mathrm{T}(L_x, L_y, L_z)
$$
其中 $T_x, T_y, T_z$ 分别是该原子穿越盒子的次数（为整数），$L_x, L_y, L_z$​ 为周期性盒子的大小。

## RefreshBoxmapTimes

更新每个原子穿越盒子的次数，即：
$$
T_\alpha'= T_\alpha + \left\lfloor\frac{\boldsymbol{r}_{i,\alpha}' - \boldsymbol{r}_{i,\alpha}}{L_{\alpha}}\right\rfloor
$$
更新了$\alpha$​方向上穿越盒子的次数。

# Long Range Correction

## Totalc6get

得到 Lennard-Jones 的 total dispersion constant：
$$
C_{6,\mathrm{tot}} =\sum_{i}^N \sum_{j > i}^N C_6(i,j)
$$
$C_6(i,j)$ 是 $i,j$ 之间的 Lennard-Jones 吸引系数。

# MD Core

## GetCenterOfMass

得到一个残基的质心：
$$
\boldsymbol{r}_{m_\mathrm{c}}  = \frac{\sum_i m_i \boldsymbol{r}_i}{\sum_{i} m_i}
$$

## MapCenterOfMass

将一个残基的所有原子映射到同一个周期性盒子中：
$$
\boldsymbol{r}_{i,\alpha}' = \boldsymbol{r}_{i,\alpha}-\left\lfloor \frac{\boldsymbol{r}_{m_\mathrm{c},\alpha }}{L_\alpha}\right\rfloor L_\alpha
$$

# Iteration

## MDIterationLeapFrog

Leap Frog 蛙跳差分算法：
$$
\boldsymbol{r}_{i, t+1} = \boldsymbol{r}_{i, t} + \boldsymbol{v}_{t+\frac{1}{2}}\Delta t\\
\boldsymbol{v}_{i,t+\frac{1}{2}} = \boldsymbol{v}_{i,t-\frac{1}{2}} + \frac{\boldsymbol{F}_{i,t}}{m_i}
$$

# Bond

## BondEnergy

计算 Bond 的能量：
$$
E_{\mathrm{bond},ij} = k (|\boldsymbol{r}_{ij}| -  r_0)^2
$$
$k,r_0$ 分别是力常数和平衡距离，下同。

## BondForce

计算 Bond 的力：
$$
\boldsymbol{F}_{\mathrm{bond},ij} = 2k \left(1-\frac{r_0}{|\boldsymbol{r}|}\right)\boldsymbol{r}_{ij}
$$
BondForceWithAtomVirial, BondForceWithAtomEnergy 同。

# Angle

# Dihedral

## DihedralEnergy

$$
E_{\mathrm{dihedral},abcd} = k(1+\cos(n\phi- \phi_0))
$$

$k,n,\phi_0$ 分别是二面角项的力常数，周期，平衡角度， $\phi$ 是四个原子 $a, b, c, d$ 形成的二面角。

## DihedralForce

$$
\boldsymbol{F}_{\mathrm{dihedral}, i} = -\frac{\partial E_{\mathrm{dihedral}}}{\partial \boldsymbol{r}_i}
$$

DihedralAtomEnergy, DihedralForceWithAtomEnergy 同。

# Angle

## AngleEnergy

$$
E_{\mathrm{angle},abc} = k(\theta - \theta_0)^2
$$

$k,\theta_0$ 分别是力常数，平衡角度，$\theta$ 是三个原子 $a,b,c$ 形成的角。

## AngleForce

$$
\boldsymbol{F}_{\mathrm{angle}, i} = -\frac{\partial E_{\mathrm{angle}}}{\partial \boldsymbol{r}_i}
$$

AngleAtomEnergy, AngleForceWithAtomEnergy 同。

# Dihedral14

## Dihedral14LJForce

$$
\boldsymbol{F}_{\mathrm{14, lj},ij} = k_{\mathrm{lj}}\left(-\frac{12A}{|\boldsymbol{r}_{ij}|^{14}} + \frac{6B}{|\boldsymbol{r}_{ij}|^8}\right)\boldsymbol{r}_{ij}
$$

其中 $k_\mathrm{lj}$ 是 nb14 的 LJ 系数，$A, B$ 是 LJ 的排斥系数和吸引系数，下同

## Dihedral14LJEnergy

$$
E_{\mathrm{14, lj},ij} = k_{\mathrm{lj}}\left(\frac{A}{|\boldsymbol{r}_{ij}|^{12} } - \frac{B}{|\boldsymbol{r}_{ij}|^6}\right)
$$

## Dihedral14CFEnergy

$$
E_{\mathrm{14,ee},ij} = k_{\mathrm{ee}}\frac{q_iq_j}{|\boldsymbol{r}_{ij}|}
$$

其中 $k_{\mathrm{ee}}$ 是 nb14 的 ee 系数，下同

## Dihedral14CFForce

$$
\boldsymbol{F}_{\mathrm{14,ee},ij} = -k_{\mathrm{ee}}\frac{q_iq_j}{|\boldsymbol{r}_{ij}|^3}\boldsymbol{r}
$$

Dihedral14LJForceWithDirectCF, Dihedral14LJForceWithAtomEnergy, Dihedral14LJAtomEnergy, Dihedral14CFEnergy, Dihedral14CFAtomEnergy 同理。

# Lennard-Jones

## LJEnergy

$$
E_{\mathrm{ lj},ij} = \frac{A}{|\boldsymbol{r}_{ij}|^{12} } - \frac{B}{|\boldsymbol{r}_{ij}|^6}
$$

## LJForce

$$
\boldsymbol{F}_{\mathrm{lj},ij} = \left(-\frac{12A}{|\boldsymbol{r}_{ij}|^{14}} + \frac{6B}{|\boldsymbol{r}_{ij}|^8}\right)\boldsymbol{r}_{ij}
$$

## LJForceWithPMEDirectForce

将 LJ 部分的力和静电的 Direct 部分一起计算：
$$
\boldsymbol{F}_{ij} = \left(-\frac{12A}{|\boldsymbol{r}_{ij}|^{14}} + \frac{6B}{|\boldsymbol{r}_{ij}|^8} - \frac{2\beta \exp(-\beta^2|\boldsymbol{r}_{ij}|^2)}{\sqrt{\pi}|\boldsymbol{r}_{ij}|} - \frac{\mathrm{erfc}(\beta |\boldsymbol{r}|^2)}{|\boldsymbol{r}_{ij}|^3}\right)\boldsymbol{r}_{ij}
$$
