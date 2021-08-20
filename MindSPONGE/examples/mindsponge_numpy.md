# MindSPONGE with numpy support

`Linux` `GPU`

<!-- TOC -->

- [Run MindSPONGE with MindSpore.Numpy](#Run-MindSPONGE-with-MindSpore.Numpy)
- [Under the hood of the Numpy demo](#Under-the-hood-of-the-Numpy-demo)
- [Performance](#Performance)

<!-- /TOC -->

## Run MindSPONGE with MindSpore.Numpy

MindSPONGE has been fully supported by MindSpore.Numpy, which runs at similar speed
compared with the Cuda version, which means energy, force and coordinate calculations
are fully supported with Numpy-like syntax, which runs on top of MindSpore primitive
operators.

For numpy version, run the following command instead:

```shell
python main_np.py --i /path/NVT_290_10ns.in \
               --amber_parm /path/WATER_ALA.parm7 \
               --c /path/WATER_ALA_350_cool_290.rst7 \
               --o /path/ala_NVT_290_10ns.out
```

Detailed implementations:

```shell
├── scripts
    ├── main_np.py                               # launch Simulation for SPONGE-Numpy
    ├── src
        ├── simulation_np.py                     # SPONGE-numpy simulation
        ├── functions                            # SPONGE-numpy modules
            ├── angle_energy.py
            ├── angle_force_with_atom_energy.py
            ├── bond_energy.py
            ├── bond_force_with_atom_energy.py
            ├── common.py
            ├── crd_to_uint_crd.py
            ├── dihedral_14_cf_energy.py
            ├── dihedral_14_ljcf_force_with_atom_energy.py
            ├── dihedral_14_lj_energy.py
            ├── dihedral_energy.py
            ├── dihedral_force_with_atom_energy.py
            ├── __init__.py
            ├── lj_energy.py
            ├── lj_force_pme_direct_force.py
            ├── md_iteration_leap_frog_liujian.py
            ├── md_temperature.py
            ├── neighbor_list_update.py
            ├── pme_common.py
            ├── pme_energy.py
            ├── pme_excluded_force.py
            └── pme_reciprocal_force.py
```

For other details, please refer to the [tutorial on case tutorial](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/mindsponge/examples/case_polypeptide.md)

## Under the hood of the numpy demo

For the Cuda version, every kernel lies inside MindSpore as Cuda kernels. To allow
better transparency for how molecular dynamics algorithm works, Cuda kernels were
refactored as numpy scripts and displayed in src/functions. 

MindSpore Numpy package contains a set of Numpy-like interfaces, which allows developers
to build models on MindSpore with similar syntax of Numpy. MindSpore.Numpy is a layer
of wrapper on MindSpore Primitives (mindspore.ops) which runs on MindSproe Tensors,
therefore it is compatible with other MindSpore features. See
[here](https://www.mindspore.cn/docs/programming_guide/en/master/numpy.html) for
more details.

By default, the numpy demo runs on top of [Graph Kernel Fusion]
(https://www.mindspore.cn/docs/programming_guide/en/master/enable_graph_kernel_fusion.html)
and [Auto Kernel Generation] (https://gitee.com/mindspore/akg). These features
lead to 50% (or more) performance improvement, compared with no kernel fusion,
and brings the numpy version to comparable performance with the Cuda version.

To setup for Graph Kernel Fusion, simply add two lines in the launch script
(mindsponge/scripts/main_np.py):

```python
# Enable Graph Mode, with GPU as backend, and allow Graph Kernel Fusion
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id, enable_graph_kernel=True)
# Make fusion rules for specific operators
context.set_context(graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=TensorScatterAdd --enable_recompute_fusion=false")
```

## Performance

| Parameter                 |   GPU |
| -------------------------- |---------------------------------- |
| Resource                   | GPU (Tesla V100 SXM2); memory 16 GB
| Upload date              |
| MindSpore version          | 1.4
| Training parameter        | step=1000
| Output                    | numpy file
| Speed                      | 5.2 ms/step
| Script                    | [Link](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/mindsponge/scripts)
