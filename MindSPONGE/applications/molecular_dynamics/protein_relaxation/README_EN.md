# Relaxation of monkeypox virus protein

A given a protein molecule may not in a stable state no matter it is a predicted result or a real structure.
Therefore, if the molecular dynamics simulation algorithm is directly executed, a slightly larger step size may lead to the problem of gradient explosion,
the molecular dynamics simulation cannot be performed normally. Therefore, pre-process of protein molecules is necessary.
Generally, before performing molecular dynamics simulation, we need to perform energy minimization process for protein molecules, that is, protein relaxation.
Mindsponge is a next-generation molecular dynamics simulation platform based on the MindSpore deep learning framework,
Thanks to the compatibility of the framework (as shown in the figure below), we can not only use MindSponge to perform the task of molecular dynamics simulation,
but also directly utilize the native optimizer of MindSpore to minimize energy for proteins.

![img.png](../../../docs/img.png)

## Example background

Here, in order to demonstrate the function of protein relaxation of MindSponge, we selected a protein fragment `Q8V4Y0` from monkeypox virus that has recently attracted wide attention as an example.
Monkeypox is a viral zoonosis (a virus transmitted from animals to humans). Its symptoms are very similar to those seen in smallpox patients in the past, but not so serious clinically.
Human monkeypox was first found in a 9-year-old boy in the Democratic Republic of the Congo in 1970. But it affects not only African countries but also other areas of the world.
According to the latest data, monkeypox has been spread to 75 countries and regions around the world, and more than 16000 cases have been reported to the World Health Organization.
More than 1000 cases of monkeypox were reported in the United States, Spain, Britain and France. The following are some important facts about monkeypox published by the World Health Organization:

- Monkeypox is caused by monkeypox virus, which is a kind of orthopoxvirus in the family poxviridae.
- Monkeypox is a viral zoonosis, mainly occurring in the tropical rain forest areas of central and West Africa, and occasionally exported to other areas.
- The typical clinical manifestations of monkeypox are fever, rash and lymph node enlargement, which may lead to a series of complications.
- Monkeypox is usually a self limiting disease with symptoms lasting 2 to 4 weeks. Serious cases may occur. In recent years, the mortality rate is about 3-6%.
- Monkeypox virus is transmitted to humans through close contact with infected people or animals, or contact with objects contaminated by the virus.
- Monkeypox virus spreads from person to person through close contact with skin damage, body fluids, respiratory droplets, bedding and other contaminated objects.
- The clinical manifestation of monkeypox is similar to that of smallpox, a related orthopox virus infection, which was declared to be eradicated worldwide in 1980. Monkeypox is not as infectious as smallpox, and the disease caused is not so serious.
- Vaccines used during the smallpox eradication program also have a protective effect on monkeypox. Newer vaccines have been developed, one of which has been approved to prevent monkeypox.
- An antiviral drug developed to treat smallpox has also been licensed to treat monkeypox.

One of the core proteins of monkeypox virus: ` E8L `, which is the cell surface binding protein of monkeypox virus, binds to chondroitin sulfate on the cell surface during the process of virus invasion and host replication to provide attachment of viral particles to the target cells.
Therefore, through the molecular dynamics study of E8L, we can further understand the mechanism of the combination of monkeypox virus and cells, and then give the blocking scheme.

## E8L protein relaxation case

E8L protein consists of 304 amino acids. The following is the specific sequence information:

```bash
>sp|Q8V4Y0|CAHH_MONPZ Cell surface-binding protein OS=Monkeypox virus (strain Zaire-96-I-16) OX=619591 GN=E8L PE=2 SV=1
MPQQLSPINIETKKAISDTRLKTLDIHYNESKPTTIQNTGKLVRINFKGGYISGGFLPNEYVLSTIHIYWGKEDDYGSNHLIDVYKYSGEINLVHWNKKKYSSYEEAKKHDDGIIIIAIF
LQVSDHKNVYFQKIVNQLDSIRSANMSAPFDSVFYLDNLLPSTLDYFTYLGTTINHSADAAWIIFPTPINIHSDQLSKFRTLLSSSNHEGKPHYITENYRNPYKLNDDTQVYYSGEIIRA
ATTSPVRENYFMKWLSDLREACFSYYQKYIEGNKTFAIIAIVFVFILTAILFLMSQRYSREKQN
```

We only need such a given sequence to use the protein prediction platform MEGAProtein that based on MindSpore to predict its structure and obtain a protein pdb file with three-dimensional structure.
Generally, the generated pdb file does not contain hydrogen atoms. Hydrogen can be added to the pdb file through MindSponge, and then force field modeling for the complete protein conformation can be performed in mindsponge.
Energy minimization and molecular dynamics simulation. As shown in the figure below, the conformation comparison of E8L protein before and after relaxation is shown, in which blue represents the conformation before relaxation and red represents the conformation after relaxation.
From this structure, we can see some changes in the secondary structure of the protein before and after relaxation.

![protein-relax](../../../docs/Q8V4Y0.png)

## Protein relaxation code

Because there are a variety of energy minimization algorithms behind protein relaxation, which can also be used in combination with various optimization strategies, there is no clear data showing which optimization strategy has obvious advantages.
Therefore, this section only introduces the basic operation of protein relaxation and the corresponding common interfaces, enabling MindSponge users to independently design protein relaxation strategies.

### Create a protein instance

Mindsponge not only supports the instantiation on molecular level, but also supports the instantiation of encapsulated proteins, which can be applied to different scenarios. For example, here we use protein:

```python
from mindsponge import Protein
pdb_name = 'pdb/case2.pdb'
system = Protein(pdb=pdb_name)
```

### Force field modeling

After constructing the protein instance, we can create a force field space based on the system corresponding to the instance. In the general scenario, the force field space always remains unchanged during the evolution of the subsequent system.

```python
from mindsponge import ForceField
energy = ForceField(system, 'AMBER.FF14SB')
```

### Set optimizer/integrator

As a deep learning framework, MindSpore has built-in many optimizers that can be used. For example, ADAM algorithm can be utilized as follows:

```python
from mindspore import nn
learning_rate = 1e-03
opt = nn.Adam(system.trainable_params(), learning_rate=learning_rate)
```

Although molecular dynamics simulation will not be used in this paper, if necessary, the integrator can be called in the same way in MindSponge:

```python
from mindsponge import DynamicUpdater
from mindsponge.control import LeapFrog
integrator = LeapFrog(system)
opt = DynamicUpdater(system, integrator=integrator, time_step=1e-3)
```

### Encapsulate sponge instances

After defining the molecular system, force field parameter model and optimizer, you can build a Sponge instance to package these modules and start running. For example, the following example defines a step number of 500 steps.
In addition, the RunInfo information output module specifies parameters such as energy and temperature for outputting an intermediate conformation every 100 steps, so that we can understand the state of the intermediate process.

```python
from mindsponge import Sponge
from mindsponge.callback import RunInfo
md = Sponge(system, energy, opt)
run_info = RunInfo(100)
md.run(500, callbacks=[run_info])
```

If we need to adjust the optimizer during the operation, the switching function is also supported in the Sponge instance:

```python
md.change_optimizer(new_opt)
md.run(500, callbacks=[run_info])
```

Of course, if you need to save the trajectory during protein relaxation, you can also save it with WriteH5MD in Callback:

```python
from mindsponge.callback import WriteH5MD
cb_h5md = WriteH5MD(system, 'example.h5md', save_freq=100, write_velocity=True, write_force=True)
md.run(500, callbacks=[run_info, cb_h5md])
```

### Complete example

According to the above introduction to the use of basic MindSponge code, we can integrate it into an example to specifically demonstrate how to use MindSponge to relax proteins.

```python
# example_relax.py
from mindsponge import Protein
from mindsponge import ForceField
from mindspore import nn
from mindsponge import Sponge
from mindsponge.callback import RunInfo, WriteH5MD

pdb_name = 'Q8V4Y0_unrelaxed.pdb'
system = Protein(pdb=pdb_name)
energy = ForceField(system, 'AMBER.FF14SB')
learning_rate = 1e-03
opt = nn.Adam(system.trainable_params(), learning_rate=learning_rate)
md = Sponge(system, energy, opt)
run_info = RunInfo(50)
cb_h5md = WriteH5MD(system, 'example.h5md', save_freq=10, write_velocity=True, write_force=True)
md.run(500, callbacks=[run_info, cb_h5md])
```

The operation output results are as follows:

```bash
$ python3 example_relax.py
1 H-Adding task complete.
Step: 0, E_pot: 293502.75,
Step: 50, E_pot: 8617.799,
Step: 100, E_pot: -9117.585,
Step: 150, E_pot: -16084.797,
Step: 200, E_pot: -19855.645,
Step: 250, E_pot: -22055.75,
Step: 300, E_pot: -23588.682,
Step: 350, E_pot: -24745.182,
Step: 400, E_pot: -25619.227,
Step: 450, E_pot: -26282.828,
```

At this time, you can also find the generated track file in h5md format under the specified path. Generally, you can use visual tools such as VMD to visualize. The results are as follows:

![protein-relax](../../../docs/Q8V4Y0.gif)

### Usage of advanced optimizer

In addition to using a single simple optimizer, MindSponge users can also customize some combination optimization strategies, such as configuring energy options, modifying energy coefficients, and using multiple optimizers.
In the example of MindSponge, we built an optimization strategy. By using this optimization strategy, we can not only reduce the potential energy of the protein system, but also maintain its original structural characteristics to a great extent.

1. Download protein optimization strategy of MindSponge.

```bash
git clone https://gitee.com/mindspore/mindscience.git && cd mindscience/MindSPONGE/applications/molecular_dynamics/protein_relaxation/
```

2. Use MindSponge's protein optimization strategy as a script.

```bash
python3 protein_relax.py -i input.pdb -o optimized.pdb
```

3. If the value of `violation loss` displayed in the optimization process is `0.0`, it means that the optimization is successful. Otherwise, the optimization strategy needs to be formulated again.

## References

1. https://www.who.int/zh/news-room/fact-sheets/detail/monkeypox

## Acknowledgement

The initial E8L protein conformation used in this case is predicted by amino acid sequence with [MEGAProtein](https://gitee.com/mindspore/mindscience/tree/9ec9fa5fe1ea3f9fa77ef734326ca94797913c81/MindSPONGE/applications/MEGAProtein)
