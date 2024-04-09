# Representation of Boundary Conditions, Coefficient Fields, and Other Elements in Computational Graph for PDEs

## Boundary Conditions

If the PDE uses periodic boundary conditions, there is no need to introduce additional nodes in the corresponding computational graph.
For non-periodic boundaries, taking the case of a left boundary Dirichlet and right boundary Neumann as an example:

![](../images/PDEformer-BC-DirNeum2.png)

(Note: Unless specifically mentioned, all scalar encoders in the diagrams below share parameters, and the same applies to function encoders.)

Here is a more complex example, with a Robin condition on the left boundary and a Mur condition on the right boundary (which is an absorbing boundary condition for wave equations):

![](../images/PDEformer-BC-RobinMur2.png)

## Non-constant Coefficients

If the coefficients involved in the PDE are not constants but a space-dependent coefficient field $s(x)$, such coefficients can be represented by a `CF` node.
The example shown below appears in equations that include a non-constant diffusion term:

![](../images/PDEformer-CF.png)

Coefficients that depend on time (i.e., of the form $s(t)$, which is a function of time $t$) can also be similarly represented by a `VC` node.
By introducing additional product nodes, a variable coefficient field of the form $s^\mathrm{T}(t)s^\mathrm{X}(x)$ (which has separable variables) can be expressed in the computational graph.
The current code does not yet support representing general variable coefficient fields $s(t,x)$.

## Second-order Time Derivative

The following diagram uses a simple wave equation as an example to show how equations containing second-order time derivatives can be represented through computational graphs:

![](../images/PDEformer-waveEqn-v2.png)

## Multivariable Equations

Systems of partial differential equations that contain multiple variables (multiple components of the equation solution) can also be represented using computational graphs. A simple example is shown below (although we currently do not provide model parameters trained on multi-component equation data):

![](../images/PDEformer-mCompn.png)

# Programming Implementation—Assigning Input Feature Embeddings to Graph Nodes

Recall the overall architecture diagram of PDEformer shown below:

![](../images/PDEformerV2Arch.png)

In the DAG construction process shown on the left, we assign an input feature embedding of dimension $d_e$ to each node in the graph. Some nodes receive input features from scalar encoders, while others receive them from function encoders.
For this setup, the most direct and naive programming implementation is as follows (to simplify the diagram, we set $L=N=2$ and use an incomplete equation for PDE 2):

![](../images/FuncEmb-naive.png)

For this approach, we need to examine the information of each input function involved in the PDE, determine the permutation position of the corresponding "branch" nodes ($\mathtt{b}_1,\dots,\mathtt{b}_N$) among all graph nodes, and then fill the output of the function encoder into the corresponding position of the overall input feature tensor (which contains all vertices). (In programming implementation, all inputs and outputs of the network are in tensor data form.) In the training process, a data batch often contains multiple different PDEs, each with a different number of input functions, and the permutation positions of the corresponding branch nodes are also different. If this implementation method is used, the code is not only cumbersome to write but also inefficient at runtime. In programming implementation, we adopt the following alternative approach:

![](../images/FuncEmb-actual.png)

(Here, we take advantage of the special nature of the graph Transformer: unlike most Transformers designed for sequence data, the forward computation process of the graph Transformer only uses topological structure information of the graph and does not consider the order of nodes, so we can rearrange nodes arbitrarily.)
This approach specifies that the input features of nodes with earlier permutation positions are provided by scalar encoders, while those with later positions are provided by function encoders.
With relatively simple tensor shape modifications (reshape) and tensor concatenation (concatenate) operations, the final overall input feature tensor can be obtained without the need for tedious judgment and assignment processes.