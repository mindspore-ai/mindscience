### Dirichlet

Since the Dirichlet boundary condition is more common in solving the possion equation, the basic case of the Dilrichlet boundary condition under different sampling intervals is made into the following table:

|    gemo    |                                          Training parameters                                          | Testing Parameters | per step time | training error | test error |
| :---------: | :---------------------------------------------------------------------------------------------------: | :----------------: | :-----------: | :------------: | :---------: |
|    disk    | n_epochs=50,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000,<br />max_lr=5e-4 |  batch_size=5000  |    163.2ms    |    0.00647    |   0.0130   |
|  triangle  | n_epochs=50,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000,<br />max_lr=5e-4 |  batch_size=5000  |    163.2ms    |    0.00768    |   0.0284   |
|  pentagon  | n_epochs=50,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000,<br />max_lr=5e-4 |  batch_size=5000  |    163.2ms    |     0.0672     |   0.0273   |
|    cone    | n_epochs=50,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000,<br />max_lr=5e-4 |  batch_size=5000  |    417.6ms    |    0.00564    |   0.0458   |
|  cylinder  | n_epochs=50,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000,<br />max_lr=5e-4 |  batch_size=5000  |    417.6ms    |    0.00812    |   0.0472   |
| tetrahedron | n_epochs=50,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000,<br />max_lr=5e-4 |  batch_size=5000  |    417.6ms    |    0.00463    |   0.0462   |

### Robin

Since Robin are not often used in possion equations, only one example in 2D and 3D is given here for reference only.

| Boundary |    gemo    |                                          Training parameters                                          | Testing Parameters | per step time | training error | test error |
| :------: | :---------: | :----------------------------------------------------------------------------------------------------: | :----------------: | :-----------: | :------------: | :---------: |
|  Robin  |    disk    | n_epochs=80,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000,<br />max_lr=6e-4 |  batch_size=5000  |    135.3ms    |    0.02613    |   0.0363   |
|  Robin  | tetrahedron | n_epochs=50,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000,<br />max_lr=55e-4 |  batch_size=5000  |    338.2ms    |     0.7490     |   0.4984   |

### Periodic

Since Periodic are not often used in possion equations, only one example in 2D and 3D is given here for reference only.

| Boundary | gemo |                                          Training parameters                                          | Testing Parameters | per step time | training error | test error |
| :------: | :--: | :---------------------------------------------------------------------------------------------------: | :----------------: | :-----------: | :------------: | :---------: |
| Periodic | disk | n_epochs=80,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000,<br />max_lr=6e-4 |  batch_size=5000  |    124.2ms    |    0.09816    |   0.0873   |
| Periodic | cone | n_epochs=50,<br />batch_size=5000,<br />BC_size=1000000, <br />domain_size=1000000,<br />max_lr=5e-4 |  batch_size=5000  |    322.4ms    |     0.0264     |   0.5582   |
