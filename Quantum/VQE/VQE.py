#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq import generate_uccsd
import mindspore as ms


def calc_lih_ground_energy():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("CPU")
    ms.set_seed(1)

    dist = 1.5
    geometry = [
        ["Li", [0.0, 0.0, 0.0 * dist]],
        ["H", [0.0, 0.0, 1.0 * dist]],
    ]
    basis = "sto3g"
    spin = 0

    molecule_of = MolecularData(
        geometry, basis, multiplicity=2 * spin + 1, data_directory="./"
    )

    molecule_of = run_pyscf(molecule_of, run_scf=1, run_ccsd=1, run_fci=1)

    molecule_of.save()
    molecule_file = molecule_of.filename

    hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(molecule_of.n_electrons)])

    ansatz_circuit, _, _, hamiltonian_QubitOp, _, _ = generate_uccsd(
        molecule_file, threshold=-1
    )

    total_circuit = hartreefock_wfn_circuit + ansatz_circuit

    sim = Simulator("mqvector", total_circuit.n_qubits)
    molecule_pqc = sim.get_expectation_with_grad(
        Hamiltonian(hamiltonian_QubitOp), total_circuit
    )

    def fun(p0, molecule_pqc):
        f, g = molecule_pqc(p0)
        f = np.real(f)[0, 0]
        g = np.real(g)[0, 0]
        return f, g

    n_params = len(total_circuit.params_name)
    p0 = np.zeros(n_params)

    res = minimize(fun, p0, args=(molecule_pqc,), method="bfgs", jac=True)

    return res.fun, molecule_of.fci_energy


if __name__ == "__main__":
    vqe_energy, fci_energy = calc_lih_ground_energy()

    relative_error = abs((vqe_energy - fci_energy) / fci_energy) * 100

    print("\n【MindQuantum VQE验证结果】")
    print(f"VQE计算的LiH基态能量: {vqe_energy:.10f} Ha")
    print(f"精确FCI计算结果: {fci_energy:.10f} Ha")
    print(f"相对误差: {relative_error:.6f}%")
