from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, Z
from mindquantum.simulator import Simulator
import numpy as np


def bitphaseflip_operator(phase_inversion_qubit, n_qubits):
    s = [1 for i in range(1 << n_qubits)]
    for i in phase_inversion_qubit:
        s[i] = -1
    if s[0] == -1:
        for i in range(len(s)):
            s[i] = -1 * s[i]
    circuit = Circuit()
    length = len(s)
    cz = []
    for i in range(length):
        if s[i] == -1:
            cz.append([])
            current = i
            t = 0
            while current != 0:
                if (current & 1) == 1:
                    cz[-1].append(t)
                t += 1
                current = current >> 1
            for j in range(i + 1, length):
                if i & j == i:
                    s[j] = -1 * s[j]
    for i in cz:
        if i:
            if len(i) > 1:
                circuit += Z.on(i[-1], i[:-1])
            else:
                circuit += Z.on(i[0])
    return circuit


def grover_search(target_state, n_qubits, iterations=1):
    sim = Simulator("mqvector", n_qubits)
    circuit = Circuit()

    circuit += UN(H, n_qubits)

    for _ in range(iterations):
        circuit += bitphaseflip_operator([target_state], n_qubits)

        circuit += UN(H, n_qubits)
        circuit += bitphaseflip_operator(list(range(1, 2**n_qubits)), n_qubits)
        circuit += UN(H, n_qubits)

    sim.apply_circuit(circuit)

    return circuit, sim


def main():
    n_qubits = 3
    target_state = 6

    N = 2**n_qubits
    optimal_iterations = int(np.pi / 4 * np.sqrt(N))

    print(f"量子比特数量: {n_qubits}")
    print(f"搜索空间大小: {N}")
    print(f"目标态: |{bin(target_state)[2:].zfill(n_qubits)}⟩")
    print(f"最优迭代次数: {optimal_iterations}")
    print("\n开始执行 Grover 搜索算法...")

    circuit, sim = grover_search(target_state, n_qubits, optimal_iterations)

    result = sim.get_qs(True)
    print("\n最终量子态:")
    print(result)

    probs = np.abs(sim.get_qs()) ** 2
    print("\n测量概率:")
    for i in range(N):
        if probs[i] > 0.01:
            print(f"|{bin(i)[2:].zfill(n_qubits)}⟩: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
