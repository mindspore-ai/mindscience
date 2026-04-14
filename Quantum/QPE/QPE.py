from mindquantum.core.gates import T, H, X, Power, BARRIER
from mindquantum.core.circuit import Circuit, UN
from mindquantum.simulator import Simulator
from mindquantum.algorithm.library import qft
import numpy as np


def build_qpe_circuit(n_estimation_qubits, target_qubit_idx):
    circuit = Circuit()
    circuit += UN(H, n_estimation_qubits)
    circuit += X.on(target_qubit_idx)

    for i in range(n_estimation_qubits):
        circuit += Power(T, 2**i).on(target_qubit_idx, n_estimation_qubits - i - 1)

    circuit += BARRIER
    circuit += qft(range(n_estimation_qubits)).hermitian()

    return circuit


def quantum_phase_estimation(n_estimation_qubits=4, shots=100):
    total_qubits = n_estimation_qubits + 1
    target_qubit_idx = n_estimation_qubits

    circuit = build_qpe_circuit(n_estimation_qubits, target_qubit_idx)

    sim = Simulator("mqvector", total_qubits)
    sim.apply_circuit(circuit)

    qs = sim.get_qs()
    index = np.argmax(np.abs(qs))

    bit_string = bin(index)[2:].zfill(total_qubits)[1:]
    bit_string = bit_string[::-1]

    phase = int(bit_string, 2) / 2**n_estimation_qubits

    return circuit, sim, phase


def format_complex(z):
    if abs(z.imag) < 1e-10:
        return f"{z.real:.6f}"
    elif z.imag >= 0:
        return f"{z.real:.6f}+{z.imag:.6f}j"
    else:
        return f"{z.real:.6f}{z.imag:.6f}j"


def main():
    n_estimation_qubits = 4
    true_phase = 1 / 8

    print(f"估计量子比特数量: {n_estimation_qubits}")
    print(f"搜索空间大小: {2**n_estimation_qubits}")
    print(f"真实相位值: {true_phase}")
    print("\n开始执行量子相位估计算法...")

    circuit, sim, estimated_phase = quantum_phase_estimation(n_estimation_qubits)

    print(f"\n估计的相位值: {estimated_phase}")
    print(f"估计误差: {abs(estimated_phase - true_phase)}")

    qs = sim.get_qs()
    print("\n最终量子态的振幅：")
    for i in range(len(qs)):
        if abs(qs[i]) > 1e-6:
            binary = bin(i)[2:].zfill(n_estimation_qubits + 1)
            print(f"|{binary}⟩: {format_complex(qs[i])}")

    probs = np.abs(qs) ** 2
    print("\n测量概率分布：")
    print("基态\t\t概率\t\t对应相位")
    print("-" * 50)

    nonzero_indices = np.where(probs > 1e-6)[0]
    sorted_indices = sorted(nonzero_indices, key=lambda x: probs[x], reverse=True)

    for i in sorted_indices:
        binary = bin(i)[2:].zfill(n_estimation_qubits + 1)
        phase_binary = binary[1:][::-1]
        phase = int(phase_binary, 2) / 2**n_estimation_qubits
        print(f"|{binary}⟩\t{probs[i]:.6f}\t{phase:.6f}")


if __name__ == "__main__":
    main()
