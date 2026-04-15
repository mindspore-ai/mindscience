#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit, dagger
from mindquantum.algorithm import qft
from mindquantum.utils import normalize
import numpy as np


def qjpeg(figure: np.ndarray, n_qubits: int, m_qubits: int) -> np.ndarray:
    """Compress input image using QJPEG algorithm.

    Args:
        figure: Input image numpy array
        n_qubits: Number of qubits for original image
        m_qubits: Number of qubits for compressed image

    Returns:
        np.ndarray: Compressed image data
    """
    origin_figure = figure.reshape(-1)

    if not isinstance(n_qubits, int) or not isinstance(m_qubits, int):
        raise ValueError("n_qubits and m_qubits should be positive int.")
    if n_qubits <= 0 or m_qubits <= 0:
        raise ValueError("n_qubits and m_qubits should be positive int.")
    if n_qubits <= m_qubits:
        raise ValueError("m_qubits should be less than n_qubits.")
    if np.log2(origin_figure.shape[0]) != n_qubits:
        raise ValueError(
            "the pixel number of the input figure should equal to 2**n_qubits."
        )
    if (n_qubits - m_qubits) % 2 != 0:
        raise ValueError("the difference between n_qubits and m_qubits should be even.")

    half_diff = (n_qubits - m_qubits) // 2
    former_qubits = list(range(0, n_qubits // 2))
    latter_qubits = list(range(n_qubits // 2, n_qubits))
    mid_qubits = former_qubits[
        len(former_qubits) - half_diff :
    ]
    last_qubits = latter_qubits[
        len(latter_qubits) - half_diff :
    ]

    state = normalize(origin_figure)
    sim = Simulator("mqmatrix", n_qubits)
    sim.set_qs(state)

    circ = Circuit()
    circ += qft(range(n_qubits))
    circ += dagger(qft(range(n_qubits - half_diff)))
    sim.apply_circuit(circ)
    rho = sim.get_partial_trace(mid_qubits + last_qubits)
    sub_pros = rho.diagonal().real
    new_figure = sub_pros.reshape((2 ** (m_qubits // 2), -1))
    return new_figure


def run_qjpeg_demo():
    grid = np.zeros((8, 8))
    grid[:2, :2] = 1
    grid[4:6, :2] = 1
    grid[:2, 4:6] = 1
    grid[2:4, 2:4] = 1
    grid[6:8, 2:4] = 1
    grid[4:6, 4:6] = 1
    grid[6:8, 6:8] = 1
    grid[2:4, 6:8] = 1

    n_qubits = 6
    m_qubits = 4

    result = qjpeg(grid, n_qubits, m_qubits)

    original_info_ratio = np.count_nonzero(grid) / grid.size

    threshold = np.mean(result)
    binary_result = (result > threshold).astype(np.float64)
    retained_info_ratio = np.sum(binary_result) / binary_result.size

    compression_ratio = 8**2 / 4**2
    information_retention = retained_info_ratio / original_info_ratio

    print(f"\n===== Q-JPEG算法执行结果 =====")
    print(f"压缩比: {compression_ratio} : 1")
    print(f"图像信息保留率: {information_retention:.4f}")


if __name__ == "__main__":
    run_qjpeg_demo()
