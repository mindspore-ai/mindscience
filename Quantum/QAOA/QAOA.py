from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, Rzz, RX
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import mindspore as ms
import mindspore.nn as nn
import networkx as nx
import numpy as np


def build_hc(g, para):
    hc = Circuit()
    for i in g.edges:
        hc += Rzz(para).on(i)
    hc.barrier()
    return hc


def build_hb(g, para):
    hb = Circuit()
    for i in g.nodes:
        hb += RX(para).on(i)
    hb.barrier()
    return hb


def build_ansatz(g, p):
    circ = Circuit()
    for i in range(p):
        circ += build_hc(g, f"g{i}")
        circ += build_hb(g, f"b{i}")
    return circ


def build_ham(g):
    ham = QubitOperator()
    for i in g.edges:
        ham += QubitOperator(f"Z{i[0]} Z{i[1]}")
    return ham


def create_graph():
    graph = nx.Graph()
    nx.add_path(graph, [0, 1])
    nx.add_path(graph, [1, 2])
    nx.add_path(graph, [2, 3])
    nx.add_path(graph, [3, 4])
    nx.add_path(graph, [0, 4])
    nx.add_path(graph, [0, 2])
    return graph


def main():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    graph = create_graph()
    print("图的节点数:", len(graph.nodes))
    print("图的边数:", len(graph.edges))

    p = 4

    ham = Hamiltonian(build_ham(graph))
    init_state_circ = UN(H, graph.nodes)
    ansatz = build_ansatz(graph, p)
    circ = init_state_circ + ansatz

    sim = Simulator("mqvector", circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ)

    net = MQAnsatzOnlyLayer(grad_ops)
    opti = nn.Adam(net.trainable_params(), learning_rate=0.05)
    train_net = nn.TrainOneStepCell(net, opti)

    print("\n开始训练量子神经网络...")
    for i in range(200):
        res = train_net()
        cut = (len(graph.edges) - res.asnumpy().item()) / 2
        if i % 10 == 0:
            print(f"训练步骤: {i}, 切割边数: {cut:.4f}")

    print("\n训练完成!")
    final_res = net()
    final_cut = (len(graph.edges) - final_res.asnumpy().item()) / 2
    print(f"最终切割边数: {final_cut:.4f}")


if __name__ == "__main__":
    main()
