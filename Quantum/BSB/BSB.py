import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from mindquantum.algorithm.qaia import BSB

# Generate random graph
G_nx = nx.erdos_renyi_graph(n=20, p=0.3, seed=42)
edges = list(G_nx.edges())
row = np.array([i for i, j in edges] + [j for i, j in edges])
col = np.array([j for i, j in edges] + [i for i, j in edges])
data = np.ones(len(row))
G = -coo_matrix((data, (row, col)), shape=(20, 20))

# Solve max-cut using BSB algorithm
solver = BSB(G, batch_size=20, n_iter=50)
solver.update()
cut_value = solver.calc_cut()
bsb_max_cut = np.max(cut_value)
print(f"BSB算法最大割值={bsb_max_cut:.2f}")


def greedy_cut(graph):
    n = graph.shape[0]
    G_dense = graph.toarray()
    assignment = np.ones(n)
    improved = True
    max_cut = 0

    while improved:
        improved = False
        for i in range(n):
            old_assignment = assignment[i]
            assignment[i] = -old_assignment
            cut = 0.25 * assignment.T @ G_dense @ assignment
            cut_value = -cut

            if cut_value > max_cut:
                max_cut = cut_value
                improved = True
            else:
                assignment[i] = old_assignment

    return max_cut


greedy_cut_value = greedy_cut(G)
print(f"贪心算法最大割值={greedy_cut_value:.2f}")
print(
    f"BSB算法相对贪心算法的改进: {((bsb_max_cut / greedy_cut_value) - 1) * 100:.2f}%"
)
