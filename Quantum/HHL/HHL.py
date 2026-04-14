import warnings
warnings.filterwarnings('ignore')
import numpy as np
from mindquantum.core.gates import H, X, RY, RZ, Measure, Power, BARRIER
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator


A = np.array([[1, 0], [0, -1]])
es, vs = np.linalg.eig(A)

# print(f"eigenvalues of A:\n {es}")
# print(f"eigenvectors of A:\n {vs}")

b = np.array([0.6, 0.8])
print(f"Solution of Ax=b by classical method: {np.linalg.solve(A, b)}")

circ = Circuit()
circ += RY(2 * np.arcsin(0.8)).on(0)
circ += Measure().on(0)

sim = Simulator(backend="mqvector", n_qubits=1)
res = sim.sampling(circ, shots=10000)

from mindquantum.algorithm.library import qft

# q1 ~ q4 : for QPE
t = 4
qc = [4, 3, 2, 1]

# store |b> , and store the result |x>
qb = 5

# use for conditional rotation
qs = 0

# choose a time evolution
dt = np.pi / 4

# choose a small C
C = 0.5

circ = Circuit()

# prepare b
circ += RY(2 * np.arcsin(0.8)).on(qb)

# QPE
for i in qc:
    circ += H.on(i)

for (i, q) in enumerate(qc):
    circ += Power(RZ(-2 * dt), 2**i).on(qb, q)

# apply inverse QFT
circ += qft(list(reversed(qc))).hermitian()

# conditional rotate
circ += BARRIER
for k in range(1, 2**t):
    for i in range(t):
        if (k & (1 << i)) == 0:
            circ += X.on(qc[i])
    phi = k / (2**t)
    if k > 2**(t-1):
        phi -= 1.0
    l = 2 * np.pi / dt * phi
    circ += RY(2 * np.arcsin(C / l)).on(qs, qc)

    for i in range(t):
        if (k & (1 << i)) == 0:
            circ += X.on(qc[i])
    circ += BARRIER

# apply inverse QPE
circ += qft(list(reversed(qc)))

for (i, q) in enumerate(qc):
    circ += Power(RZ(2 * dt), 2**i).on(qb, q)

for i in qc:
    circ += H.on(i)

sim = Simulator(backend="mqvector", n_qubits=2 + t)
sim.apply_circuit(circ)

circ_m = Circuit()
circ_m += Measure().on(qs)
circ_m += Measure().on(qb)

res = sim.sampling(circ_m, shots=100000)

res = res.select_keys("q5")
solution0 = np.sqrt(res.data.get("0", 0) / (res.data.get("0", 0) + res.data.get("1", 0)))
solution1 = np.sqrt(res.data.get("1", 0) / (res.data.get("0", 0) + res.data.get("1", 0)))
print(f"Solution of Ax=b by HHL: {solution0, -solution1}")