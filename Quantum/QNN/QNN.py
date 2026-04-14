import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import mindspore as ms
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Adam
from mindspore.train import Accuracy, Model, LossMonitor
from mindspore.dataset import NumpySlicesDataset
from mindspore import ops

from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, X, RZ, RY
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator


def main():
    # Load dataset
    iris_dataset = datasets.load_iris()
    iris_data = iris_dataset.data[:100, :].astype(np.float32)
    y = iris_dataset.target[:100].astype(int)

    alpha = iris_data[:, :3] * iris_data[:, 1:]
    iris_data = np.append(iris_data, alpha, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        iris_data, y, test_size=0.2, random_state=0, shuffle=True
    )

    # Build encoder circuit
    prg = PRGenerator("alpha")
    encoder = Circuit()
    encoder += UN(H, 4)
    for i in range(4):
        encoder += RZ(prg.new()).on(i)
    for j in range(3):
        encoder += X.on(j + 1, j)
        encoder += RZ(prg.new()).on(j + 1)
        encoder += X.on(j + 1, j)

    encoder = encoder.no_grad()

    ansatz = HardwareEfficientAnsatz(
        4, single_rot_gate_seq=[RY], entangle_gate=X, depth=3
    ).circuit

    circuit = encoder.as_encoder() + ansatz.as_ansatz()

    hams = [Hamiltonian(QubitOperator(f"Z{i}")) for i in [2, 3]]

    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("CPU")
    ms.set_seed(1)

    sim = Simulator("mqvector", circuit.n_qubits)
    grad_ops = sim.get_expectation_with_grad(hams, circuit, parallel_worker=5)
    QuantumNet = MQLayer(grad_ops)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opti = Adam(QuantumNet.trainable_params(), learning_rate=0.1)

    model = Model(QuantumNet, loss, opti, metrics={"Acc": Accuracy()})

    train_loader = NumpySlicesDataset(
        {"features": X_train, "labels": y_train}, shuffle=False
    ).batch(5)
    test_loader = NumpySlicesDataset({"features": X_test, "labels": y_test}).batch(5)

    class StepAcc(ms.Callback):
        def __init__(self, model, test_loader):
            self.model = model
            self.test_loader = test_loader
            self.acc = []

        def on_train_step_end(self, run_context):
            self.acc.append(
                self.model.eval(self.test_loader, dataset_sink_mode=False)["Acc"]
            )

    monitor = LossMonitor(16)
    acc = StepAcc(model, test_loader)
    model.train(20, train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)

    predict = np.argmax(ops.Softmax()(model.predict(ms.Tensor(X_test))), axis=1)
    accuracy = model.eval(test_loader, dataset_sink_mode=False)["Acc"]

    print(f"量子神经网络鸢尾花分类任务完成 - 准确率: {accuracy:.4f}")
    print(f"预测类别: {predict}")
    print(f"实际类别: {y_test}")

    return accuracy


if __name__ == "__main__":
    accuracy = main()
    print("\n===== 测试结果 =====")
    print(f"鸢尾花量子神经网络分类准确率: {accuracy}")
