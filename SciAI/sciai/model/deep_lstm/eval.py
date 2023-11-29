"""evaluation process"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
import mindspore as ms
from mindspore import context

from src.network import DeepLSTM

context.set_context(mode=context.GRAPH_MODE, device_id=0)

parser = argparse.ArgumentParser(description="Evaluate a DeepLSTM")
parser.add_argument(
    "--dataset", required=False, default='data_BoucWen.mat',
    help="using dataset name, please select from "
         "[data_BoucWen.mat, data_MRFDBF.mat, data_SanBernardino.mat]")
parser.add_argument(
    "--model", required=False, default='lstm-s',
    help="using model name, use lstm-s or lstm-f")

args = parser.parse_args()


def generate_data(x_data0, y_data0, window_size=50):
    """Process full sequence to stacked sequence"""
    x_new_temp = []
    y_new_temp = []
    for x_temp, y_temp in zip(x_data0, y_data0):
        x_new = []
        y_new = []
        try:
            for jj in range(int(np.floor(len(x_temp) / window_size))):
                x_new.append(x_temp[jj * window_size:(jj + 1) * window_size])
                y_new.append(y_temp[(jj + 1) * window_size - 1, :])
        except ZeroDivisionError:
            pass

        x_new_temp.append(np.array(x_new))
        y_new_temp.append(np.array(y_new))

    x_data_new0 = np.array(x_new_temp)
    y_data_new0 = np.array(y_new_temp)

    return x_data_new0, y_data_new0


# Load data
data_dir = os.getcwd().replace('\\', '/')
data_name = 'Bouc-Wen (LSTM-f)'
save_dir = os.path.join(data_dir, 'results/Bouc-Wen (LSTM-f)')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
mat = scipy.io.loadmat(data_dir + '/data/{}'.format(args.dataset))

if args.dataset == "data_BoucWen.mat":
    x_data = mat['input_tf']
    y_data = mat['target_tf']
    train_indices = mat['trainInd'] - 1
    test_indices = mat['valInd'] - 1

    # Scale data
    x_data_flatten = np.reshape(x_data, [x_data.shape[0] * x_data.shape[1], 1])
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(x_data_flatten)
    x_data_flatten_map = scaler_x.transform(x_data_flatten)
    x_data_map = np.reshape(x_data_flatten_map, [x_data.shape[0], x_data.shape[1], 1])

    y_data_flatten = np.reshape(y_data, [y_data.shape[0] * y_data.shape[1], y_data.shape[2]])
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.fit(y_data_flatten)
    y_data_flatten_map = scaler_y.transform(y_data_flatten)
    y_data_map = np.reshape(y_data_flatten_map, [y_data.shape[0], y_data.shape[1], y_data.shape[2]])

    # Unknown data
    x_pred = mat['input_pred_tf']
    y_pred_ref = mat['target_pred_tf']

    # Scale data
    x_pred_flatten = np.reshape(x_pred, [x_pred.shape[0] * x_pred.shape[1], 1])
    x_pred_flatten_map = scaler_x.transform(x_pred_flatten)
    x_pred_map = np.reshape(x_pred_flatten_map, [x_pred.shape[0], x_pred.shape[1], 1])

    y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0] * y_pred_ref.shape[1], y_pred_ref.shape[2]])
    y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
    y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

    windowsize = 20
    x_data_new, y_data_new = generate_data(x_data_map, y_data_map, windowsize)

    x_data_new = np.reshape(x_data_new, [x_data_new.shape[0], x_data_new.shape[1], x_data_new.shape[2]])

    x_train = x_data_new[0:len(train_indices[0])]
    y_train = y_data_new[0:len(train_indices[0])]
    x_test = x_data_new[len(train_indices[0]):]
    y_test = y_data_new[len(train_indices[0]):]

    x_pred, y_pred_ref = generate_data(x_pred_map, y_pred_ref_map, windowsize)
    x_pred = np.reshape(x_pred, [x_pred.shape[0], x_pred.shape[1], x_pred.shape[2]])
elif args.dataset == "data_SanBernardino.mat":
    x_data = mat['input_tf']
    y_data = mat['target_tf']
    train_indices = mat['trainInd'] - 1
    test_indices = mat['valInd'] - 1

    x_data = np.reshape(x_data, [x_data.shape[0], x_data.shape[1], 1])

    # Scale data
    x_data_flatten = np.reshape(x_data, [x_data.shape[0] * x_data.shape[1], 1])
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(x_data_flatten)
    x_data_flatten_map = scaler_x.transform(x_data_flatten)
    x_data_map = np.reshape(x_data_flatten_map, [x_data.shape[0], x_data.shape[1], 1])

    y_data_flatten = np.reshape(y_data, [y_data.shape[0] * y_data.shape[1], y_data.shape[2]])
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.fit(y_data_flatten)
    y_data_flatten_map = scaler_y.transform(y_data_flatten)
    y_data_map = np.reshape(y_data_flatten_map, [y_data.shape[0], y_data.shape[1], y_data.shape[2]])

    # Unknown data
    x_pred = mat['input_pred_tf']
    y_pred_ref = mat['target_pred_tf']
    x_pred = np.reshape(x_pred, [x_pred.shape[0], x_pred.shape[1], 1])

    # Scale data
    x_pred_flatten = np.reshape(x_pred, [x_pred.shape[0] * x_pred.shape[1], 1])
    x_pred_flatten_map = scaler_x.transform(x_pred_flatten)
    x_pred_map = np.reshape(x_pred_flatten_map, [x_pred.shape[0], x_pred.shape[1], 1])

    y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0] * y_pred_ref.shape[1], y_pred_ref.shape[2]])
    y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
    y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

    windowsize = 2
    x_data_new, y_data_new = generate_data(x_data_map, y_data_map, windowsize)

    x_data_new = np.reshape(x_data_new, [x_data_new.shape[0], x_data_new.shape[1], x_data_new.shape[2]])

    x_train = x_data_new[0:len(train_indices[0])]
    y_train = y_data_new[0:len(train_indices[0])]
    x_test = x_data_new[len(train_indices[0]):]
    y_test = y_data_new[len(train_indices[0]):]
elif args.dataset == "data_MRFDBF.mat":
    train_indices = mat['trainInd'] - 1
    test_indices = mat['valInd'] - 1
    pred_indices = mat['testInd'] - 1
    x_data = mat['input_tf'][np.concatenate([train_indices[0], test_indices[0]])]
    y_data = mat['target_tf'][np.concatenate([train_indices[0], test_indices[0]])]

    # Scale data
    x_data_flatten = np.reshape(x_data, [x_data.shape[0] * x_data.shape[1], 1])
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(x_data_flatten)
    x_data_flatten_map = scaler_x.transform(x_data_flatten)
    x_data_map = np.reshape(x_data_flatten_map, [x_data.shape[0], x_data.shape[1], 1])

    y_data_flatten = np.reshape(y_data, [y_data.shape[0] * y_data.shape[1], y_data.shape[2]])
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.fit(y_data_flatten)
    y_data_flatten_map = scaler_y.transform(y_data_flatten)
    y_data_map = np.reshape(y_data_flatten_map, [y_data.shape[0], y_data.shape[1], y_data.shape[2]])

    # Unknown data
    x_pred = mat['input_tf'][pred_indices[0]]
    y_pred_ref = mat['target_tf'][pred_indices[0]]

    # Scale data
    x_pred_flatten = np.reshape(x_pred, [x_pred.shape[0] * x_pred.shape[1], 1])
    x_pred_flatten_map = scaler_x.transform(x_pred_flatten)
    x_pred_map = np.reshape(x_pred_flatten_map, [x_pred.shape[0], x_pred.shape[1], 1])

    y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0] * y_pred_ref.shape[1], y_pred_ref.shape[2]])
    y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
    y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

    # Generate stacked data
    windowsize = 10
    x_data_new, y_data_new = generate_data(x_data_map, y_data_map, windowsize)
    x_data_new = np.reshape(x_data_new, [x_data_new.shape[0], x_data_new.shape[1], x_data_new.shape[2]])

    x_train = x_data_new[0:len(train_indices[0])]
    y_train = y_data_new[0:len(train_indices[0])]
    x_test = x_data_new[len(train_indices[0]):]
    y_test = y_data_new[len(train_indices[0]):]
else:
    raise ValueError("Invalid dataset")

x_pred = np.reshape(x_pred, [x_pred.shape[0], x_pred.shape[1], x_pred.shape[2]])

data_dim = x_train.shape[2]  # number of input features
timesteps = x_train.shape[1]
num_classes = y_train.shape[2]  # number of output features


if args.model == "lstm-s":
    net = DeepLSTM(input_dim=data_dim, num_classes=num_classes, embed_dim=100)
elif args.model == "lstm-f":
    net = DeepLSTM(input_dim=data_dim, num_classes=num_classes, embed_dim=30)
else:
    raise ValueError("Invalid model")

# Load the best model
param_dict = ms.load_checkpoint(save_dir + '/{}_best_model.ckpt'.format(data_name))
ms.load_param_into_net(net, param_dict)

x_train = x_data_new[0:len(train_indices[0])]
y_train = y_data_new[0:len(train_indices[0])]
x_test = x_data_new[len(train_indices[0]):]
y_test = y_data_new[len(train_indices[0]):]

n, c, d = x_train.shape
h0_x_train = ms.Tensor(np.zeros([1 * 1, n, 100]).astype(np.float32))
c0_x_train = ms.Tensor(np.zeros([1 * 1, n, 100]).astype(np.float32))
n, c, d = x_test.shape
h0_x_test = ms.Tensor(np.zeros([1 * 1, n, 100]).astype(np.float32))
c0_x_test = ms.Tensor(np.zeros([1 * 1, n, 100]).astype(np.float32))
n, c, d = x_pred.shape
h0_x_pred = ms.Tensor(np.zeros([1 * 1, n, 100]).astype(np.float32))
c0_x_pred = ms.Tensor(np.zeros([1 * 1, n, 100]).astype(np.float32))

# Make predictions using the best model
y_train_pred = net(ms.Tensor.from_numpy(x_train.astype(np.float32)), h0_x_train, c0_x_train).asnumpy()
y_test_pred = net(ms.Tensor.from_numpy(x_test.astype(np.float32)), h0_x_test, c0_x_test).asnumpy()
y_pure_preds = net(ms.Tensor.from_numpy(x_pred.astype(np.float32)), h0_x_pred, c0_x_pred).asnumpy()

# Reverse map to original magnitude
if args.dataset == "data_BoucWen.mat":
    x_train_orig = x_data[0:len(train_indices[0])]
    y_train_orig = y_data[0:len(train_indices[0])]
    x_test_orig = x_data[len(train_indices[0]):]
    y_test_orig = y_data[len(train_indices[0]):]
    x_pred_orig = mat['input_pred_tf']
    y_pred_ref_orig = mat['target_pred_tf']

    y_train_pred_flatten = np.reshape(y_train_pred,
                                      [y_train_pred.shape[0] * y_train_pred.shape[1], y_train_pred.shape[2]])
    y_train_pred = scaler_y.inverse_transform(y_train_pred_flatten)
    y_train_pred = np.reshape(y_train_pred, [y_train.shape[0], y_train.shape[1], y_train.shape[2]])
elif args.dataset == "data_SanBernardino.mat":
    x_train_orig = x_data[0:len(train_indices[0])]
    y_train_orig = y_data[0:len(train_indices[0])]
    x_test_orig = x_data[len(train_indices[0]):]
    y_test_orig = y_data[len(train_indices[0]):]
    x_pred_orig = mat['input_pred_tf']
    y_pred_ref_orig = mat['target_pred_tf']

    y_train_pred_flatten = np.reshape(y_train_pred,
                                      [y_train_pred.shape[0] * y_train_pred.shape[1], y_train_pred.shape[2]])
    y_train_pred = scaler_y.inverse_transform(y_train_pred_flatten)
    y_train_pred = np.reshape(y_train_pred, [y_train.shape[0], y_train.shape[1], y_train.shape[2]])
elif args.dataset == "data_MRFDBF.mat":
    x_train_orig = mat['input_tf'][train_indices[0]]
    y_train_orig = mat['target_tf'][train_indices[0]]
    x_test_orig = mat['input_tf'][test_indices[0]]
    y_test_orig = mat['target_tf'][test_indices[0]]
    x_pred_orig = mat['input_tf'][pred_indices[0]]
    y_pred_ref_orig = mat['target_tf'][pred_indices[0]]

    y_train_pred_flatten = np.reshape(y_train_pred,
                                      [y_train_pred.shape[0] * y_train_pred.shape[1], y_train_pred.shape[2]])
    y_train_pred = scaler_y.inverse_transform(y_train_pred_flatten)
    y_train_pred = np.reshape(y_train_pred, [y_train.shape[0], y_train.shape[1], y_train.shape[2]])

    for sample in range(len(y_train)):
        plt.figure()
        plt.plot(y_train_orig[sample][0::windowsize, 0], label='True')
        plt.plot(y_train_pred[sample][:, 0], label='Predict')
        plt.title('Training')
        plt.legend()
        plt.show()

    y_test_pred_flatten = np.reshape(y_test_pred, [y_test_pred.shape[0] * y_test_pred.shape[1], y_test_pred.shape[2]])
    y_test_pred = scaler_y.inverse_transform(y_test_pred_flatten)
    y_test_pred = np.reshape(y_test_pred, [y_test.shape[0], y_test.shape[1], y_test.shape[2]])
else:
    raise ValueError("Invalid dataset")

for sample in range(len(y_train)):
    plt.figure()
    plt.plot(y_train_orig[sample][windowsize::windowsize, 0], label='True')
    plt.plot(y_train_pred[sample][:, 0], label='Predict')
    plt.title('Training')
    plt.legend()
    plt.show()

y_test_pred_flatten = np.reshape(y_test_pred, [y_test_pred.shape[0] * y_test_pred.shape[1], y_test_pred.shape[2]])
y_test_pred = scaler_y.inverse_transform(y_test_pred_flatten)
y_test_pred = np.reshape(y_test_pred, [y_test.shape[0], y_test.shape[1], y_test.shape[2]])

for sample in range(len(y_test)):
    plt.figure()
    plt.plot(y_test_orig[sample][windowsize::windowsize, 0], label='True')
    plt.plot(y_test_pred[sample][:, 0], label='Predict')
    plt.title('Testing')
    plt.legend()
    plt.show()

y_pure_preds_flatten = np.reshape(y_pure_preds, [y_pure_preds.shape[0] * y_pure_preds.shape[1], y_pure_preds.shape[2]])
y_pure_preds = scaler_y.inverse_transform(y_pure_preds_flatten)
y_pure_preds = np.reshape(y_pure_preds, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

for sample in range(len(y_pred_ref)):
    plt.figure()
    plt.plot(y_pred_ref_orig[sample][windowsize::windowsize, 0], label='True')
    plt.plot(y_pure_preds[sample][:, 0], label='Predict')
    plt.title('Prediction')
    plt.legend()
    plt.show()
