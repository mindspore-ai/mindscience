"""training process"""
import os
import argparse
from random import shuffle
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
import mindspore as ms
from mindspore import dataset as ds
from mindspore import context
import mindspore.nn as nn

from src.utils import adjust_learning_rate
from src.network import DeepLSTM, NetWithLoss, TrainOneStepCell, EvalNet

context.set_context(mode=context.GRAPH_MODE, device_id=0)

parser = argparse.ArgumentParser(description="Train a DeepLSTM")
parser.add_argument(
    "--dataset", required=False, default='data_BoucWen.mat',
    help="using dataset name, select from "
         "[data_BoucWen.mat, data_MRFDBF.mat, data_SanBernardino.mat]")
parser.add_argument(
    "--model", required=False, default='lstm-s',
    help="using model name, use lstm-s or lstm-f")

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
save_dir = data_dir + '/results/{}'.format(args.dataset.replace('.mat', ''))
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

# batch-size & epochs setting
if args.dataset == "data_BoucWen.mat":
    windowsize = 20
    batch_size = 10
    epochs = 50000
elif args.dataset == "data_SanBernardino.mat":
    windowsize = 2
    batch_size = 2
    epochs = 20000
elif args.dataset == "data_MRFDBF.mat":
    windowsize = 10
    batch_size = 10
    epochs = 50000
else:
    raise ValueError("Invalid dataset")

x_pred = np.reshape(x_pred, [x_pred.shape[0], x_pred.shape[1], x_pred.shape[2]])

data_dim = x_train.shape[2]  # number of input features
timesteps = x_train.shape[1]
num_classes = y_train.shape[2]  # number of output features


class DataSource:
    def __init__(self, DataX, DataY):
        self.x = DataX
        self.y = DataY

    def __getitem__(self, idx):
        return self.x[idx].astype(np.float32), self.y[idx].astype(np.float32)

    def __len__(self):
        return len(self.x)


lr_schedule = adjust_learning_rate(base_lr=0.001, decay=0.0001, step_total=50000 * 4)

if args.model == "lstm-s":
    net = DeepLSTM(input_dim=data_dim, num_classes=num_classes, embed_dim=100)
elif args.model == "lstm-f":
    net = DeepLSTM(input_dim=data_dim, num_classes=num_classes, embed_dim=30)
else:
    raise ValueError("Invalid model")
mseloss = nn.MSELoss()
net_with_loss = NetWithLoss(net, mseloss)
adam = nn.Adam(params=net.trainable_params(), learning_rate=lr_schedule)

model = TrainOneStepCell(net_with_loss, adam)

eval_net = EvalNet(net_with_loss)

best_loss = 100
train_loss = []
test_loss = []
history = []

start = time.time()

logger.info("Train total epoch: %d", epochs)
logger.info("---------------train start---------------")
step = 0
for e in range(epochs):
    Ind = list(range(len(x_data_new)))
    shuffle(Ind)
    ratio_split = 0.7
    Ind_train = Ind[0:round(ratio_split * len(x_data_new))]
    Ind_test = Ind[round(ratio_split * len(x_data_new)):]

    x_train = x_data_new[Ind_train]
    y_train = y_data_new[Ind_train]
    x_test = x_data_new[Ind_test]
    y_test = y_data_new[Ind_test]

    train_source = DataSource(DataX=x_train, DataY=y_train)
    train_ds = ds.GeneratorDataset(train_source, column_names=['x', 'y'], shuffle=True)
    train_ds = train_ds.batch(batch_size)

    epoch_train_loss = []
    for i, item in enumerate(train_ds):
        step_begin_time = time.time()
        (x, y) = item[0], item[1]
        n, c, d = x.shape
        h0 = ms.Tensor(np.zeros([1 * 1, n, 100]).astype(np.float32))
        c0 = ms.Tensor(np.zeros([1 * 1, n, 100]).astype(np.float32))
        loss = model(x, h0, c0, y)
        epoch_train_loss.append(loss.asnumpy())
        step_end_time = time.time()
        logger.info('step: %d epoch: %d batch: %d loss:%.9f', step, e, i, loss)
        logger.info('step time is %d', step_end_time - step_begin_time)
        step += 1

    valid_source = DataSource(DataX=x_test, DataY=y_test)
    valid_ds = ds.GeneratorDataset(valid_source, column_names=['x', 'y'], shuffle=True)
    valid_ds = valid_ds.batch(batch_size)
    epoch_test_loss = []
    # evaluate
    for _, item in enumerate(valid_ds):
        (x, y) = item[0], item[1]
        n, c, d = x.shape
        h0 = ms.Tensor(np.zeros([1 * 1, n, 100]).astype(np.float32))
        c0 = ms.Tensor(np.zeros([1 * 1, n, 100]).astype(np.float32))
        loss = eval_net(x, h0, c0, y)
        epoch_test_loss.append(loss.asnumpy())
    score0 = np.mean(epoch_train_loss)
    score = np.mean(epoch_test_loss)

    train_loss.append(score0)
    test_loss.append(score)
    logger.info("train_mse:%.9f test_mse:%.9f", score0, score)

    if test_loss[e] < best_loss:
        best_loss = test_loss[e]
        ms.save_checkpoint(net, os.path.join(save_dir, 'my_best_model.ckpt'))

end = time.time()
running_time = (end - start) / 3600
logger.info("Best test_mse: %.9f", best_loss)
logger.info('Running Time: %d hour', running_time)

# Plot training and testing loss
plt.figure()
plt.plot(np.array(train_loss), 'b-')
plt.plot(np.array(test_loss), 'm-')

# Load the best model
param_dict = ms.load_checkpoint(os.path.join(save_dir, '/my_best_model.ckpt'))
model_best = ms.load_param_into_net(net, param_dict)

x_train = x_data_new[0:len(train_indices[0])]
y_train = y_data_new[0:len(train_indices[0])]
x_test = x_data_new[len(train_indices[0]):]
y_test = y_data_new[len(train_indices[0]):]

# Make predictions using the best model
y_train_pred = model_best.predict(x_train)
y_test_pred = model_best.predict(x_test)
y_pure_preds = model_best.predict(x_pred)

# Reverse map to original magnitude
x_train_orig = x_data[0:len(train_indices[0])]
y_train_orig = y_data[0:len(train_indices[0])]
x_test_orig = x_data[len(train_indices[0]):]
y_test_orig = y_data[len(train_indices[0]):]
x_pred_orig = mat['input_pred_tf']
y_pred_ref_orig = mat['target_pred_tf']

y_train_pred_flatten = np.reshape(y_train_pred, [y_train_pred.shape[0] * y_train_pred.shape[1], y_train_pred.shape[2]])
y_train_pred = scaler_y.inverse_transform(y_train_pred_flatten)
y_train_pred = np.reshape(y_train_pred, [y_train.shape[0], y_train.shape[1], y_train.shape[2]])

for sample in range(len(y_train)):
    plt.figure()
    plt.plot(y_train_orig[sample][windowsize::windowsize, 0], label='True')
    plt.plot(y_train_pred[sample][:, 0], label='Predict')
    plt.title('Training')
    plt.legend()

y_test_pred_flatten = np.reshape(y_test_pred, [y_test_pred.shape[0] * y_test_pred.shape[1], y_test_pred.shape[2]])
y_test_pred = scaler_y.inverse_transform(y_test_pred_flatten)
y_test_pred = np.reshape(y_test_pred, [y_test.shape[0], y_test.shape[1], y_test.shape[2]])

for sample in range(len(y_test)):
    plt.figure()
    plt.plot(y_test_orig[sample][windowsize::windowsize, 0], label='True')
    plt.plot(y_test_pred[sample][:, 0], label='Predict')
    plt.title('Testing')
    plt.legend()

y_pure_preds_flatten = np.reshape(y_pure_preds, [y_pure_preds.shape[0] * y_pure_preds.shape[1], y_pure_preds.shape[2]])
y_pure_preds = scaler_y.inverse_transform(y_pure_preds_flatten)
y_pure_preds = np.reshape(y_pure_preds, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

for sample in range(len(y_pred_ref)):
    plt.figure()
    plt.plot(y_pred_ref_orig[sample][windowsize::windowsize, 0], label='True')
    plt.plot(y_pure_preds[sample][:, 0], label='Predict')
    plt.title('Prediction')
    plt.legend()

scipy.io.savemat(data_dir + 'results/{}/results.mat'.format(args.dataset.replace('.mat', '')),
                 {'y_train': y_train, 'y_train_orig': y_train_orig, 'y_train_pred': y_train_pred,
                  'y_test': y_test, 'y_test_orig': y_test_orig, 'y_test_pred': y_test_pred,
                  'y_pred_ref': y_pred_ref, 'y_pred_ref_orig': y_pred_ref_orig, 'y_pure_preds': y_pure_preds,
                  'x_train': x_train, 'x_test': x_test, 'x_pred': x_pred,
                  'train_indices': train_indices[0], 'test_indices': test_indices[0],
                  'train_loss': train_loss, 'test_loss': test_loss, 'best_loss': best_loss,
                  'windowsize': windowsize, 'running_time': running_time, 'epochs': epochs})
