# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

'''
This file defines and trains a 3D_CAE model
'''
import numpy as np
from sklearn.model_selection import train_test_split
from mindspore import nn, context
from mindspore.train.callback import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from mindspore.train import Model
import mindspore.dataset as ds

from mindspore.train.callback import CheckpointConfig, LossMonitor
from mindspore.train.callback import Callback
from tqdm import tqdm

# Model architecture
class AutoEncoder(nn.Cell):
    '''Definition of AutoEncoder'''
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.SequentialCell([
            nn.Conv3d(1, 32, kernel_size=3, pad_mode='pad', padding=1),  # Output shape: [batch_size, 32, 12, 12, 12]
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, pad_mode='pad', padding=1),  # Output shape: [batch_size, 64, 12, 12, 12]
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, pad_mode='pad', padding=1),  # Output shape: [batch_size, 128, 12, 12, 12]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, pad_mode='same')  # Output shape: [batch_size, 128, 6, 6, 6]
        ])

        self.decoder = nn.SequentialCell([
            nn.Upsample(scale_factor=(2.0, 2.0, 2.0), mode='nearest'),  # Upsample to [batch_size, 128, 12, 12, 12]
            nn.Conv3d(128, 64, kernel_size=3, pad_mode='pad', padding=1),  # Output shape: [batch_size, 64, 12, 12, 12]
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, pad_mode='pad', padding=1),   # Output shape: [batch_size, 32, 12, 12, 12]
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, pad_mode='pad', padding=1),    # Output shape: [batch_size, 1, 12, 12, 12]
        ])

    def construct(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class TQDMProgressBar(Callback):
    '''add progress bar while training the model'''
    def __init__(self, tot_steps_per_epoch, tot_epochs):
        super(TQDMProgressBar, self).__init__()
        self.total_steps_per_epoch = tot_steps_per_epoch
        self.total_epochs = tot_epochs
        self.progress_bar = None
        self.current_epoch = 0

    def on_train_epoch_begin(self):
        self.current_epoch += 1
        if self.progress_bar is not None:
            self.progress_bar.close()  # Close the previous bar
        self.progress_bar = tqdm(total=self.total_steps_per_epoch, \
                                 desc=f"Epoch {self.current_epoch}/{self.total_epochs}",\
                                      ncols=100, unit=" step", leave=True)

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()  # Get the loss value
        self.progress_bar.set_postfix_str(f"loss: {loss:.6f}")  # Append loss value to progress bar
        self.progress_bar.update(1)  # Update the progress bar
        tqdm.write(f"Epoch: {self.current_epoch}, Step: {cb_params.cur_step_num}, Loss: {loss:.6f}", end='\r')

    def on_train_epoch_end(self,):
        self.progress_bar.close()

class CustomLossMonitor(LossMonitor):
    '''自定义损失监控'''
    def __init__(self):
        super(CustomLossMonitor, self)

    def step_end(self, run_context):
        """Override to avoid multi-line printing."""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        tqdm.write(f"\rEpoch: {cb_params.cur_epoch_num}, \
            Step: {cb_params.cur_step_num}, Loss: {loss.asnumpy():.6f}", end='')

def get_cae_data(args):
    '''get cae data'''
    # GPU Configuration
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
    context.set_context(max_call_depth=10000)

    # Import
    # matrix = np.load('data/3D_CAE_Train.npy', allow_pickle=True)
    matrix = np.load(f"{args.load_data_path}/3D_CAE_Train.npy", allow_pickle=True)
    ran = range(len(matrix))
    x = matrix.reshape(17835, 12, 12, 12, 1)

    # Convert the data type to Float32
    x = x.astype(np.float32)

    # Adjust the shape to be [batch_size, channels, depth, height, width]
    x = np.transpose(x, (0, 4, 1, 2, 3))  # Now X is [batch_size, channels, depth, height, width]

    ran = np.arange(len(matrix))  # Create labels or indices, similar to TensorFlow code

    # Split
    x_train, x_test, _, _ = train_test_split(x, ran, test_size=0.2, random_state=1)

    return x_train, x_test

def get_cae_model(x_train, x_test, args):
    '''get cae model'''
    # Model parameters
    # b_size = 64
    b_size = args.batch_size
    # k_size = 4
    # f_size = 60
    # lr = 0.000753014797772
    lr = args.lr

    # Create MindSpore training & testing dataset
    train_data = ds.NumpySlicesDataset((x_train, x_train), shuffle=True)
    test_data = ds.NumpySlicesDataset((x_test, x_test), shuffle=False)

    # Apply batching to the datasets
    train_data = train_data.batch(batch_size=b_size, drop_remainder=True)
    test_data = test_data.batch(batch_size=b_size, drop_remainder=True)

    # Create Model
    autoencoder = AutoEncoder()

    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(autoencoder.trainable_params(), learning_rate=lr)

    # Configure the Checkpoint
    config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=5)
    # ModelCheckpoint callback
    mc = ModelCheckpoint(prefix='3D_CAE_model', directory='model', config=config_ck)

    # LossMonitor callback to print loss values
    # ls = LossMonitor()

    # Callbacks
    re = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10)
    es = EarlyStopping(monitor='loss', mode='min', patience=20)

    # Configure training model
    model = Model(autoencoder, loss_fn=loss_fn, optimizer=optimizer)

    return model, train_data, es, mc, re

def train_cae_model(model, train_data, es, mc, re, args):
    '''train cae model'''
    # Number of steps per epoch
    total_steps_per_epoch = train_data.get_dataset_size()
    # Total number of epochs
    # total_epochs = 500
    total_epochs = args.epochs

    # 初始化进度条回调
    tqdm_callback = TQDMProgressBar(tot_steps_per_epoch=total_steps_per_epoch, tot_epochs=total_epochs)

    # 自定义损失监控
    custom_loss_monitor = CustomLossMonitor()

    # 训练模型
    model.train(total_epochs, train_data, \
                callbacks=[es, mc, re, custom_loss_monitor, tqdm_callback], dataset_sink_mode=False)
