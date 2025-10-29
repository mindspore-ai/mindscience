'''
P2C2Net for solving the Burgers equation.
'''


import argparse
import json
import os
import signal
import sys
import time
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context, nn, set_seed, ms
from mindspore.amp import auto_mixed_precision
from mindspore.dataset import GeneratorDataset
from src.data import generate_dataset, evl_error, ensure_directories, plot_loss, MyDataset, get_data, UnitGaussianNormalizer
from src.model import P2N2Net, RCNN

set_seed(12345)
np.random.seed(12345)

data_dir = "data"
model_dir = "model"
model_name = "P2C2Net_burgers"
config_dir = "config"
result_dir = "result"
loss_dir = "loss"
error_dir = "error"

def train_stage(train_loader, pretrain_loader,experiment_dir, config, use_ascend, continue_from, compute_dtype):
    '''
    Train the model.
    '''
    milestone_num = config["milestone_num"]
    epochs = config['epochs']
    weight_decay = config["weight_decay"]
    save_every = config["save_every"]
    gamma = config['gamma']
    lr = config['learning_rate']
    model_config = config["model"]
    down = config["down"]
    size = config["size"]
    resolution = size // down
    deno = 1 / size * down
    train_win = config["train_window"]
    delta_t = config["delta_t"]
    modes = model_config['modes']
    depths = model_config['depths']
    h_channels = model_config['hidden_channels']
    p_channels = model_config['projection_channels']

    if milestone_num is not None:
        milestones = [(epochs // milestone_num) * (i + 1) for i in range(milestone_num)]
        learning_rates = [gamma**i * float(lr) for i in range(milestone_num)]
        learning_rate = nn.piecewise_constant_lr(milestones, learning_rates)
    else:
        learning_rate = config['learning_rate']

    start_epoch = 1

    model = RCNN(
        **model_config,
        delta_t=delta_t, compute_dtype=compute_dtype,
        resolution=resolution, deno=deno,
        time_steps=train_win, use_ascend=use_ascend
    )
    print("Model")
    print(model)

    model_param = f"e{epochs}_k{modes}_d{depths}_l{h_channels}_p{p_channels}"
    result_save_dir = os.path.join(experiment_dir, result_dir)
    model_save_dir = os.path.join(result_save_dir, model_dir, model_param, model_name)
    loss_save_dir = os.path.join(result_save_dir, loss_dir, model_param)
    ensure_directories(result_save_dir, model_save_dir, loss_save_dir)

    if continue_from is not None:
        ms.load_checkpoint(model_save_dir + f'checkpoints_{continue_from}.ckpt', model)
        start_epoch = continue_from
        print(f"training continum from checkpoints {continue_from}")

    net = P2N2Net(
        model, learning_rate,
        weight_decay, use_ascend
    )

    model.set_train(True)
    if pretrain_loader:
        pretrain_iters = config["pretrain_iters"]
        pretrain_window = config["pretrain_window"]
        print("pretrain start")
        model.steps = pretrain_window
        for i in range(pretrain_iters):
            print("pretrain ", i)
            pretrain_batch_count = 0
            pretrain_epoch_loss = 0
            for batch_data in pretrain_loader.create_dict_iterator():
                # [ic, uv, nx, ny] , [step, uv, nx, ny]
                pretrain_batch_loss = net(batch_data)
                pretrain_epoch_loss += pretrain_batch_loss
                pretrain_batch_count += 1
                print(f'batch {pretrain_batch_count} loss {pretrain_batch_loss}')
            print("Pretraining epoch Loss: ", pretrain_epoch_loss / pretrain_batch_count)

    start = time.time()
    train_loss_list = []
    print("training start")
    model.steps = train_win
    for epoch in range(start_epoch, 1 + epochs):
        print(f'epoch {epoch} start')
        epoch_start = time.time()
        epoch_loss = 0
        batch_count = 0
        for batch_data in train_loader.create_dict_iterator():
            # [ic, uv, nx, ny] , [step, uv, nx, ny]
            batch_loss = net(batch_data)
            epoch_loss += batch_loss
            batch_count += 1
            print(f'batch {batch_count} loss {batch_loss}')

        epoch_loss = epoch_loss / batch_count
        train_loss_list.append(epoch_loss)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        total_time = epoch_end - start
        print("training epoch Loss: ", epoch_loss, "epoch Time: ", epoch_time, "total Time: ", total_time)

        np.savetxt(loss_save_dir + "/train_loss.txt", train_loss_list)
        if epoch % save_every == 0 or epoch ==  epochs:
            ms.save_checkpoint(
                save_obj=model,
                ckpt_file_name = model_save_dir + f"_checkpoints_{epoch}.ckpt"
            )
            ms.save_checkpoint(
                save_obj=model,
                ckpt_file_name = model_save_dir + ".ckpt"
            )
    plot_loss(train_loss_list, loss_save_dir)
    end = time.time()
    total_time = end - start
    print("training stage end, total time = ", total_time)

def test_stage(
    test_data,
    exp_dir,
    config,
    compute_dtype,
    use_ascend,
    normalizer=None,
    checkpoint=None
):
    '''
    Evaluate the trained model on test data and compute error metrics.
    '''
    model_config = config["model"]
    down = config["down"]
    size = config["size"]
    resolution = size // down
    deno = 1 / size * down
    train_win = config["train_window"]
    inferstep = config["inferstep"]
    delta_t = config["delta_t"]
    epochs = config["epochs"]
    modes = model_config['modes']
    depths = model_config['depths']
    h_channels = model_config['hidden_channels']
    p_channels = model_config['projection_channels']

    model = RCNN(
        **model_config,
        compute_dtype=compute_dtype,
        resolution=resolution,
        deno=deno,
        time_steps=train_win
    )
    model_param = f"e{epochs}_k{modes}_d{depths}_l{h_channels}_p{p_channels}"
    result_save_dir = os.path.join(exp_dir, result_dir)
    model_save_dir = os.path.join(result_save_dir, model_dir, model_param, model_name)
    error_save_dir = os.path.join(result_save_dir, error_dir, model_param)
    ensure_directories(error_save_dir)

    if checkpoint:
        model_save_dir += f"_checkpoints_{checkpoint}"
    if use_ascend:
        auto_mixed_precision(model, 'O1')

    try:
        ms.load_checkpoint(model_save_dir + '.ckpt', model)
        print("Successfully loaded model")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(0)

    model.set_train(False)
    evl_error(
        test_data[:, 0: inferstep],
        model,
        inferstep,
        delta_t,
        resolution,
        error_save_dir,
        compute_dtype,
        normalizer,
    )

def main_function(
    experiment_dir,
    specs_filename,
    continue_from = None,
    trainstage = True,
    teststage = True,
):
    config_path = os.path.join(experiment_dir, config_dir, specs_filename)
    with open(config_path, 'r', encoding='utf-8') as f:
        burgers_config = json.load(f)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    compute_dtype = mstype.float32

    num = burgers_config["num_data"]
    train_win = burgers_config["train_window"]
    timesteps = burgers_config["timesteps"]
    batch_size = burgers_config["batch_size"]
    down = burgers_config["down"]
    nolap = burgers_config["nolap"]
    drop = burgers_config["drop"]
    normalization = burgers_config["normalization"]
    pretrain_iters = burgers_config["pretrain_iters"]
    pretrain_window = burgers_config["pretrain_window"]
    n_train = burgers_config["num_train"]
    n_test = burgers_config["num_test"]

    def signal_handler(sig, frame):  # pylint: disable=unused-argument
        """Handle interrupt signal for graceful shutdown."""

        print("Stopping early...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # [batch, ic, steps, uv, nx, ny]
    all_data = get_data(
        experiment_dir,
        num,
        down
    )

    normalizer = None
    if normalization:
        normalizer = UnitGaussianNormalizer(
            x = all_data,
            normalization_dim = (0, 1, 3, 4)
        )
        all_data = normalizer.encode(all_data)

    train_data = all_data[: n_train]
    test_data = all_data[-n_test:]

    if trainstage:
        print("*****************trainingStage****************")

        all_data_set = generate_dataset(
            train_data[:, :timesteps],
            train_win=train_win,
            icnum=n_train,
            nolap=nolap
        )
        print("all data set shape: ", all_data_set.shape)
        # [400, 5, 2, 26, 26] [ic, times, uv, nx, ny]
        train_dataset = GeneratorDataset(
            source = MyDataset(all_data_set),
            column_names = ['data', 'labels'],
            shuffle=False
        )
        train_loader = train_dataset.batch(batch_size, drop)

        if pretrain_iters != 0 and pretrain_window != 0:
            pretrain_data_set = generate_dataset(
                train_data[:, :timesteps],
                train_win=pretrain_window,
                icnum=n_train,
                nolap=nolap
            )
            pretrain_dataset = GeneratorDataset(
            source = MyDataset(pretrain_data_set),
            column_names = ['data', 'labels'],
            shuffle=False
            )
            pretrain_loader = pretrain_dataset.batch(batch_size, drop)
        else:
            pretrain_loader=None

        train_stage(
            train_loader,
            pretrain_loader,
            experiment_dir,
            burgers_config,
            use_ascend,
            continue_from,
            compute_dtype,
        )
    if teststage:
        print("*****************testingStage****************")
        test_stage(
            test_data,
            experiment_dir,
            burgers_config,
            compute_dtype,
            use_ascend,
            normalizer,
            checkpoint=continue_from
        )

def parse_args():
    """parse input args"""
    parser = argparse.ArgumentParser(description="burgers train")
    parser.add_argument(
        "--experiment",
        type=str,
        dest="experiment_directory",
        required=True,
        help="The experiment directory. "
    )
    parser.add_argument(
        "--config_filename",
        type=str,
        dest="config_filename",
        default="burgers.json",
        help="The filename of experiment specifications"
    )
    parser.add_argument(
        "--continue",
        type=int,
        dest="continue_from",
        default=None,
        help="A snapshot to continue from.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        dest="mode",
        default="PYNATIVE",
        choices=["GRAPH", "PYNATIVE"],
        help="Running in GRAPH_MODE OR PYNATIVE_MODE"
    )
    parser.add_argument(
        "--device_target",
        type=str,
        dest="device_target",
        default="Ascend",
        choices=["GPU", "Ascend"],
        help="The target device to run, support 'Ascend', 'GPU'"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        dest="device_id",
        default=0,
        help="ID of the target device"
    )
    parser.add_argument(
        "--train_stage",
        type=bool,
        dest="train_stage",
        default=True,
        help="Whether to run training stage"
    )
    parser.add_argument(
        "--test_stage",
        type=bool,
        dest="train_stage",
        default=True,
        help="Whether to run test stage"
    )
    input_args = parser.parse_args()
    return input_args

if __name__ == '__main__':
    args = parse_args()
    ms.context.set_context(
        mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE)
    ms.set_device(args.device_target, args.device_id)
    ms.set_recursion_limit(99999999)
    main_function(
        experiment_dir = args.experiment_directory,
        specs_filename = args.config_filename,
        continue_from = args.continue_from,
        trainstage = args.train_stage,
        teststage = args.test_stage,
    )
