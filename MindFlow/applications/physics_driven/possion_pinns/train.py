"""Training."""
from argparse import ArgumentParser

from mindspore import context, nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindflow.loss import Constraints, MTLWeightedLossCell
from mindflow.solver import Solver
from mindflow.common import LossAndTimeMonitor
from mindflow.utils import load_yaml_config

from src.model import create_model
from src.lr_scheduler import OneCycleLR
from src.dataset import create_dataset
from src.possion import Possion


context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU", device_id=0)


def train(geom_name, file_cfg, ckpt_dir, n_epochs):
    """Train a model."""
    # Load config
    config = load_yaml_config(file_cfg)

    # Create the dataset
    (dataset, ds_train), (steps_per_epoch, n_dim) = create_dataset(geom_name, config)

    # Create the model
    model = create_model(**config['model'][f'{n_dim}d'])

    # Create the loss and the optimizer
    mtl = MTLWeightedLossCell(num_losses=dataset.num_dataset)
    params = model.trainable_params() + mtl.trainable_params()
    learning_rate = OneCycleLR(total_steps=steps_per_epoch*n_epochs, **config['optimizer'])
    opt = nn.Adam(params, learning_rate=learning_rate)

    # Create constraints
    domain_name, bc_name = dataset.columns_list
    problem_list = [Possion(model, n_dim, domain_name, bc_name) \
        for _ in range(dataset.num_dataset)]
    train_constraints = Constraints(dataset, problem_list)

    # Create the solver
    solver = Solver(
        model,
        optimizer=opt,
        train_constraints=train_constraints,
        loss_fn='l2_loss',
        mtl_weighted_cell=mtl,
    )

    # Create callbacks
    loss_time_callback = LossAndTimeMonitor(steps_per_epoch)
    prefix = f"{geom_name}_{n_dim}d"
    config = CheckpointConfig(
        save_checkpoint_steps=config['save_checkpoint_steps'],
        keep_checkpoint_max=config['keep_checkpoint_max'])
    ckpt_callback = ModelCheckpoint(prefix=prefix, directory=ckpt_dir, config=config)
    callbacks = [loss_time_callback, ckpt_callback]

    # Train
    solver.train(n_epochs, ds_train, callbacks=callbacks, dataset_sink_mode=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('geom_name', type=str)
    parser.add_argument('--file_cfg', default='possion_cfg.yaml')
    parser.add_argument('--ckpt_dir', default='./')
    parser.add_argument('--n_epochs', default=1, type=int)
    args = parser.parse_args()

    train(args.geom_name, args.file_cfg, args.ckpt_dir, args.n_epochs)
