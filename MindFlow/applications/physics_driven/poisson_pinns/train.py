"""Training."""
import os
import time
from argparse import ArgumentParser

from mindspore import context, save_checkpoint, nn, ops, jit
from mindflow import load_yaml_config

from src.model import create_model
from src.lr_scheduler import OneCycleLR
from src.dataset import create_dataset
from src.poisson import Poisson


context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU", device_id=0)


def train(geom_name, file_cfg, ckpt_dir, n_epochs):
    """Train a model."""
    # Load config
    config = load_yaml_config(file_cfg)

    # Create the dataset
    ds_train, n_dim = create_dataset(geom_name, config)

    # Create the model
    model = create_model(**config['model'][f'{n_dim}d'])

    # Create the problem and optimizer
    problem = Poisson(model, n_dim)
    params = model.trainable_params() + problem.loss_fn.trainable_params()
    steps_per_epoch = config['data']['domain']['size']//config['batch_size']
    learning_rate = OneCycleLR(total_steps=steps_per_epoch*n_epochs, **config['optimizer'])
    opt = nn.Adam(params, learning_rate=learning_rate)

    # Create
    grad_fn = ops.value_and_grad(problem.get_loss, None, opt.parameters, has_aux=False)

    @jit
    def train_step(pde_data, bc_data):
        loss, grads = grad_fn(pde_data, bc_data)
        loss = ops.depend(loss, opt(grads))
        return loss

    def train_epoch(model, dataset, i_epoch):
        n_step = dataset.get_dataset_size()
        model.set_train()
        for i_step, (pde_data, bc_data) in enumerate(dataset):
            local_time_beg = time.time()
            loss = train_step(pde_data, bc_data)

            if i_step%50 == 0 or i_step + 1 == n_step:
                print("\repoch: {}, loss: {:>f}, time elapsed: {:.1f}ms [{}/{}]".format(
                    i_epoch, float(loss), (time.time() - local_time_beg)*1000, i_step + 1, n_step))

    keep_ckpt_max = config['keep_checkpoint_max']

    for i_epoch in range(n_epochs):
        train_epoch(model, ds_train, i_epoch)

        # Save last checkpoints
        save_name = os.path.join(ckpt_dir, "{}_{}d_{}.ckpt".format(
            geom_name, n_dim, i_epoch%keep_ckpt_max))
        save_checkpoint(model, save_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('geom_name', type=str)
    parser.add_argument('--file_cfg', default='poisson_cfg.yaml')
    parser.add_argument('--ckpt_dir', default='./')
    parser.add_argument('--n_epochs', default=1, type=int)
    args = parser.parse_args()

    time_beg = time.time()
    train(args.geom_name, args.file_cfg, args.ckpt_dir, args.n_epochs)
    print("End-to-End total time: {:.1f} s".format(time.time() - time_beg))
