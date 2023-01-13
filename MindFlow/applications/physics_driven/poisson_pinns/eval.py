"""Evaluating."""
from argparse import ArgumentParser

import mindspore as ms
import mindflow as mf

from src.model import create_model
from src.poisson import AnalyticSolution
from src.dataset import create_dataset


def calculate_l2_error(model, ds_test, n_dim):
    """Calculate the relative L2 error."""
    # Create solution
    solution = AnalyticSolution(n_dim)

    # Evaluate
    metric_domain = mf.L2()
    metric_bc = mf.L2()
    for x_domain, x_bc in ds_test:
        y_pred_domain = model(x_domain)
        y_test_domain = solution(x_domain)

        y_pred_bc = model(x_bc)
        y_test_bc = solution(x_bc)
        metric_domain.update(y_pred_domain.asnumpy(), y_test_domain.asnumpy())
        metric_bc.update(y_pred_bc.asnumpy(), y_test_bc.asnumpy())

    print("Relative L2 error (domain): {:.4f}".format(metric_domain.eval()))
    print("Relative L2 error (bc): {:.4f}".format(metric_bc.eval()))
    print("")


def test(geom_name, checkpoint, file_cfg, n_samps):
    """Evaluate a model."""
    # Create the dataset
    config = mf.load_yaml_config(file_cfg)
    ds_test_, n_dim_ = create_dataset(geom_name, config, n_samps)

    # Load the model
    model_ = create_model(**config['model'][f'{n_dim_}d'])
    checkpoint = ms.load_checkpoint(checkpoint)
    ms.load_param_into_net(model_, checkpoint)

    # Evaluate the model
    calculate_l2_error(model_, ds_test_, n_dim_)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('geom_name', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--file_cfg', default='poisson_cfg.yaml')
    parser.add_argument('--n_samps', default=5000, type=int)
    args = parser.parse_args()

    test(args.geom_name, args.checkpoint, args.file_cfg, args.n_samps)
