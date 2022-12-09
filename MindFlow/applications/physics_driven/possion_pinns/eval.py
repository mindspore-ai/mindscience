"""Evaluating."""
from argparse import ArgumentParser

import mindspore as ms
from mindspore import nn
from mindflow.common import L2
from mindflow.utils import load_yaml_config

from src.model import create_model
from src.possion import AnalyticSolution
from src.dataset import create_dataset


class WithEvalCell(nn.Cell):
    """Cell for evaluating models."""
    def __init__(self, model, solution):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self.model = model
        self.solution = solution

    def construct(self, x_domain, x_bc):
        y_pred_domain = self.model(x_domain)
        y_test_domain = self.solution(x_domain)

        y_pred_bc = self.model(x_bc)
        y_test_bc = self.solution(x_bc)
        return (y_pred_domain, y_test_domain), (y_pred_bc, y_test_bc)


def test(geom_name, checkpoint, file_cfg, n_samps, batch_size):
    """Evaluate a model."""
    # Create the dataset
    config = load_yaml_config(file_cfg)
    config['data']['domain']['size'] = n_samps
    config['data']['BC']['size'] = n_samps
    config['batch_size'] = batch_size
    (_, ds_test), (_, n_dim) = create_dataset(geom_name, config)

    # Create the model
    model = create_model(**config['model'][f'{n_dim}d'])
    checkpoint = ms.load_checkpoint(args.checkpoint)
    ms.load_param_into_net(model, checkpoint)

    # Create solution
    solution = AnalyticSolution(n_dim)

    # Evaluate
    eval_net = WithEvalCell(model, solution)
    eval_net.set_train(False)
    metric_domain = L2()
    metric_bc = L2()

    for x_domain, x_bc in ds_test:
        (y_pred_domain, y_test_domain), (y_pred_bc, y_test_bc) = eval_net(x_domain, x_bc)
        metric_domain.update(y_pred_domain.asnumpy(), y_test_domain.asnumpy())
        metric_bc.update(y_pred_bc.asnumpy(), y_test_bc.asnumpy())

    print("Relative L2 error (domain): {:.4f}".format(metric_domain.eval()))
    print("Relative L2 error (bc): {:.4f}".format(metric_bc.eval()))
    print("")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('geom_name', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--file_cfg', default='possion_cfg.yaml')
    parser.add_argument('--n_samps', default=5000, type=int)
    parser.add_argument('--batch_size', default=5000, type=int)
    args = parser.parse_args()

    test(args.geom_name, args.checkpoint, args.file_cfg, args.n_samps, args.batch_size)
