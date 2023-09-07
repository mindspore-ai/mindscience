"""gppinns eval"""
import mindspore as ms
import numpy as np

from sciai.common import Sampler
from sciai.context import init_project
from sciai.utils import print_log, data_type_dict_amp
from sciai.utils.python_utils import print_time
from src.network import Helmholtz2D, HelmholtzEqn
from src.plot import plot, plot_elements_namedtuple
from src.process import generate_test_data, prepare


@print_time("eval")
def main(args):
    # Define Helmholtz Equation
    helm = HelmholtzEqn(a1=1, a2=4, lam=1.0)

    # Domain boundaries
    dom_coords = np.array([[-1.0, -1.0], [1.0, 1.0]])

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, helm.f, name='Forcing')

    # build and load Helmholtz model
    dtype = data_type_dict_amp.get(args.amp_level, ms.float32)
    model = Helmholtz2D(args.layers, res_sampler, 1.0, args.method, dtype)
    if dtype == ms.float16:
        model.to_float(ms.float16)
    print_log(f"model {model.model} built successfully")
    ms.load_checkpoint(args.load_ckpt_path, model)

    # Test data
    x1, x2, x_star = generate_test_data(dom_coords)

    # Evaluate
    u_pred, u_star = model.evaluate(helm, x_star)

    if args.save_fig:
        elements = plot_elements_namedtuple(args, model, data=(x_star, u_star, u_pred), test_data=(x1, x2))
        plot(elements)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
