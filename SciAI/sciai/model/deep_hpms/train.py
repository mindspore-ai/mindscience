"""deep hpms train"""
import os

import mindspore as ms

from sciai.context import init_project
from sciai.utils import amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.plot import plot_train
from src.process import load_data, prepare


@print_time("train")
def main(args, problem):
    """main module"""
    dtype = amp2datatype(args.amp_level)
    x_sol_star_, exact_sol, t_sol, x_sol, tensors = load_data(args, dtype)
    lb_idn, ub_idn, lb_sol, ub_sol, \
    t_train, x_train, u_train, \
    t_idn_star, x_idn_star, u_idn_star, \
    x_f_train_, tb_train, x0_train, u0_train, \
    t_sol_star, x_sol_star, u_sol_star = tensors

    model = problem(args.u_layers, args.pde_layers, args.layers, lb_idn, ub_idn, lb_sol, ub_sol)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, model)
    u_pred = train(model, args, x_f_train_, lb_sol, t_idn_star, t_sol_star, t_train, tb_train, u0_train, u_idn_star,
                   u_sol_star, u_train, ub_sol, x0_train, x_idn_star, x_sol_star, x_train)
    if args.save_ckpt:
        ms.save_checkpoint(model, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
    if args.save_fig:
        plot_train(exact_sol, t_sol, x_sol, x_sol_star_, lb_sol.asnumpy(), u_pred, ub_sol.asnumpy(), args)


def train(model, args, *inputs):
    """train module"""
    x_f_train_, lb_sol, t_idn_star, t_sol_star, t_train, tb_train, u0_train, u_idn_star, u_sol_star, \
    u_train, ub_sol, x0_train, x_idn_star, x_sol_star, x_train = inputs
    model.idn_u_train(args, t_train, x_train, u_train)
    model.idn_f_train(args, t_train, x_train)
    model.eval_idn(t_idn_star, x_idn_star, u_idn_star)
    model.sol_train(x0_train, u0_train, tb_train, x_f_train_, lb_sol, ub_sol, args)
    u_pred = model.eval_sol(t_sol_star, u_sol_star, x_sol_star, t_idn_star, x_idn_star, u_idn_star)
    return u_pred


if __name__ == "__main__":
    args_problem = prepare()
    init_project(args=args_problem[0])
    main(*args_problem)
