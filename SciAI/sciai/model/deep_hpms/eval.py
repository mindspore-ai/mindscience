"""deep hpms eval"""
import mindspore as ms
from sciai.context import init_project
from sciai.utils import amp2datatype
from sciai.utils.python_utils import print_time
from src.plot import plot_train
from src.process import load_data, prepare


@print_time("eval")
def main(args, problem):
    dtype = amp2datatype(args.amp_level)

    x_sol_star_, exact_sol, t_sol, x_sol, tensors = load_data(args, dtype)
    lb_idn, ub_idn, lb_sol, ub_sol, _, _, _, \
    t_idn_star, x_idn_star, u_idn_star, _, _, _, _, \
    t_sol_star, x_sol_star, u_sol_star = tensors

    model = problem(args.u_layers, args.pde_layers, args.layers, lb_idn, ub_idn, lb_sol, ub_sol)
    ms.load_checkpoint(args.load_ckpt_path, model)
    model.to_float(dtype)

    u_pred = model.eval_sol(t_sol_star, u_sol_star, x_sol_star, t_idn_star, x_idn_star, u_idn_star)
    if args.save_fig:
        plot_train(exact_sol, t_sol, x_sol, x_sol_star_, lb_sol.asnumpy(), u_pred, ub_sol.asnumpy(), args)


if __name__ == "__main__":
    args_problem = prepare()
    init_project(args=args_problem[0])
    main(*args_problem)
