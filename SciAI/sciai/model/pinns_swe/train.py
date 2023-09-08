"""pinns swe train"""
from sciai.context import init_project
from sciai.utils.python_utils import print_time

from src.plot import plot_init_solution, plot_loss, plot_comparison_with_truth, plot_comparison_with_initial
from src.problem import Problem
from src.process import collocation_points, prepare


@print_time("train")
def main(args):
    problem = Problem(args)

    if args.save_fig:
        plot_init_solution(problem, args)

    t_bdry = [problem.t0, problem.t_final]
    x_bdry = [problem.lmbd_left, problem.lmbd_right]
    y_bdry = [problem.tht_lower, problem.tht_upper]
    pdes, inits = collocation_points(args, t_bdry, x_bdry, y_bdry)

    loss = problem.train(pdes, inits)

    if args.save_fig:
        plot_loss(problem, args, loss)
        plot_comparison_with_truth(problem, args, pdes)
        plot_comparison_with_initial(problem, args, pdes)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
