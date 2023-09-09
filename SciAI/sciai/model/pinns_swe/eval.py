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
"""pinns swe eval"""
from sciai.context import init_project
from sciai.utils import print_log
from sciai.utils.python_utils import print_time

from src.plot import plot_init_solution, plot_comparison_with_truth, plot_comparison_with_initial
from src.problem import Problem
from src.process import collocation_points, prepare


@print_time("eval")
def main(args):
    problem = Problem(args)

    plot_init_solution(problem, args)

    t_bdry = [problem.t0, problem.t_final]
    x_bdry = [problem.lmbd_left, problem.lmbd_right]
    y_bdry = [problem.tht_lower, problem.tht_upper]
    pdes, inits = collocation_points(args, t_bdry, x_bdry, y_bdry)
    loss_val = problem.evaluate(pdes, inits)
    print_log('validation loss:', loss_val)

    problem.nr_models = 4
    plot_comparison_with_truth(problem, args, pdes)
    plot_comparison_with_initial(problem, args, pdes)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
