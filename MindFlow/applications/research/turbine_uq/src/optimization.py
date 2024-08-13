# Copyright 2024 Huawei Technologies Co., Ltd
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
"""optimization"""
import os
import time
from typing import List, Tuple

import numpy as np
from mindspore import dtype as mstype
from mindflow import load_yaml_config, print_log
from omegaconf import DictConfig, OmegaConf
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from .uncertainty import InputTransformer, UQTransformer
from .postprocess import CFDPost2D, get_grid_interp
from .dataset import DataNormer, create_dataloader
from .utils import Record, repeat_tensor
from .model import reload_model


def get_uq_optimization_task(record: Record, var_name: str, uq_number=128,
                             parameter_list: List[str] = ("Total_total_efficiency",),
                             uq_name: List[str] = ('tangle', 'ttem', 'tpre', 'rotate')):
    """ make the optimization in uncetainty condition"""
    config = OmegaConf.create(load_yaml_config(record.config))
    predictor = FieldPredictor(config=config, record=record)
    adapter_gvrb = UQTransformer(var_name, uq_name=uq_name, uq_number=uq_number)
    problem = TurboPredictor(model=predictor,
                             adapter=adapter_gvrb,
                             is_uq_opt=True,
                             n_var=adapter_gvrb.num_var,
                             parameter_list=parameter_list)
    algorithm = NSGA2(pop_size=32)
    return problem, algorithm


def get_dtm_optimization_task(record: Record, var_name: str,
                              parameter_list: List[str] = ("Total_total_efficiency",)):
    """ make the optimization in determine condition"""
    config = OmegaConf.create(load_yaml_config(record.config))
    predictor = FieldPredictor(config=config, record=record)
    adapter_gvrb = InputTransformer(var_name)
    problem = TurboPredictor(model=predictor,
                             adapter=adapter_gvrb,
                             n_var=adapter_gvrb.num_var,
                             parameter_list=parameter_list)
    algorithm = GA(pop_size=32)
    return problem, algorithm


def run_optimization(record: Record, var_name: Tuple[str] = ('S1', 'R1'),
                     optimize_type='uq', opt_iter=30):
    """run_optimization"""
    print_log("optimizing...")
    if optimize_type == 'uq':
        problem, algorithm = get_uq_optimization_task(record, var_name)
    elif optimize_type == 'dtm':
        problem, algorithm = get_dtm_optimization_task(record, var_name)
    opt_start_time = time.time()
    res = minimize(problem, algorithm, termination=('n_gen', opt_iter),
                   verbose=True, save_history=True)
    print_log(f"optimal sample of {optimize_type} task: {res.X}")
    print_log(f"optimal value of {optimize_type} task: {res.F}")
    print_log(f"total optimization time of {optimize_type} task: {time.time() - opt_start_time:>5.3f}s")
    dict_rst = {'time': time.time() - opt_start_time}
    for name in ('X', 'F'):
        dict_rst.update({name: np.array(get_history_value(res.history, name))})
    save_path = os.path.join(record.record_path, optimize_type)
    os.makedirs(save_path, exist_ok=True)
    np.savez(os.path.join(save_path, '_'.join(var_name) + '.npz'), **dict_rst)


class FieldPredictor:
    """FieldPredictor"""
    def __init__(self, config: DictConfig, record: Record,
                 grid_size_r=64, grid_size_z=128):
        self.netmodel = reload_model(config, ckpt_file_path=record.ckpt_model)
        self.in_norm = DataNormer(data_type='x_norm')
        self.out_norm = DataNormer(data_type='y_norm')
        self.grid_size_r = grid_size_r
        self.grid_size_z = grid_size_z

    def predictor_cfd_value(self, inputs_fields, parameter_list=None, set_opt=False):
        """predictor_cfd_value"""
        pred_2d = self._predicter_2d(inputs_fields)
        post_pred = CFDPost2D(data=pred_2d, grid=get_grid_interp())
        rst_list = []
        for parameter in parameter_list:
            value = post_pred.get_performance(parameter)
            rst = value[..., np.newaxis].copy()
            if set_opt:
                para_max_min_dict = load_yaml_config(os.path.join('./configs', 'optimization.yaml'))['max_min_para']
                rst *= para_max_min_dict[parameter]
            rst_list.append(rst)
        return np.concatenate(rst_list, axis=-1)

    def _predicter_2d(self, input_all, batch_size=32):
        """predicter_2d"""
        input_all = self.in_norm.norm(input_all).astype(np.float32)
        loader = create_dataloader(input_all, np.zeros((input_all.shape[0], 1)), batch_size=batch_size)
        pred = []
        for inputs, _ in loader:
            inputs = inputs.astype(mstype.float32)
            pred.append(self.netmodel(repeat_tensor(inputs)).asnumpy())
        pred = np.concatenate(pred, axis=0)
        return self.out_norm.un_norm(pred.reshape([pred.shape[0], self.grid_size_r, self.grid_size_z, -1]))


class TurboPredictor(Problem):
    """The predictor of turbine performance in optimization process"""
    def __init__(self, model: FieldPredictor = None,
                 adapter: InputTransformer = None, parameter_list=None,
                 n_var=None, is_uq_opt=False):
        self.model = model
        self.adapter = adapter
        self.parameter_list = parameter_list
        n_obj = len(parameter_list)
        if is_uq_opt:
            n_obj = len(adapter.uq_list)
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out):
        out["F"] = self.adapter.transfor_outputs(
                        self.model.predictor_cfd_value(
                            self.adapter.transfor_inputs(x),
                            parameter_list=self.parameter_list,
                            set_opt=True))


def get_history_value(history, name='X'):
    """get_history_value"""
    rst = []
    for population in history:
        population_rst = []
        for data in population.pop:
            population_rst.append(getattr(data, name))
        rst.append(population_rst)
    return rst
