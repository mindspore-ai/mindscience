# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
#pylint: disable=W0613
"""
This dataset module supports various type of datasets, including .... Some of the operations that are
provided to users to preprocess data include shuffle, batch, repeat, map, and zip.
"""
from __future__ import absolute_import


class Problem():
    """
    Base class of user-defined problems.
    All user-defined problems to set constraint on each dataset should be and must be inherited from this class.
    It is utilized to establish the mapping between each sub-dataset and used-defined loss functions.
    The mapping will be constructed by the Constraint API, and the loss will be calculated automatically by the
    constraint type of each sub-dataset. Corresponding member functions must be overloaded by user based on the
    constraint type in order to obtain the target label output. For example, for dataset1 the constraint type is
    set to be "Equation", so the member function "governing_equation" must be overloaded to tell that how to get
    the equation residual.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self):
        super(Problem, self).__init__()
        self.problem_type = type(self).__name__
        self.domain_name = None
        self.bc_name = None
        self.ic_name = None
        self.bc_label_name = None
        self.ic_label_name = None

    def set_name(self, key, value):
        """Set name for the problem"""
        if key == "domain":
            self.domain_name = value
        elif key == "bc":
            self.bc_name = value
        elif key == "ic":
            self.ic_name = value
        elif key == "bc_label":
            self.bc_label_name = value
        elif key == "ic_label":
            self.ic_label_name = value
        else:
            raise ValueError("the value of {} should be domain, bc, ic, bc_label or ic_label, \
                             but got: {}".format(key, value))

    def governing_equation(self, *output, **kwargs):
        """
        governing equation, abstract method.
        this function must be overloaded, if the corresponding constraint type is "Equation".
        if equation is f(inputs) = 0, the residual f will be returned, inputs are data in governing domain.

        Args:
            output (tuple): output of surrogate model, such as electric field and magnetic field,etc.
            kwargs (dict): input of surrogate model, such as time and space coordinates,etc.
        """
        return None

    def boundary_condition(self, *output, **kwargs):
        """
        boundary condition, abstract method.
        this function must be overloaded, if the corresponding constraint type is "BC".
        if boundary condition can be expressed as f(bc_points) = 0, the residual f will be returned,
        bc_points are data on the boundary.

        Args:
            output (tuple): output of surrogate model, such as electric field and magnetic field,etc.
            kwargs (dict): input of surrogate model, such as time and space coordinates,etc.
        """
        return None

    def initial_condition(self, *output, **kwargs):
        """
        initial condition, abstract method.
        this function must be overloaded, if the corresponding constraint type is "IC"
        if initial condition can be expressed f(ic_points) = 0, the residual f will be returned,
        ic_points are data at initial time.

        Args:
            output (tuple): output of surrogate model, such as electric field and magnetic field,etc.
            kwargs (dict): input of surrogate model, such as time and space coordinates,etc.
        """
        return None

    def constraint_function(self, *output, **kwargs):
        """
        general case of functional constraint, abstract method.
        this function must be overloaded, if the corresponding constraint type is "Label" or "Function".
        It's more general case of constraint types which can be expressed as f(inputs) = 0,
        inputs are data corresponding to general function. The residual f will be returned.

        Args:
            output (tuple): output of surrogate model, such as electric field and magnetic field,etc.
            kwargs (dict): input of surrogate model, such as time and space coordinates,etc.
        """
        return None

    def custom_function(self, *output, **kwargs):
        """
        self-defined constraint function, abstract method.
        This function must be overloaded, if the corresponding constraint type is "Custom".
        It's more general case which the constraint can be expresed as f(inputs) = 0.
        The residual f will be returned.

        Args:
            output (tuple): output of surrogate model, such as electric field and magnetic field,etc.
            kwargs (dict): input of surrogate model, such as time and space coordinates,etc.
        """
        return None