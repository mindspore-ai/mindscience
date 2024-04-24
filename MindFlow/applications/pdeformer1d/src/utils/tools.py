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
r"""Some tool functions."""
import random
from typing import Tuple, Callable, List
import numpy as np
import sympy as sp

import mindspore as ms
from mindspore import nn, ops, Tensor

from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QGroupBox, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPixmap, QPainter, QPainterPath

def set_seed(seed: int) -> None:
    r"""Set random seed"""
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)

def calculate_num_params(model: nn.Cell) -> str:
    r"""Calculate the number of parameters."""
    num_params = 0
    for param in model.trainable_params():
        num_params += np.prod(param.shape)

    if num_params < 1000:
        num_str = str(num_params)
    elif num_params < 1000 * 1000:
        num_str = f"{(num_params / 1000):.2f}" + "K"
    elif num_params < 1000 * 1000 * 1000:
        num_str = f"{(num_params / (1000*1000)):.2f}" + "M"
    else:
        num_str = f"{(num_params / (1000*1000*1000)):.2f}" + "G"

    return num_str

class AllGather(nn.Cell):
    r"""Use nn.Cell to encapsulate ops.AllGather()."""

    def __init__(self) -> None:
        super().__init__()
        self.allgather = ops.AllGather()

    def construct(self, x: Tensor) -> Tensor:
        return self.allgather(x)

def postprocess_batch_data(label: Tensor,
                           pred: Tensor,
                           data_info: dict,
                           idx: int,
                           model_type: str) -> Tuple[np.ndarray, np.ndarray, str]:
    r"""
    Postprocess the batch data to get pde latex and the label and predicted values (with only first component).

    Args:
        label (Tensor): The label tensor. Shape: math:`[bsz, n\_t\_grid * n\_x\_grid, dim\_out]`
            or `[bsz, dim\_out, n\_t\_grid, n\_x\_grid]` or `[bsz, n\_t\_grid, n\_x\_grid, dim\_out]`.
        pred (Tensor): The predicted tensor. Shape: math:`[bsz, n\_t\_grid * n\_x\_grid, dim\_out]`
            or `[bsz, dim\_out, n\_t\_grid, n\_x\_grid]` or `[bsz, n\_t\_grid, n\_x\_grid, dim\_out]`.
        data_info (dict): The data information.
        idx (int): The index of the sample in the batch.
        model_type (str): The model type.

    Returns:
        Tuple[np.ndarray, np.ndarray, str]: The label and predicted values (with only first component) and pde latex.
    """
    if len(label.shape) == 3:
        tx_grid_shape = (data_info["n_t_grid"], data_info["n_x_grid"])
        label_ = label[idx, :, 0].view(tx_grid_shape)
        pred_ = pred[idx, :, 0].view(tx_grid_shape)
    elif len(label.shape) == 4:
        if model_type == "u-net":
            label_ = label[idx, 0, :, :]
            pred_ = pred[idx, 0, :, :]
        elif model_type == "fno":
            label_ = label[idx, :, :, 0]
            pred_ = pred[idx, :, :, 0]
        else:
            raise ValueError(f"The model_type {model_type} is not supported!")
    else:
        raise RuntimeError("Tensor shape error!")
    label_ = label_.asnumpy().astype(np.float32)
    pred_ = pred_.asnumpy().astype(np.float32)
    return label_, pred_, data_info["pde_latex"]

def postprocess_data(label: Tensor,
                     pred: Tensor,
                     data_info: dict,
                     model_type: str) -> Tuple[np.ndarray, np.ndarray, str]:
    r"""
    Postprocess the data (only one sample) to get pde latex and the label and predicted
    values (with only first component).

    Args:
        label (Tensor): The label tensor. Shape: math:`[n\_t\_grid * n\_x\_grid, dim\_out]`
            or `[dim\_out, n\_t\_grid, n\_x\_grid]` or `[n\_t\_grid, n\_x\_grid, dim\_out]`.
        pred (Tensor): The predicted tensor. Shape: math:`[n\_t\_grid * n\_x\_grid, dim\_out]`
            or `[dim\_out, n\_t\_grid, n\_x\_grid]` or `[n\_t\_grid, n\_x\_grid, dim\_out]`.

    Returns:
        Tuple[np.ndarray, np.ndarray, str]: The label and predicted values (with only first component) and pde latex.
    """
    if len(label.shape) == 2:
        tx_grid_shape = (data_info["n_t_grid"], data_info["n_x_grid"])
        label_ = label[:, 0].view(tx_grid_shape)
        pred_ = pred[:, 0].view(tx_grid_shape)
    elif len(label.shape) == 3:
        if model_type == "u-net":
            label_ = label[0, :, :]
            pred_ = pred[0, :, :]
        elif model_type == "fno":
            label_ = label[:, :, 0]
            pred_ = pred[:, :, 0]
        else:
            raise ValueError(f"The model_type {model_type} is not supported!")
    else:
        raise RuntimeError("Tensor shape error!")

    label_ = label_.asnumpy().astype(np.float32)
    pred_ = pred_.asnumpy().astype(np.float32)
    return label_, pred_, data_info["pde_latex"]

def sym_2_np_function(sym_str: str) -> Callable[[np.ndarray], np.ndarray]:
    r"""
    Change a symbolic expression to a numpy function.

    This function examines a symbolic expression represented as a string. If the expression is constant
    with respect to the symbol 'x', it returns a function yielding an array filled with this constant value
    for any input array. Otherwise, it converts the symbolic expression to a NumPy-compatible numerical
    function.

    Args:
        sym_str (str): The symbolic string expression involving variable 'x'.

    Returns:
        Callable[[np.ndarray], np.ndarray]: A function that takes a NumPy array as an argument and returns
        a NumPy array of the evaluated expression values matching the input array's shape.
    """
    x = sp.symbols('x')
    expr = sp.sympify(sym_str)
    if expr.diff(x) == 0:
        const_value = float(expr)
        return lambda arr: np.full_like(arr, const_value)
    return sp.lambdify(x, expr, 'numpy')

def to_latex(expr_str: str) -> str:
    r"""
    Convert a string expression to a LaTeX string.

    This function takes a mathematical expression in string form, converts it into a SymPy expression,
    and then renders that expression into a LaTeX formatted string.

    Args:
        expr_str (str): The mathematical expression as a string.

    Returns:
        str: A string containing the LaTeX representation of the input expression.
    """
    # Convert the string expression to a sympy expression
    sympy_expr = sp.sympify(expr_str)
    # Convert the sympy expression to LaTeX
    return sp.latex(sympy_expr)

def get_shadow():
    r"""
    Create a drop shadow effect for a widget.

    This function initializes a QGraphicsDropShadowEffect, sets its properties for blur radius, X and Y offsets,
    and color, and then returns the configured shadow effect.

    Returns:
        QGraphicsDropShadowEffect: A drop shadow effect with predefined properties.
    """
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(30)
    shadow.setXOffset(0)
    shadow.setYOffset(0)
    shadow.setColor(QColor(0, 0, 0, 60))
    return shadow

def pixmap_with_rounded_corners(pixmap: QPixmap, radius: float = 30.0) -> QPixmap:
    r"""
    Create a new pixmap with rounded corners.

    This function takes a QPixmap object and a radius, creates a new QPixmap with the same size and
    transparent background, draws the original pixmap into it with rounded corners of the specified radius,
    and returns the new pixmap.

    Args:
        pixmap (QPixmap): The original pixmap to modify.
        radius (float): The radius of the rounded corners.

    Returns:
        QPixmap: A new pixmap with rounded corners.
    """
    # Create a new pixmap filled with transparent color
    rounded = QPixmap(pixmap.size())
    rounded.fill(Qt.transparent)

    # Use QPainter to draw the rounded corners
    painter = QPainter(rounded)
    painter.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(0, 0, pixmap.width(), pixmap.height(), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()

    return rounded

def configure_group_box(title: str = "",
                        font_family: str = "Arial",
                        font_size: int = 14,
                        bold: bool = True) -> None:
    r"""Configures the appearance of a QGroupBox with the specified title, font, and style settings.

    Args:
        group_box (QGroupBox): The QGroupBox to configure.
        title (str): The title text for the QGroupBox.
        font_family (str): The font family to use for the title. Default is "Arial".
        font_size (int): The font size for the title. Default is 14.
        bold (bool): Boolean indicating if the font should be bold. Default is True.
    """
    group_box = QGroupBox()

    # Set the QGroupBox title
    group_box.setTitle(title)

    # Set the style sheet to center the title text and apply custom border and padding
    group_box.setStyleSheet(
        "QGroupBox {"
        "    font-weight: bold;"
        "    margin-top: 10px;"
        "    border: 2px solid gray;"
        "    border-radius: 5px;"
        "    padding: 3px 3px 3px 3px;"
        "} "
        "QGroupBox::title {"
        "    subcontrol-origin: margin;"
        "    subcontrol-position: top center;"  # Centers the title
        "    padding: 0 3px 0 3px;"
        "}"
    )

    # Create and set the font for the title
    font_bold = QFont(font_family, font_size)
    font_bold.setBold(bold)
    group_box.setFont(font_bold)

    # Create a drop shadow effect
    shadow_effect = QGraphicsDropShadowEffect()
    shadow_effect.setBlurRadius(4)
    shadow_effect.setColor(QColor(0, 0, 0, 60))  # Semi-transparent black color
    shadow_effect.setOffset(2, 2)

    # Apply the shadow effect to the group box
    group_box.setGraphicsEffect(shadow_effect)

    return group_box

def add_widget(layout: QVBoxLayout, widgets: List[QWidget]):
    r"""Add a list of widgets to a layout."""
    for widget in widgets:
        layout.addWidget(widget)
