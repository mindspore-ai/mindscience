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
r"""User interface for the PDEformer-1 application."""
import sys
from typing import Optional, Tuple
import numpy as np
from mindspore import context
from mindspore import dtype as mstype
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QLabel, QLineEdit, QHBoxLayout, QSlider, QVBoxLayout, QWidget, QApplication,
    QSpacerItem, QSizePolicy, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from src import load_config, get_model, generate_plot
from src.utils.visual import plot_field
from src.utils.tools import (
    sym_2_np_function, to_latex, get_shadow,
    pixmap_with_rounded_corners, configure_group_box)

class InteractiveChart:
    r"""Create an interactive chart using matplotlib."""
    def __init__(self,
                 fig_size: Optional[Tuple[int, int]] = None,
                 coord: Optional[np.ndarray] = None,
                 value: Optional[np.ndarray] = None,
                 title: Optional[str] = None):
        if fig_size is None:
            self.figure = plt.figure()
        else:
            self.figure = plt.figure(figsize=fig_size)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setGraphicsEffect(get_shadow())
        # Plot the initial field
        if coord is not None:
            plot_field(self.figure, self.canvas, coord, value, title)

class MainWindow(QWidget):
    r"""
    The main GUI interface for the PDEformer-1 application, a numerical PDE solver.

    This class provides interactive controls for PDE parameters (including coefficients, initial condition,
    boundary conditions, source term, and viscosity term), visualizing equation components. It features an
    integrated environment that combines widgets such as sliders, input fields, checkboxes, and displays
    using matplotlib for real-time simulation and visualization.

    The MainWindow organizes UI components into logical groups and supports responsive updates to
    simulation visuals based on user interactions. It serves as an educational and research tool
    in the field of numerical PDE solving.
    """
    def __init__(self):
        super().__init__()

        # Define spatial and temporal coordinates
        self.x_coord, self.t_coord = np.linspace(-1, 1, 257), np.linspace(0, 1, 111)

        # Define initial condition
        self.g = np.sin(2 * np.pi * self.x_coord) - 0.3 *np.sin(np.pi * self.x_coord) + 0.2

        self.source, self.source_buffer = None, np.zeros_like(self.x_coord)
        self.kappa, self.kappa_buffer = None, np.zeros_like(self.x_coord)
        self.u_pred = None
        self.periodic = True
        self.robin_list = [0.15, 0.2, 0.0, -0.3]

        self.sliders = {}
        self.slider_values = {'c_00': 0, 'c_01': 0, 'c_02': 0,
                              'c_10': 0, 'c_11': 0.5, 'c_12': 0,
                              'kappa': 0.0, 'source': 0.0}
        self.slider_name_unicode = {'c_00': 'c\u2080\u2080', 'c_01': 'c\u2080\u2081', 'c_02': 'c\u2080\u2082',
                                    'c_10': 'c\u2081\u2080', 'c_11': 'c\u2081\u2081', 'c_12': 'c\u2081\u2082',
                                    'kappa': '\u03BA', 'source': 's'}

        self.bc_boxes = []  # List to keep references to the boundary conditions value boxes
        self.bc_boxes_label = [self.initialize_label("θ\u2081", width=60), # θ₁
                               self.initialize_label("γ\u2081", width=60), # γ₁
                               self.initialize_label("θ\u2082", width=60), # θ₂
                               self.initialize_label("γ\u2082", width=60)] # γ₂

        # Initializes the main user interface for the PDEformer-1 application
        self.resize(2040, 1300)
        self.setWindowTitle("PDEformer-1")
        self.whole_layout = QVBoxLayout()
        self.two_col_layout = QHBoxLayout()
        self.field_layout = QVBoxLayout()   # Field Input layout
        self.result_layout = QVBoxLayout()  # Coef Input and Inference layout

        # Set up font configurations
        self.font = QFont("Arial", 14)
        self.font_bold = QFont("Arial", 14)
        self.font_bold.setBold(True)

        # Image Setup
        self.image_label = QLabel()

        # Setup for the first pixmap image
        pixmap = QPixmap('./images/latex_render_sx_kappa.png')
        new_width = self.width()
        self.scaled_pixmap = pixmap.scaled(new_width, pixmap.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scaled_pixmap = pixmap_with_rounded_corners(self.scaled_pixmap)
        self.image_label.setPixmap(self.scaled_pixmap)
        self.image_label.setGraphicsEffect(get_shadow())

        # Setup for the second pixmap image
        pixmap2 = QPixmap('./images/latex_render_sx_bc_kappa.png')
        self.scaled_pixmap2 = pixmap2.scaled(new_width, pixmap2.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scaled_pixmap2 = pixmap_with_rounded_corners(self.scaled_pixmap2)

        # Adding the image label to the layout
        self.whole_layout.addWidget(self.image_label)

        # Create sublayouts
        input_layout1 = QHBoxLayout()
        input_layout1_v, input_layout2_v, input_layout3_v = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()

        input1_group_box = configure_group_box()
        input2_group_box = configure_group_box()
        input3_group_box = configure_group_box()

        # Initial Condition
        self.initial_condition_label, self.initial_condition_input = \
            self.setup_input_field("Initial Condition", 220, 440, "sin(2*pi*x)-0.3*sin(pi*x)+0.2")
        input_layout1.addWidget(self.initial_condition_label)
        input_layout1.addWidget(self.initial_condition_input)
        input_layout1.addStretch()
        self.initial_condition_input.returnPressed.connect(lambda: self.on_input_enter('initial_condition'))
        self.ic_fig = InteractiveChart(coord=self.x_coord, value=self.g)

        input_layout1_v.addLayout(input_layout1)
        input_layout1_v.addWidget(self.ic_fig.canvas)
        input1_group_box.setLayout(input_layout1_v)
        self.field_layout.addWidget(input1_group_box)

        # Source Term
        self.source_term_label, self.toggle_source_cb, self.source_term_input, _ = \
            self.setup_field_with_toggle_and_slider("Source", 'source', -300, 300, "0", 100, input_layout2_v)
        self.source_fig = InteractiveChart(coord=self.x_coord, value=self.source_buffer, title='s(x)')

        input_layout2_v.addWidget(self.source_fig.canvas)
        input2_group_box.setLayout(input_layout2_v)
        self.field_layout.addWidget(input2_group_box)

        # Kappa Term
        self.kappa_term_label, self.toggle_kappa_cb, self.kappa_term_input, _ = \
            self.setup_field_with_toggle_and_slider("Viscosity", 'kappa', 0, 100, "0", 10, input_layout3_v)
        self.kappa_fig = InteractiveChart(coord=self.x_coord, value=self.kappa_buffer, title='\u03BA(x)')

        input_layout3_v.addWidget(self.kappa_fig.canvas)
        input3_group_box.setLayout(input_layout3_v)
        self.field_layout.addWidget(input3_group_box)

        # Checkbox for toggling the visibility of boundary conditions boxes
        right_align_box = QHBoxLayout()  # Create a new horizontal layout
        self.toggle_bc_cb = self.setup_toggle_checkbox("Non-periodic", "boundary_condition", width=230)
        right_align_box.addStretch()
        right_align_box.addWidget(self.toggle_bc_cb)
        self.result_layout.addLayout(right_align_box)

        # Set up the sliders and value boxes for the coefficients and boundary conditions
        self.sliders_bc = []
        self.setup_coef()

        self.result_layout.addSpacerItem(QSpacerItem(20, 200, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Inference figure
        self.fig = InteractiveChart(fig_size=(13, 9))
        self.fig.canvas.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.result_layout.addStretch(1)
        self.result_layout.addWidget(self.fig.canvas, alignment=Qt.AlignBottom)

        # Wrap layout with a group box
        self.result_group_box = configure_group_box()
        self.result_group_box.setLayout(self.result_layout)

        self.two_col_layout.addLayout(self.field_layout)
        self.two_col_layout.addWidget(self.result_group_box)
        self.whole_layout.addLayout(self.two_col_layout)
        self.setLayout(self.whole_layout)

        # Initial plot
        self.update_plot()

    def setup_coef(self) -> None:
        r"""Set up the sliders and value boxes for the coefficients and boundary conditions."""
        i = 0
        for name, initial_value in self.slider_values.items():

            if name in ('kappa', 'source'):
                continue

            slider_layout, slider, value_box, min_label, max_label = self.setup_slider(-300, 300, 100)
            if name not in self.slider_name_unicode:
                raise ValueError(f"Slider name '{name}' not found in the unicode dictionary.")
            label_html = f"<b>{self.slider_name_unicode[name]}</b>"
            label = self.initialize_label(label_html, width=60)
            slider.setValue(int(initial_value * 100))
            value_box.setText(f"{initial_value}")
            slider_layout.insertWidget(0, label) # Insert the custom label at the beginning

            if i < 4:
                _, slider_bc, bc_box, min_label_bc, max_label_bc = \
                    self.setup_slider(-300 if i in (1, 3) else 0, 300 if i in (1, 3) else 100, 100)
                self.bc_boxes.append(bc_box)  # Keep track of the QLineEdit for boundary conditions
                bc_box.setText(str(self.robin_list[i]))
                bc_box.returnPressed.connect(lambda i=i, b=bc_box: self.unified_line_edit_enter(line_edit=b, i=i))
                # Adjust the slider to the current value
                slider_bc.setValue(int(self.robin_list[i] * 100))
                slider_bc.valueChanged.connect(lambda value, i=i: self.unified_slider_changed(value, i=i))
                # Disable widgets
                for widget in [bc_box, min_label_bc, slider_bc, max_label_bc, self.bc_boxes_label[i]]:
                    widget.setEnabled(False)

                slider_layout.addStretch(1)
                slider_layout.addWidget(self.bc_boxes_label[i])
                slider_layout.addWidget(min_label_bc, alignment=Qt.AlignLeft)
                slider_layout.addWidget(slider_bc)
                slider_layout.addWidget(max_label_bc, alignment=Qt.AlignRight)
                slider_layout.addWidget(bc_box)
            else:
                slider_layout.addStretch(1)

            self.result_layout.addLayout(slider_layout)

            # Connecting slider and line edit interactions
            slider.valueChanged.connect(lambda value, name=name: self.unified_slider_changed(value, name))
            value_box.returnPressed.connect(
                lambda name=name, b=value_box: self.unified_line_edit_enter(line_edit=b, name=name))

            # Storing references to sliders and line edits
            if i < 4:
                i += 1
                self.sliders_bc.append((slider_bc, bc_box, min_label_bc, max_label_bc))
            self.sliders[name] = (slider, value_box, min_label, max_label, label)

    def setup_input_field(self,
                          label: str,
                          label_width: int,
                          input_width: int, default_text: str) -> Tuple[QLabel, QLineEdit]:
        r"""Creates a labeled input field and configures its properties.

        Args:
            label (str): Text to be displayed by the QLabel.
            label_width (int): The designated fixed width for the QLabel.
            input_width (int): The designated fixed width for the QLineEdit.
            default_text (str): The default text value to pre-populate the QLineEdit with.

        Returns:
            Tuple[QLabel, QLineEdit]: A tuple containing the configured QLabel and QLineEdit instances.
        """
        # Create and configure the label
        field_label = QLabel(label)
        field_label.setFont(self.font_bold)
        field_label.setFixedWidth(label_width)

        # Create and configure the input field
        field_input = QLineEdit()
        field_input.setFont(self.font)  # Sets the font and font size
        field_input.setFixedWidth(input_width)
        field_input.setText(default_text)

        return field_label, field_input

    def setup_toggle_checkbox(self, label: str, field: str, width: int = 120) -> QCheckBox:
        r"""Creates and setups a QCheckBox with specified attributes and connects
        its stateChanged signal to toggle visibility or other actions.

        Args:
            label (str): The text label for the checkbox.
            field (str): The name of the field this checkbox is related to, used in connecting signals.
            width (int): Optional width of the checkbox, defaults to 120.

        Returns:
            QCheckBox: The created and configured checkbox.
        """
        toggle_cb = QCheckBox(label, self)
        toggle_cb.setFont(self.font_bold)
        toggle_cb.setFixedWidth(width)
        toggle_cb.stateChanged.connect(lambda state: self.toggle_visibility(state, field))

        return toggle_cb

    def setup_field_with_toggle_and_slider(self,
                                           input_label: str,
                                           toggle_field: str,
                                           min_slider_value: int,
                                           max_slider_value: int,
                                           default_text: str,
                                           slider_step_interval: int,
                                           layout_v: QVBoxLayout) -> Tuple[QLabel, QCheckBox, QLineEdit, QSlider]:
        r"""
        Constructs a user interface section with field inputs or slider constants.

        Args:
            input_label (str): Input field label.
            toggle_field (str): Unique identifier used for toggling and referencing the components.
            min_slider_value (int): Minimum value for the slider.
            max_slider_value (int): Maximum value for the slider.
            default_text (str): Default text to display in the input field.
            slider_step_interval (int): Interval between steps in the slider.
            layout_v (QVBoxLayout): The vertical layout to which this section will be added.

        Returns:
            Tuple[QLabel, QCheckBox, QLineEdit, QSlider]
        """

        # Setup label and input field
        field_label, field_input = self.setup_input_field(input_label, 150, 360, default_text)

        # Setup toggle checkbox
        toggle_cb = self.setup_toggle_checkbox("Field", toggle_field)

        # Setup slider
        slider_layout, slider, value_box, min_label, max_label = \
            self.setup_slider(min_slider_value, max_slider_value, slider_step_interval)

        # Connecting signals
        field_input.returnPressed.connect(lambda: self.on_input_enter(toggle_field))
        slider.valueChanged.connect(lambda value, name=toggle_field: self.unified_slider_changed(value, name=name))
        value_box.returnPressed.connect(
            lambda name=toggle_field, b=value_box: self.unified_line_edit_enter(line_edit=b, name=name))
        self.sliders[toggle_field] = (slider, value_box, min_label, max_label)  # Keep a reference if needed

        # Assemble the layout
        layout_h = QHBoxLayout()
        layout_h.addWidget(field_label)
        layout_h.addWidget(toggle_cb)
        layout_h.addStretch()
        layout_h.addWidget(field_input)

        layout_v.addLayout(layout_h)
        layout_v.addLayout(slider_layout)
        field_input.setVisible(False)  # Initially hidden, toggle with checkbox

        output = (field_label, toggle_cb, field_input, slider)
        return output

    def setup_slider(self,
                     min_value: int,
                     max_value: int,
                     tick_interval: int,
                     width: int = 350) -> Tuple[QHBoxLayout, QSlider, QLineEdit, QLabel, QLabel]:
        r"""
        Sets up a horizontal slider layout with labels and a value box for GUI component configuration.

        Args:
            min_value (int): The minimum value of the slider.
            max_value (int): The maximum value of the slider.
            tick_interval (int): The interval between ticks on the slider.
            width (int): The width of the slider. Defaults to 350.

        Returns:
            Tuple[QHBoxLayout, QSlider, QLineEdit, QLabel, QLabel]: A tuple containing the QHBoxLayout
            (which itself contains all widgets), the QSlider widget, the QLineEdit for showing current value,
            and labels for min and max values.
        """
        slider_layout = QHBoxLayout()

        # Setup labels for the min and max slider values
        min_label, max_label = QLabel(f"<b>{min_value/100}<b>"), QLabel(f"<b>{max_value/100}<b>")
        min_label.setFixedWidth(50)
        max_label.setFixedWidth(50)
        min_label.setAlignment(Qt.AlignCenter)
        max_label.setAlignment(Qt.AlignCenter)

        # Setup the slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(tick_interval)
        slider.setFixedWidth(width)

        # Setup the value box
        value_box = QLineEdit("0.0")
        value_box.setFixedWidth(60)

        # Add widgets to the slider layout
        slider_layout.addWidget(min_label)
        slider_layout.addWidget(slider)
        slider_layout.addWidget(max_label)
        slider_layout.addWidget(value_box)

        output = (slider_layout, slider, value_box, min_label, max_label)
        return output

    def initialize_label(self,
                         text: str,
                         font_name: str = "STIX",
                         font_size: int = 14,
                         width: int = 60) -> QLabel:
        r"""
        Initializes and returns a QLabel with specific formatting and text.

        Args:
            text (str): The text to display in the label.
            font_name (str, optional): The name of the font. Defaults to "STIX".
            font_size (int, optional): The size of the font. Defaults to 14.
            width (int, optional): The fixed width of the label in pixels. Defaults to 60.

        Returns:
            QLabel: Configured QLabel object ready to be used in the GUI.
        """
        label = QLabel(f"<b>{text}</b>")
        label.setFont(QFont(font_name, font_size))
        label.setAlignment(Qt.AlignCenter)
        label.setFixedWidth(width)
        return label

    def toggle_visibility(self, state: int, control_type: str) -> None:
        r"""Toggles the visibility and activity of UI elements based on the selected control type.

        Args:
            state (int): Indicates the desired visibility state, typically derived from a checkbox.
            control_type (str): Specifies the type of control ('source', 'kappa', or 'boundary_condition')
                                whose visibility is being toggled.
        """
        is_visible = bool(state)
        if control_type == 'source':
            if 'source' not in self.sliders:
                raise ValueError("Source sliders not found in the sliders dictionary.")
            for i in range(len(self.sliders['source'])):
                self.sliders['source'][i].setEnabled(not is_visible)
            self.source_term_input.setVisible(is_visible)
            # Adjust the plotting for source visibility
            if not is_visible:
                self.source = None
                if 'source' not in self.slider_values:
                    raise ValueError("Source term not found in the slider values dictionary.")
                plot_field(self.source_fig.figure, self.source_fig.canvas, self.x_coord,
                           self.slider_values['source'], title='s(x)')
            else:
                self.source = self.source_buffer
                plot_field(self.source_fig.figure, self.source_fig.canvas,
                           self.x_coord, self.source_buffer, title='s(x)')
        elif control_type == 'kappa':
            if 'kappa' not in self.sliders:
                raise ValueError("Kappa sliders not found in the sliders dictionary.")
            for i in range(len(self.sliders['kappa'])):
                self.sliders['kappa'][i].setEnabled(not is_visible)
            self.kappa_term_input.setVisible(is_visible)
            # Adjust the plotting for kappa visibility
            if not is_visible:
                self.kappa = None
                plot_field(self.kappa_fig.figure, self.kappa_fig.canvas, self.x_coord,
                           self.slider_values['kappa'], title='\u03BA(x)')
            else:
                self.kappa = self.kappa_buffer
                plot_field(self.kappa_fig.figure, self.kappa_fig.canvas,
                           self.x_coord, self.kappa_buffer, title='\u03BA(x)')
        elif control_type == 'boundary_condition':
            for i in range(4):
                self.bc_boxes_label[i].setEnabled(is_visible)
                for j in range(4):
                    self.sliders_bc[i][j].setEnabled(is_visible)

            self.image_label.setPixmap(self.scaled_pixmap2 if is_visible else self.scaled_pixmap)
            self.periodic = not is_visible

        self.update_plot()

    def on_input_enter(self, input_type: str) -> None:
        r"""Handles input updates for various simulation conditions.

        Args:
            input_type (str): The type of input being entered.
                              It can be 'initial_condition', 'source_term', or 'kappa_term'.
        """
        input_text = ""
        if input_type == 'initial_condition':
            input_text = self.initial_condition_input.text()
            self.g = sym_2_np_function(input_text)(self.x_coord)
            plot_field(self.ic_fig.figure, self.ic_fig.canvas, self.x_coord, self.g)
        elif input_type == 'source':
            input_text = self.source_term_input.text()
            self.source_buffer = sym_2_np_function(input_text)(self.x_coord)
            self.source = self.source_buffer
            plot_field(self.source_fig.figure, self.source_fig.canvas, self.x_coord, self.source_buffer, title='s(x)')
        elif input_type == 'kappa':
            input_text = self.kappa_term_input.text()
            self.kappa_buffer = sym_2_np_function(input_text)(self.x_coord)
            self.kappa = self.kappa_buffer
            plot_field(self.kappa_fig.figure, self.kappa_fig.canvas, self.x_coord, self.kappa_buffer, title='\u03BA(x)')

        self.update_plot()
        print(f"{input_type.replace('_', ' ').capitalize()}: {input_text}")

    def unified_slider_changed(self, value: int, name: Optional[str] = None, i: Optional[int] = None) -> None:
        r"""Updates simulation parameters or boundary conditions based on slider inputs.

        Args:
            value (int): The current value of the slider.
            name (Optional[str], optional): Specifies the name of the simulation parameter. Defaults to None.
            i (Optional[int], optional): Specifies the index for the boundary condition slider. Defaults to None.
        """
        value = value / 100.0
        if name:  # This is for sliders related to 'source', 'kappa', etc.
            self.sliders[name][1].setText(f"{value:.3f}")
            self.slider_values[name] = value
            if name == 'kappa':
                plot_field(self.kappa_fig.figure, self.kappa_fig.canvas, self.x_coord,
                           self.slider_values[name], title='\u03BA(x)')
            elif name == 'source':
                plot_field(self.source_fig.figure, self.source_fig.canvas, self.x_coord,
                           self.slider_values[name], title='s(x)')
        elif i is not None:  # This is specifically for boundary condition sliders
            self.sliders_bc[i][1].setText(f"{value:.3f}")
            self.robin_list[i] = float(value)
        self.update_plot()

    def unified_line_edit_enter(self,
                                line_edit: QLineEdit,
                                name: Optional[str] = None,
                                i: Optional[int] = None) -> None:
        r"""Updates simulation parameters or boundary conditions based on line edit inputs.

        Args:
            line_edit (QLineEdit): The QLineEdit object that received input.
            name (Optional[str], optional): Specifies the name of the simulation parameter. Defaults to None.
            i (Optional[int], optional): Specifies the index for the boundary condition. Defaults to None.
        """

        value = line_edit.text()
        try:
            float_value = float(value)
            slider_value = int(float_value * 100)
            if name:  # This is for general value inputs
                slider = self.sliders[name][0]
                slider.setValue(slider_value)
                self.slider_values[name] = float_value
            elif i is not None:  # This is for boundary condition inputs
                slider = self.sliders_bc[i][0]
                slider.setValue(slider_value)
                self.robin_list[i] = float_value
            self.update_plot()
        except ValueError:
            # Handle case where value is not a float
            pass

    def update_plot(self) -> None:
        r"""Updates the interactive plot based on the current values of the sliders, line edits, or checkbox status.

        This method takes the current slider values from `self.slider_values` and
        passes them as arguments to `self.interactive_plot` to refresh the plot.
        It is called in response to changes made using sliders, entering text into line edits, or toggling a checkbox,
        ensuring the plot accurately reflects all current input settings.
        """
        self.interactive_plot(**self.slider_values)

    def interactive_plot(self,
                         c_00: float,
                         c_01: float,
                         c_02: float,
                         c_10: float,
                         c_11: float,
                         c_12: float,
                         kappa: float,
                         source: float) -> None:
        r"""Generates an interactive plot based on provided coefficients and other initial/boundary conditions.

        This function takes the coefficients for various terms, values for kappa (diffusion coefficient) and
        source (source term), and uses them along with internal states such as initial conditions,
        boundary conditions, and coordinate grids to generate an interactive plot of the model's behavior.

        Args:
            c_00, c_01, c_02, c_10, c_11, c_12 (float): Coefficients for the general PDE terms.
            kappa (float): Diffusion coefficient used in the simulation (ignored if kappa_field is provided).
            source (float): Source term coefficient used in the simulation (ignored if source_field is provided).
        """
        c_list = [c_00, c_01, c_02, c_10, c_11, c_12]
        generate_plot(model=model,
                      config=config,
                      u_ic=self.g,
                      x_coord=self.x_coord,
                      t_coord=self.t_coord,
                      c_list=c_list,
                      kappa=kappa if self.kappa is None else self.kappa,
                      source=source if self.source is None else self.source,
                      periodic=self.periodic,
                      robin_theta_l=self.robin_list[0],
                      bc_value_l=self.robin_list[1],
                      robin_theta_r=self.robin_list[2],
                      bc_value_r=self.robin_list[3],
                      figure=self.fig.figure,
                      canvas=self.fig.canvas,
                      ic_latex=to_latex(self.initial_condition_input.text()),)

if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    config, _ = load_config("configs/inference_pdeformer-L.yaml")

    model = get_model(config, compute_type=mstype.float32)

    app = QApplication(sys.argv)
    window = MainWindow()

    window.show()
    app.exec_()
