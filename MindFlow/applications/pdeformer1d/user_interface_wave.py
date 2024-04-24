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
r"""User interface for the PDEformer-1 application (Wave equation)."""
import sys
from typing import Optional, Tuple
import numpy as np
from mindspore import context
from mindspore import dtype as mstype
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QLabel, QLineEdit, QHBoxLayout, QSlider, QVBoxLayout, QWidget, QApplication,
    QSpacerItem, QSizePolicy, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from src import load_config, get_model, generate_plot_wave
from src.utils.visual import plot_field
from src.utils.tools import (
    sym_2_np_function, to_latex, get_shadow,
    pixmap_with_rounded_corners, configure_group_box, add_widget)

class InteractiveChart:
    r"""Create an interactive chart using matplotlib."""
    def __init__(self,
                 fig_size: Optional[Tuple[int, int]] = None,
                 coord: Optional[np.ndarray] = None,
                 value: Optional[np.ndarray] = None):
        if fig_size is None:
            self.figure = plt.figure()
        else:
            self.figure = plt.figure(figsize=fig_size)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setGraphicsEffect(get_shadow())
        # Plot the initial field
        if coord is not None:
            plot_field(self.figure, self.canvas, coord, value)

class MainWindow(QWidget):
    r"""
    The main GUI interface for the PDEformer-1 application (including Wave Equation), a numerical PDE solver.

    This class provides interactive controls for PDE parameters (including coefficients, initial condition,
    boundary conditions, source term), visualizing equation components. It features an integrated environment
    that combines widgets such as sliders, input fields, checkboxes, and displays using matplotlib for real-time
    simulation and visualization.

    The MainWindow organizes UI components into logical groups and supports responsive updates to
    simulation visuals based on user interactions. It serves as an educational and research tool
    in the field of numerical PDE solving.
    """
    def __init__(self):
        super().__init__()

        # Define Spatial and Temporal Coordinates
        self.x_coord, self.t_coord = np.linspace(-1, 1, 257), np.linspace(0, 1, 111)

        # Define Initial Condition
        self.g, self.h = np.sin(2 * np.pi * self.x_coord), np.zeros_like(self.x_coord)

        self.source, self.source_buffer = None, np.zeros_like(self.x_coord)
        self.wave_speed, self.wave_speed_buffer, self.wave_type = None, 1.4 * np.ones_like(self.x_coord), 0
        self.u_pred = None
        self.periodic = True
        self.robin_list = [0.5, -0.05, 0.75, -2]
        self.use_mur_l, self.use_mur_r = False, False
        self.sol_title = "Solution Field"

        self.sliders = {}
        self.slider_values = {'b_val': 0, 'c_1': 0, 'c_2': 0, 'c_3': 0, 'wave_speed': 1.4, 'source': 0, 'mu_value': 0}
        self.slider_name_unicode = {'b_val': 'b', 'c_1': 'c\u2081', 'c_2': 'c\u2082', 'c_3': 'c\u2083',
                                    'wave_speed': '\u03BA', 'source': 's', 'mu_value': '\u03BC',}

        self.bc_boxes = []  # List to keep references to the boundary conditions value boxes
        self.bc_boxes_label = [self.initialize_label("θ\u2081", width=60),    # θ₁
                               self.initialize_label("γ\u2081", width=60),    # γ₁
                               self.initialize_label("θ\u2082", width=60),    # θ₂
                               self.initialize_label("γ\u2082", width=60)]    # γ₂

        # Initializes the User Interface for the PDEformer-1 Application
        self.resize(2140, 1340)
        self.setWindowTitle("PDEformer-1")
        self.whole_layout = QVBoxLayout()
        self.three_col_layout = QHBoxLayout()
        self.ic_layout, self.field_layout = QVBoxLayout(), QVBoxLayout()    # Field Input layout
        self.result_layout, self.form_layout = QVBoxLayout(), QVBoxLayout() # Coef Input and Inference layout

        # Set up Font Configurations
        self.font = QFont("Arial", 14)
        self.font_bold = QFont("Arial", 14)
        self.font_bold.setBold(True)

        # Image setup
        self.image_label, self.pixmap_list, self.pixmap_bc_list = QLabel(), [], []
        self.display_equation_form()
        self.whole_layout.addWidget(self.image_label)

        # Initial Condition (displacement)
        ic_displace_group_box = configure_group_box()
        ic_displace_layout = QVBoxLayout()
        self.ic_displace_label, self.ic_displace_input = \
            self.setup_input_field("Initial Displacement", 300, 440, "sin(2*pi*x)")
        self.ic_displace_input.returnPressed.connect(lambda: self.on_input_enter('initial_condition'))
        self.ic_displace_fig = InteractiveChart(coord=self.x_coord, value=self.g)
        add_widget(ic_displace_layout, [self.ic_displace_label, self.ic_displace_input, self.ic_displace_fig.canvas])
        ic_displace_group_box.setLayout(ic_displace_layout)

        # Initial Condition (speed)
        ic_speed_group_box = configure_group_box()
        ic_speed_layout = QVBoxLayout()
        self.ic_speed_label, self.ic_speed_input = \
            self.setup_input_field("Initial Velocity", 300, 440, "0")
        self.ic_speed_input.returnPressed.connect(lambda: self.on_input_enter('initial_speed'))
        self.ic_speed_fig = InteractiveChart(coord=self.x_coord, value=self.h)
        add_widget(ic_speed_layout, [self.ic_speed_label, self.ic_speed_input, self.ic_speed_fig.canvas])
        ic_speed_group_box.setLayout(ic_speed_layout)

        # Full Initial Condition Layout
        add_widget(self.ic_layout, [ic_displace_group_box, ic_speed_group_box])

        # Source Term
        source_group_box = configure_group_box()
        source_layout = QVBoxLayout()

        self.source_term_label, self.toggle_source_cb, self.source_term_input, _ = \
            self.setup_field_with_toggle_and_slider("Source", 'source', -300, 300, 0, 100, source_layout)
        self.source_fig = InteractiveChart(coord=self.x_coord, value=self.source_buffer)
        source_layout.addWidget(self.source_fig.canvas)
        source_group_box.setLayout(source_layout)
        self.field_layout.addWidget(source_group_box)

        # Wave Speed Term
        wave_speed_group_box = configure_group_box()
        wave_speed_layout = QVBoxLayout()

        self.wave_speed_term_label, self.toggle_wave_speed_cb, self.wave_speed_term_input, _ = \
            self.setup_field_with_toggle_and_slider("Wave Speed", 'wave_speed', 0, 200, 1.4, 100, wave_speed_layout)
        self.wave_speed_fig = InteractiveChart(coord=self.x_coord, value=self.wave_speed_buffer)
        wave_speed_layout.addWidget(self.wave_speed_fig.canvas)
        wave_speed_group_box.setLayout(wave_speed_layout)
        self.field_layout.addWidget(wave_speed_group_box)

        # Initialize the ComboBox
        self.combo_box = self.initialize_combo_box()

        # Checkbox for toggling the visibility of value boxes
        right_align_box = QHBoxLayout()  # Create a new horizontal layout
        self.wave_form_label = QLabel("Wave Eq. Form")
        self.wave_form_label.setFont(self.font_bold)
        right_align_box.addWidget(self.wave_form_label)
        right_align_box.addWidget(self.combo_box)
        self.toggle_bc_cb = self.setup_toggle_checkbox("Non-periodic", "boundary_condition", width=230)

        right_align_box.addStretch()
        right_align_box.addWidget(self.toggle_bc_cb)
        self.form_layout.addLayout(right_align_box)

        # Set up the sliders and value boxes for the coefficients and boundary conditions
        self.sliders_bc = []
        self.toggle_use_mur_l_cb = self.setup_toggle_checkbox("Mur (Left)", "use_mur_l", width=200)
        self.toggle_use_mur_r_cb = self.setup_toggle_checkbox("Mur (Right)", "use_mur_r", width=200)
        self.setup_coef()

        self.result_layout.addLayout(self.form_layout)

        self.result_layout.addSpacerItem(QSpacerItem(20, 200, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Inference figure
        self.fig = InteractiveChart(fig_size=(13, 9))
        self.fig.canvas.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.result_layout.addStretch(1)
        self.result_layout.addWidget(self.fig.canvas, alignment=Qt.AlignBottom)

        self.three_col_layout.addLayout(self.ic_layout)
        self.three_col_layout.addLayout(self.field_layout)

        # Wrap layout with a group box
        result_group_box = configure_group_box()
        result_group_box.setLayout(self.result_layout)
        self.three_col_layout.addWidget(result_group_box)
        self.whole_layout.addLayout(self.three_col_layout)
        self.setLayout(self.whole_layout)

        # Initial plot
        self.update_plot()

    def display_equation_form(self) -> None:
        r"""Displays different equation forms based on user selection."""
        def pixmap_element(wave_type=0, enable_bc=False, left=0, right=0):
            new_width = self.width()
            if enable_bc:
                pixmap = QPixmap(f'images/latex_render_sx_wave_{wave_type}_bc_mur_l{left}_mur_r{right}.png')
            else:
                pixmap = QPixmap(f'images/latex_render_sx_wave_{wave_type}.png')
            scaled_pixmap = pixmap.scaled(new_width, pixmap.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled_pixmap = pixmap_with_rounded_corners(scaled_pixmap)
            return scaled_pixmap

        # Generate standard pixmap elements
        self.pixmap_list.append(pixmap_element(0))
        self.pixmap_list.append(pixmap_element(1))
        self.pixmap_list.append(pixmap_element(2))

        # Generate pixmap elements with boundary conditions
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    self.pixmap_bc_list.append(pixmap_element(i, True, j, k))

        # Setup image label properties
        self.image_label.setPixmap(self.pixmap_list[0])
        self.image_label.setGraphicsEffect(get_shadow())

    def setup_coef(self) -> None:
        r"""Set up the sliders and value boxes for the coefficients and boundary conditions."""
        i = 0
        for name, initial_value in self.slider_values.items():

            if name in ['wave_speed', 'source']:
                continue

            if name == 'mu_value':
                slider_layout, slider, value_box, min_label, max_label = self.setup_slider(0, 400, 100)
            else:
                slider_layout, slider, value_box, min_label, max_label = self.setup_slider(-300, 300, 100)
            if name not in self.slider_name_unicode:
                raise ValueError(f"Slider name '{name}' not found in slider_name_unicode dictionary.")
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
                slider_layout.addWidget(self.toggle_use_mur_l_cb)
                slider_layout.addWidget(self.toggle_use_mur_r_cb)

                self.toggle_use_mur_l_cb.setEnabled(False)  # Disables the QCheckBox
                self.toggle_use_mur_r_cb.setEnabled(False)  # Disables the QCheckBox

            self.form_layout.addLayout(slider_layout)

            # Connecting slider and line edit interactions
            slider.valueChanged.connect(lambda value, name=name: self.unified_slider_changed(value, name))
            value_box.returnPressed.connect(
                lambda name=name, b=value_box: self.unified_line_edit_enter(line_edit=b, name=name))

            # Storing references to sliders and line edits
            if i < 4:
                i += 1
                self.sliders_bc.append((slider_bc, bc_box, min_label_bc, max_label_bc))
            self.sliders[name] = (slider, value_box, min_label, max_label, label)

    def setup_field_with_toggle_and_slider(self,
                                           input_label: str,
                                           toggle_field: str,
                                           min_slider_value: int,
                                           max_slider_value: int,
                                           default_value: float,
                                           slider_step_interval: int,
                                           layout_v: QVBoxLayout) -> Tuple[QLabel, QCheckBox, QLineEdit, QSlider]:
        r"""
        Constructs a user interface section with field inputs or slider constants.

        Args:
            input_label (str): Input field label.
            toggle_field (str): Unique identifier used for toggling and referencing the components.
            min_slider_value (int): Minimum value for the slider.
            max_slider_value (int): Maximum value for the slider.
            default_value (float): Default value for slider and value box.
            slider_step_interval (int): Interval between steps in the slider.
            layout_v (QVBoxLayout): The vertical layout to which this section will be added.

        Returns:
            Tuple[QLabel, QCheckBox, QLineEdit, QSlider]
        """

        # Setup label and input field
        field_label, field_input = self.setup_input_field(input_label, 190, 120, default_text=str(default_value))

        # Setup toggle checkbox
        toggle_cb = self.setup_toggle_checkbox("Field", toggle_field)

        # Setup slider
        slider_layout, slider, value_box, min_label, max_label = \
            self.setup_slider(min_slider_value, max_slider_value, slider_step_interval)
        slider.setValue(int(default_value * 100))
        value_box.setText(f"{default_value}")

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

    def setup_slider(self,
                     min_value: int,
                     max_value: int,
                     tick_interval: int,
                     width: int = 250) -> Tuple[QHBoxLayout, QSlider, QLineEdit, QLabel, QLabel]:
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

    def initialize_combo_box(self) -> QComboBox:
        r"""Initializes and returns a QComboBox with different wave equation form options."""
        combo_box = QComboBox()
        combo_box.setFont(self.font)
        combo_box.addItem("-c(x)^2 u_{xx}")
        combo_box.addItem("-c(x)(c(x) u_x)_x")
        combo_box.addItem("-(c(x)^2 u_x)_x")
        combo_box.currentIndexChanged.connect(self.combo_selection_changed)
        return combo_box

    def toggle_visibility(self, state: int, control_type: str) -> None:
        r"""Toggles the visibility and activity of UI elements based on the selected control type.

        Args:
            state (int): Indicates the desired visibility state, typically derived from a checkbox.
            control_type (str): Specifies the type of control to modify ('source', 'wave_speed',
            'use_mur_l', 'use_mur_r', 'boundary_condition').
        """
        is_visible = bool(state)

        if control_type in ['source', 'wave_speed']:
            if control_type not in self.sliders:
                raise ValueError(f"Control type '{control_type}' not found in sliders dictionary.")
            for i in range(len(self.sliders[control_type])):
                self.sliders[control_type][i].setEnabled(not is_visible)
            getattr(self, f"{control_type}_term_input").setVisible(is_visible)
            if not is_visible:
                setattr(self, control_type, None)
                if control_type not in self.slider_values:
                    raise ValueError(f"Control type '{control_type}' not found in slider_values dictionary.")
                value = self.slider_values[control_type]
            else:
                setattr(self, control_type, getattr(self, f"{control_type}_buffer"))
                value = getattr(self, f"{control_type}_buffer")
            plot_field(getattr(self, f"{control_type}_fig").figure,
                       getattr(self, f"{control_type}_fig").canvas,
                       self.x_coord,
                       value)
        elif control_type in ['use_mur_l', 'use_mur_r']:
            label_index = 0 if control_type == 'use_mur_l' else 2
            setattr(self, control_type, is_visible)
            self.bc_boxes_label[label_index].setEnabled(not is_visible)
            for j in range(4):
                self.sliders_bc[label_index][j].setEnabled(not is_visible)
            self.show_eq_form()
        elif control_type == 'boundary_condition':
            self.toggle_use_mur_l_cb.setEnabled(is_visible)
            self.toggle_use_mur_r_cb.setEnabled(is_visible)
            for i in range(4):
                self.bc_boxes_label[i].setEnabled(is_visible)
                for j in range(4):
                    self.sliders_bc[i][j].setEnabled(is_visible)
            self.periodic = not is_visible
            self.show_eq_form()

        self.update_plot()

    def show_eq_form(self):
        r"""Updates the equation formimage based on the selected wave equation form and boundary conditions."""
        if not self.toggle_bc_cb.isChecked():
            self.image_label.setPixmap(self.pixmap_list[self.wave_type])
        else:
            self.image_label.setPixmap(
                self.pixmap_bc_list[self.wave_type*4 +
                                    int(self.use_mur_l)*2 +
                                    int(self.use_mur_r)])

    def on_input_enter(self, input_type: str) -> None:
        r"""Handles input updates for various simulation conditions.

        Args:
            input_type (str): The type of input being entered. It can be 'initial_condition',
                              'initial_speed', 'source' or 'wave_speed'.
        """
        input_text = ""
        if input_type == 'initial_condition':
            input_text = self.ic_displace_input.text()
            self.g = sym_2_np_function(input_text)(self.x_coord)
            plot_field(self.ic_displace_fig.figure, self.ic_displace_fig.canvas, self.x_coord, self.g)
        elif input_type == 'initial_speed':
            input_text = self.ic_speed_input.text()
            self.h = sym_2_np_function(input_text)(self.x_coord)
            plot_field(self.ic_speed_fig.figure, self.ic_speed_fig.canvas, self.x_coord, self.h)
        elif input_type == 'source':
            input_text = self.source_term_input.text()
            self.source_buffer = sym_2_np_function(input_text)(self.x_coord)
            self.source = self.source_buffer
            plot_field(self.source_fig.figure, self.source_fig.canvas, self.x_coord, self.source_buffer, title='s(x)')
        elif input_type == 'wave_speed':
            input_text = self.wave_speed_term_input.text()
            self.wave_speed_buffer = sym_2_np_function(input_text)(self.x_coord)
            self.wave_speed = self.wave_speed_buffer
            plot_field(self.wave_speed_fig.figure, self.wave_speed_fig.canvas,
                       self.x_coord, self.wave_speed_buffer, title='c(x)')

        self.update_plot()
        print(f"{input_type.replace('_', ' ').capitalize()}: {input_text}")


    def unified_slider_changed(self, value: int, name: Optional[str] = None, i: Optional[int] = None) -> None:
        r"""Updates simulation parameters or boundary conditions based on slider inputs.

        Args:
            value (int): The current value of the slider.
            name (Optional[str], optional): Specifies the name of the simulation parameter. Defaults to None.
            i (Optional[int], optional): Specifies the index for the boundary condition slider. Defaults to None.
        """
        adjusted_value = value / 100.0  # Adjust value scaling similar to sliders above
        if name is not None:
            self.sliders[name][1].setText(f"{adjusted_value:.3f}")
            self.slider_values[name] = adjusted_value

            if name == 'source':
                plot_field(self.source_fig.figure, self.source_fig.canvas,
                           self.x_coord, adjusted_value, title='s(x)')
            elif name == 'wave_speed':
                plot_field(self.wave_speed_fig.figure, self.wave_speed_fig.canvas,
                           self.x_coord, adjusted_value, title='c(x)')
        elif i is not None:
            self.sliders_bc[i][1].setText(f"{adjusted_value:.3f}")
            self.robin_list[i] = adjusted_value

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

    def combo_selection_changed(self, index):
        r"""Called when the ComboBox selection changes"""
        self.wave_type = index
        self.show_eq_form()
        self.update_plot()
        print(f"Selected Wave Equation Form: {self.combo_box.itemText(index)}")

    def update_plot(self):
        r"""Updates the interactive plot based on the current values of the sliders, line edits, or checkbox status.

        This method takes the current slider values from `self.slider_values` and passes them as arguments to
        `self.interactive_plot` to refresh the plot. It is called in response to changes made using sliders,
        entering text into line edits, toggling a checkbox, or changing the selection in a combo box,
        ensuring the plot accurately reflects all current input settings.
        """
        self.interactive_plot(**self.slider_values)

    def interactive_plot(self, b_val, c_1, c_2, c_3, wave_speed, source, mu_value):
        r"""Generates an interactive plot based on provided coefficients and other initial/boundary conditions.

        This function takes the coefficients for various terms, values for wave speed term and source term,
        and uses them along with internal states such as initial conditions, boundary conditions, and coordinate
        grids to generate an interactive plot of the model's behavior.

        Args:
            b_val, c_1, c_2, c_3, mu_value (float): Coefficients for the general PDE terms.
            wave_speed (float): Diffusion coefficient used in the simulation (ignored if wave_speed_field is provided).
            source (float): Source term coefficient used in the simulation (ignored if source_field is provided).
        """
        coef_list = (b_val, c_1, c_2, c_3)
        generate_plot_wave(model=model,
                           config=config,
                           u_ic=self.g,
                           ut_ic=self.h,
                           x_coord=self.x_coord,
                           t_coord=self.t_coord,
                           coef_list=coef_list,
                           wave_speed=wave_speed if self.wave_speed is None else self.wave_speed,
                           wave_type=self.wave_type,
                           source=source if self.source is None else self.source,
                           mu_value=mu_value,
                           periodic=self.periodic,
                           use_mur_l=self.use_mur_l,
                           robin_theta_l=self.robin_list[0],
                           bc_value_l=self.robin_list[1],
                           use_mur_r=self.use_mur_r,
                           robin_theta_r=self.robin_list[2],
                           bc_value_r=self.robin_list[3],
                           figure=self.fig.figure,
                           canvas=self.fig.canvas,
                           ic_latex=to_latex(self.ic_displace_input.text()),
                           ic_speed_latex=to_latex(self.ic_speed_input.text()))

if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    config, _ = load_config("configs/inference_pdeformer-L-wave.yaml")
    model = get_model(config, compute_type=mstype.float32)

    app = QApplication(sys.argv)
    window = MainWindow()

    window.show()
    app.exec_()
