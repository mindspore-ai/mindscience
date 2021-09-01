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
# ============================================================================
"""BD_BARO"""


class BDBARO:
    """BDBARO"""

    def __init__(self, controller, target_pressure, box_length, mode):
        self.constant_pres_convertion = 6.946827162543585e4
        print("START INITIALIZING MC BAROSTAT:\n")
        self.module_name = "bd_baro"
        self.is_controller_printf_initialized = 0
        print("    The target pressure is %.2f bar\n" % (target_pressure * self.constant_pres_convertion))
        self.v0 = box_length[0] * box_length[1] * box_length[2]
        self.new_v = self.v0
        self.dt = 1e-3 if "dt" not in controller.command_set else float(controller.command_set["dt"])
        print("    The dt is %f ps\n" % self.dt)
        self.taup = 1.0 if "tau" not in controller.command_set else float(controller.command_set["tau"])
        print("    The time constant tau is %f ps\n" % self.taup)
        self.compressibility = 4.5e-5 if "compressibility" not in controller.command_set else float(
            controller.command_set["compressibility"])
        print("    The compressibility constant is %f bar^-1\n" % self.compressibility)
        self.update_interval = 10 if "update_interval" not in controller.command_set else int(
            controller.command_set["update_interval"])
        print("    The update_interval is %d\n" % self.update_interval)
        self.system_reinitializing_count = 0
        if mode == 2 and controller.command_set["barostat"] == "berendsen":
            self.is_initialized = 1
        else:
            self.is_initialized = 0

        if self.is_initialized and not self.is_controller_printf_initialized:
            self.is_controller_printf_initialized = 1
        print("END INITIALIZING BERENDSEN BAROSTATn")
