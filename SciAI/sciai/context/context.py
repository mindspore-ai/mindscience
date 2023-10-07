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
# ==============================================================================
"""context"""
import mindspore as ms

from sciai.utils.file_utils import make_sciai_dirs
from sciai.utils.log_utils import print_log, set_log
from sciai.utils.ms_utils import set_seed
from sciai.utils.python_utils import download_resource, print_args


def init_project(mode=None, device_id=None, seed=1234, args=None):
    """
    Initialize one project with context setting, random seed setting, directory creation and log level setting.

    Args:
        mode (Union(int, None)): ms.PYNATIVE_MODE for dynamic graph, ms.GRAPHE_MODE for static graph. If None,
            ms.GRAPHE_MODE would be set. Default: None.
        device_id (Union(int, None)): Set device_id if given, which suppresses args.device_id. Default: None.
        seed (int): Random seed. Default: 1234.
        args (Union(None, Namespace)): Arguments namespace. Default: None.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Raises:
        ValueError: If input parameters are not legal.
    """
    if mode is not None:
        pass
    elif hasattr(args, "mode"):
        mode = args.mode
    else:
        mode = ms.GRAPH_MODE
    device_id = args.device_id if device_id is None and hasattr(args, "device_id") else device_id
    make_sciai_dirs()
    set_log(args)
    set_context_auto(mode, device_id)
    if args is not None:
        print_args(args)
    set_seed(seed)
    if hasattr(args, "download_data") and hasattr(args, "force_download"):
        download_resource(model_name=args.download_data, is_force=args.force_download)


def set_context_auto(mode=ms.GRAPH_MODE, device_id=None):
    """
    Automatically set context as given mode, recognize platform. If `device_id` is None, no card would be set.

    Args:
        mode (int): Mindspore running mode, which can be ms.PYNATIVE_MODE or ms.GRAPH_MODE. Default: ms.GRAPH_MODE.
        device_id (Union(int, None)): Set device_id if given. Default: None.

    Raises:
        ValueError: If device_id is illegal.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from sciai.context import set_context_auto
        >>> set_context_auto(mode=ms.GRAPH_MODE, device_id=2)
    """
    ms.set_context(mode=mode)
    if isinstance(device_id, int):
        ms.set_context(device_id=device_id)
        print_log(f"device_id is set as {device_id}")
