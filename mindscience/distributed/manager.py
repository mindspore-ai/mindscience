# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Utilities to create and manage orthogonal communication groups used by
different model parallelisms (tensor/context/data/pipeline) in distributed
training. The module builds and initializes backend communication groups so
that code can query group sizes, ranks and names for each parallelism.
"""

from operator import mul
from functools import reduce
from mindspore.mint.distributed import is_initialized
from mindspore.communication import create_group, get_group_size, get_group_rank_from_world_rank, get_rank

_LOCAL_PARALLELISMS_GROUP = {}

class CommGroupBase:
    """
    Base abstraction for a communication group used by different parallelisms.

    This lightweight base class stores the logical parallelism name and the
    underlying backend group metadata (group name, size and the local rank).
    It is used both for uninitialized/singleton parallelisms (where size==1)
    and for fully-initialized groups represented by :class:`CommGroup`.

    Attributes
    ----------
    name : str
        Parallelism key (for example: "tp", "dp", "cp", or combined keys like
        "dp-cp").
    _group_name : Optional[str]
        Name of the backend communication group. If not initialized, accessing
        the `group_name` property will raise a RuntimeError.
    _group_size : int
        Size of the communication group (defaults to 1 for a singleton).
    _group_rank : int
        Rank of the current process within the group (defaults to 0).
    """
    def __init__(self, parallelisms):
        self.name = parallelisms
        self._group_name = None
        self._group_size = 1
        self._group_rank = 0

    @property
    def size(self):
        return self._group_size

    @property
    def rank(self):
        return self._group_rank

    @property
    def group_name(self):
        raise RuntimeError(f"{self.name} group is not initialized.")

    def __str__(self):
        return f"{self.name}: {self._group_name}, group_rank: {self._group_rank}, group_size: {self._group_size}"


class CommGroup(CommGroupBase):
    """
    Represents an initialized communication group.

    A :class:`CommGroup` wraps a backend communication group created by
    `create_group` and provides convenient access to the group's name, size,
    and the current process' rank within that group.

    Parameters
    ----------
    parallelisms : str
        Parallelism key used to identify the group (e.g. "tp", "dp", or
        "dp-cp").
    group_name : str
        Backend name of the created group (usually a string containing the
        parallelism key and the member ranks).
    group_size : int
        Number of ranks in the group.
    """
    def __init__(self, parallelisms, group_name, group_size):
        super().__init__(parallelisms)
        self._group_name = group_name
        self._group_size = group_size
        self._group_rank = self._get_group_rank()

    def _get_group_rank(self):
        rank = get_rank()
        group_rank = get_group_rank_from_world_rank(rank, self._group_name)
        return group_rank

    @property
    def group_name(self):
        return self._group_name


class CommGroupCreator:
    """
    Create and initialize orthogonal communication groups for model parallelism.

    This helper constructs communication groups for different parallelisms (tensor, context,
    data, and pipeline) by computing orthogonal partitions of the global rank space. Groups
    are named using the pattern "<parallelisms>-<rank0>-<rank1>-..." where <parallelisms>
    is the parallelism key requested (for example, "tp" or "dp-cp").

    The class exposes methods to compute group rank id lists and to initialize a
    communication group for the calling process if it belongs to one of the groups.

    Parameters
    ----------
    tp : int
        Size of tensor parallelism.
    cp : int
        Size of context parallelism.
    dp : int
        Size of data parallelism.
    pp : int
        Size of pipeline parallelism (currently expected to be 1 since pipeline
        parallelism is not fully implemented in this module).
    order : str
        A dash-separated string specifying the ordering of dimensions when
        computing orthogonal partitions, e.g. "tp-cp-dp-pp". The order determines
        how the world ranks are decomposed into multi-dimensional indices used to
        form groups.

    Attributes
    ----------
    world_size : int
        Total number of ranks (tp * cp * dp * pp).
    rank : int
        World rank of the current process (obtained from communication backend).
    ordered_size : List[int]
        Sizes of the parallelism dimensions in the configured `order`.

    Public methods
    --------------
    get_group_rank_ids(parallelisms)
        Return a list of lists; each inner list holds world ranks that form a group
        for the given `parallelisms` key (e.g. "tp", "dp-cp").
    init_group(parallelisms)
        Create the communication group for the calling process if it belongs to one
        of the groups; returns a `CommGroup` instance when the process is part of
        a group, otherwise returns None.

    Notes
    -----
    The internal algorithm builds orthogonal groupings so that different parallelism
    axes partition the world ranks without overlap. This allows combining axes such
    as "dp-cp" to form bigger groups while keeping other axes orthogonal.
    """
    def __init__(self, tp, cp, dp, pp, order):
        self.tp = tp
        self.cp = cp
        self.dp = dp
        self.pp = pp
        self.world_size = tp * cp * dp * pp
        self.rank = get_rank()

        self.order = order
        self.ordered_size = []
        for parallelism in order.split("-"):
            self.ordered_size.append(getattr(self, parallelism))

    def _get_mask(self, parallelisms):
        ordered_parallelism = self.order.split("-")
        parallelisms = parallelisms.split("-")
        mask = [p in parallelisms for p in ordered_parallelism]
        return mask

    def _generate_orthogonal_group_rank_ids(self, mask):
        """
        Generate orthogonal groups of world ranks according to a mask.

        Given a boolean `mask` that selects some axes from `self.ordered_size`,
        build orthogonal partitions of the global rank space and return a list
        of groups. Each returned element is a list of world ranks that belong
        to the same communication group for the selected axes.

        Parameters
        ----------
        mask : List[bool]
            Boolean list with the same length as `self.ordered_size`. True
            indicates the axis is included in the group, False indicates it
            is part of the outer grouping.

        Returns
        -------
        List[List[int]]
            A list where each inner list contains world ranks forming a group.
        """
        def prefix_product(a):
            r = [1]
            for v in a:
                r.append(r[-1] * v)
            return r

        def inner_product(a, b):
            return sum(x * y for (x, y) in zip(a, b))

        def decompose(index, size):
            stride = prefix_product(size)
            idx = [(index // d) % s for s, d in zip(size, stride)]
            assert (
                sum(x * y for (x, y) in zip(idx, stride[:-1])) == index
            ), f"idx {index} with size {size} mismatch the return idx {idx}."
            return idx

        masked_size = [s for s, m in zip(self.ordered_size, mask) if m]
        unmasked_size = [s for s, m in zip(self.ordered_size, mask) if not m]

        global_stride = prefix_product(self.ordered_size)
        masked_stride = [s for s, m in zip(global_stride, mask) if m]
        unmasked_stride = [s for s, m in zip(global_stride, mask) if not m]

        group_size = reduce(mul, masked_size)
        num_of_group = self.world_size // group_size

        ranks = []
        for group_index in range(num_of_group):
            decomposed_group_idx = decompose(group_index, unmasked_size)
            rank = []
            for rank_in_group in range(group_size):
                decomposed_rank_idx = decompose(rank_in_group, masked_size)
                rank.append(inner_product(decomposed_rank_idx, masked_stride)
                            + inner_product(decomposed_group_idx, unmasked_stride))
            ranks.append(rank)
        return ranks

    def get_group_rank_ids(self, parallelisms):
        mask = self._get_mask(parallelisms)
        ranks = self._generate_orthogonal_group_rank_ids(mask)
        return ranks

    def init_group(self, parallelisms):
        """
        Initialize and create a communication group for the given parallelisms.

        This method computes the orthogonal group partitions for the requested
        `parallelisms` (e.g. "tp", "dp-cp") and creates the backend group for
        the subset that contains the current world rank. If a group for the
        same `parallelisms` is already present in the local registry and has
        size > 1, a RuntimeError is raised to prevent re-initialization.

        Parameters
        ----------
        parallelisms : str
            Dash-separated key or keys identifying the parallelism axes to
            group (for example "tp", "dp-cp").

        Returns
        -------
        CommGroup or None
            Returns a :class:`CommGroup` instance when the current process is a
            member of a created group, or `None` if the process does not belong
            to any group for the given `parallelisms`.
        """
        if parallelisms in _LOCAL_PARALLELISMS_GROUP and _LOCAL_PARALLELISMS_GROUP[parallelisms].size > 1:
            raise RuntimeError(f"{parallelisms} group is already initialized.")

        group_rank_ids_list = self.get_group_rank_ids(parallelisms)
        for group_rank_ids in group_rank_ids_list:
            if self.rank in group_rank_ids:
                group = parallelisms + "-" + "-".join(map(str, group_rank_ids))
                create_group(group, group_rank_ids)
                return CommGroup(parallelisms, group, len(group_rank_ids))
        return None

def initialize_parallel(
    tensor_parallel_size=1,
    context_parallel_size=1,
    pipeline_parallel_size=1,
    order="tp-cp-dp-pp"
):
    """
    Initialize parallel communication groups for the current distributed world.

    This function validates the communication backend is initialized, computes
    the data parallel size from the global group size and the provided tensor/
    context/pipeline sizes, then uses :class:`CommGroupCreator` to build and
    register communication groups for supported parallelisms into the module
    registry `_LOCAL_PARALLELISMS_GROUP`.

    Parameters
    ----------
    tensor_parallel_size : int
        Size of tensor parallelism (tp).
    context_parallel_size : int
        Size of context parallelism (cp).
    pipeline_parallel_size : int
        Size of pipeline parallelism (pp). Note: pipeline parallelism is
        currently limited and will be coerced to 1 if >1.
    order : str
        Dash-separated ordering of axes used to compose groups (default
        "tp-cp-dp-pp").

    Raises
    ------
    RuntimeError
        If the communication backend is not initialized or the world size is
        not divisible by the requested parallelism product.
    """
    if not is_initialized():
        raise RuntimeError("MindSpore communication is not initialized.")

    if pipeline_parallel_size > 1:
        print("Pipeline parallelism has not yet been implemented, set pipeline_parallel_size = 1.")
        pipeline_parallel_size = 1

    world_size = get_group_size()
    minimum_world_size = tensor_parallel_size * context_parallel_size * pipeline_parallel_size
    if world_size % minimum_world_size != 0:
        raise RuntimeError(
            f"world_size {world_size} is not divisible by tensor_parallel_size {tensor_parallel_size} "
            f"x pipeline_parallel_size {pipeline_parallel_size} x context_parallel_size {context_parallel_size}."
        )

    data_parallel_size = world_size // minimum_world_size

    comm_creator = CommGroupCreator(tp=tensor_parallel_size, cp=context_parallel_size,
                                    dp=data_parallel_size, pp=pipeline_parallel_size, order=order)

    _LOCAL_PARALLELISMS_GROUP["dp"] = comm_creator.init_group("dp") if data_parallel_size > 1 \
                                                                    else CommGroupBase("dp")
    _LOCAL_PARALLELISMS_GROUP["tp"] = comm_creator.init_group("tp") if tensor_parallel_size > 1 \
                                                                    else CommGroupBase("tp")
    _LOCAL_PARALLELISMS_GROUP["cp"] = comm_creator.init_group("cp") if context_parallel_size > 1 \
                                                                    else CommGroupBase("cp")
    _LOCAL_PARALLELISMS_GROUP["dp-cp"] = comm_creator.init_group("dp-cp") if data_parallel_size > 1 \
                                                                          or context_parallel_size > 1 \
                                                                          else CommGroupBase("dp-cp")

def get_data_parallel_rank():
    return _LOCAL_PARALLELISMS_GROUP["dp"].rank

def get_tensor_parallel_rank():
    return _LOCAL_PARALLELISMS_GROUP["tp"].rank

def get_context_parallel_rank():
    return _LOCAL_PARALLELISMS_GROUP["cp"].rank

def get_data_context_parallel_rank():
    return _LOCAL_PARALLELISMS_GROUP["dp-cp"].rank

def get_data_parallel_world_size():
    return _LOCAL_PARALLELISMS_GROUP["dp"].size

def get_tensor_parallel_world_size():
    return _LOCAL_PARALLELISMS_GROUP["tp"].size

def get_context_parallel_world_size():
    return _LOCAL_PARALLELISMS_GROUP["cp"].size

def get_data_context_parallel_world_size():
    return _LOCAL_PARALLELISMS_GROUP["dp-cp"].size

def get_data_parallel_group_name():
    return _LOCAL_PARALLELISMS_GROUP["dp"].group_name

def get_tensor_parallel_group_name():
    return _LOCAL_PARALLELISMS_GROUP["tp"].group_name

def get_context_parallel_group_name():
    return _LOCAL_PARALLELISMS_GROUP["cp"].group_name

def get_data_context_parallel_group_name():
    return _LOCAL_PARALLELISMS_GROUP["dp-cp"].group_name

def get_data_parallel_group():
    return _LOCAL_PARALLELISMS_GROUP["dp"]

def get_tensor_parallel_group():
    return _LOCAL_PARALLELISMS_GROUP["tp"]

def get_context_parallel_group():
    return _LOCAL_PARALLELISMS_GROUP["cp"]

def get_data_context_parallel_group():
    return _LOCAL_PARALLELISMS_GROUP["dp-cp"]
