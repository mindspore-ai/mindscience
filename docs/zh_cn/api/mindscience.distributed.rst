mindscience.distributed
========================

通信组管理
-----------------

.. mscnautosummary::
    :toctree: distributed
    :nosignatures:

    mindscience.distributed.manager.get_context_parallel_group
    mindscience.distributed.manager.get_context_parallel_group_name
    mindscience.distributed.manager.get_context_parallel_rank
    mindscience.distributed.manager.get_context_parallel_world_size
    mindscience.distributed.manager.get_data_context_parallel_group
    mindscience.distributed.manager.get_data_context_parallel_group_name
    mindscience.distributed.manager.get_data_context_parallel_rank
    mindscience.distributed.manager.get_data_context_parallel_world_size
    mindscience.distributed.manager.get_data_parallel_group
    mindscience.distributed.manager.get_data_parallel_group_name
    mindscience.distributed.manager.get_data_parallel_rank
    mindscience.distributed.manager.get_data_parallel_world_size
    mindscience.distributed.manager.get_tensor_parallel_group
    mindscience.distributed.manager.get_tensor_parallel_group_name
    mindscience.distributed.manager.get_tensor_parallel_rank
    mindscience.distributed.manager.get_tensor_parallel_world_size
    mindscience.distributed.manager.initialize_parallel

张量排布
-----------------

.. mscnautosummary::
    :toctree: distributed
    :nosignatures:

    mindscience.distributed.mappings.all_to_all_from_hidden_to_sequence
    mindscience.distributed.mappings.all_to_all_from_sequence_to_hidden
    mindscience.distributed.mappings.copy_to_all
    mindscience.distributed.mappings.gather_from_hidden
    mindscience.distributed.mappings.gather_from_sequence
    mindscience.distributed.mappings.reduce_from_all
    mindscience.distributed.mappings.reduce_scatter_to_sequence
    mindscience.distributed.mappings.scatter_to_hidden
    mindscience.distributed.mappings.scatter_to_sequence

分布式模块
-------------------------

.. mscnautosummary::
    :toctree: distributed
    :nosignatures:

    mindscience.distributed.modules.ColumnParallelLinear
    mindscience.distributed.modules.RowParallelLinear
    mindscience.distributed.modules.initialize_affine_weight
