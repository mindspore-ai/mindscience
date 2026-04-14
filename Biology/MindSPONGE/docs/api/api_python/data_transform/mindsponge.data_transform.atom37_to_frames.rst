mindsponge.data_transform.atom37_to_frames
============================================

.. py:function:: mindsponge.data_transform.atom37_to_frames(aatype, all_atom_positions, all_atom_mask, is_affine=False)

    根据8个刚性组计算每个氨基酸的局部坐标系，返回shape为 :math:`[N_{res}, 8, 12]`， 其中8表示每个残基依据原子对扭转角的依赖可分成至多8个刚性组，
    分别为一个骨架刚体组和7个由扭转角定义的刚体组（包括3个主链扭转角和4个侧脸扭转角）。12表示9个局部坐标系相对于全局坐标系的旋转矩阵和
    3个局部坐标系相对于全局坐标系的平移矩阵。

    参数：
        - **aatype** (numpy.array) - 氨基酸序列, :math:`[N_{res}]`。
        - **all_atom_positions** (numpy.array) - 所有原子的坐标，用atom37的方式呈现, :math:`[N_{res}, 37, 3]`。
        - **all_atom_mask** (numpy.array) - 所有原子坐标的mask, :math:`[N_{res}, 37]` 。
        - **is_affine** (bool) - 是否进行仿射变换, 默认值： ``False``。

    返回：
        字典，具体内容如下。

        - **rigidgroups_gt_frames** (numpy.array) - 将氨基酸序列位置用刚体变换组表示, :math:`[N_{res}, 8, 12]`。
        - **rigidgroups_gt_exists** (numpy.array) - rigidgroups_gt_frames的mask，表示该刚体变换组是不是存在实验解析获得的真实结构,
          :math:`[N_{res}, 8]`。
        - **rigidgroups_group_exists** (numpy.array) - rigidgroups_gt_frames的mask，表示该刚体变换组根据氨基酸残基的理想结构是否存在,
          :math:`[N_{res}, 8]` 。
        - **rigidgroups_group_is_ambiguous** (numpy.array) - rigidgroups_gt_frames的mask，表示该位置是存在手性对称,
          :math:`[N_{res}, 8]` 。
        - **rigidgroups_alt_gt_frames** (numpy.array) - 将近似氨基酸序列位置用扭转角度表示 :math:`[N_{res}, 8, 12]` 。
        - **backbone_affine_tensor** (numpy.array) - 每个氨基酸局部坐标相对全局坐标的平移与旋转, :math:`[N_{res}, 7]`
          对于最后一维，前四个分量是表征旋转的四元数，代表局部坐标系相对全局坐标系的旋转， 后三个分量是三维空间的平移。
