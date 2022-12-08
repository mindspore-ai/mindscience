mindsponge.common.get_aligned_seq
=================================

.. py:function:: mindsponge.common.get_aligned_seq(gt_seq, pr_seq)

    输入两条蛋白质序列，对序列进行对齐，分别输出对齐后的序列以及两条序列的相同位置（顺序无要求）。

    参数：
        - **gt_seq** (str) - 一条蛋白质序列，如"ABAAABAA"。
        - **pr_seq** (str) - 需要比对的另外一条序列，如"AAABBBA"。

    返回：
        - **aligned_gt_seq** (str) 一条蛋白质序列。
        - **aligned_info** (str) 两条序列的不同点。
        - **aligned_pr_seq** (str) 另一条蛋白质序列。
