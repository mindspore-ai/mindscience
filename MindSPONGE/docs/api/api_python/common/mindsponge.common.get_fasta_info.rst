mindsponge.common.get_fasta_info
================================

.. py:function:: mindsponge.common.get_fasta_info(pdb_path)

    给定一条蛋白质pdb文件(输入pdb文件路径)，从pdb文件中获取序列信息并输出序列文本。

    参数：
        - **pdb_path** (str) - 输入pdb文件的路径。

    返回：
        - **fasta_seq** (str) 输出蛋白质pdb文件对应序列，该序列为pdb中氨基酸排列顺序和氨基酸编号无关。如 ``"GSHMGVQ"``。
