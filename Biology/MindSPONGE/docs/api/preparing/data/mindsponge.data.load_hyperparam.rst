mindsponge.data.load_hyperparam
===============================

.. py:function:: mindsponge.data.load_hyperparam(ckpt_file_name, prefix="hyperparam", dec_key, dec_mode="AES-GCM")

    从checkpoint文件中加载超参数。

    参数：
        - **ckpt_file_name** (str) - Checkpoint文件名称。
        - **prefix** (Union[str, list[str], tuple[str]]) - 只有开头带有prefix的参数才会被加载。默认值："hyperparam"。
        - **dec_key** (Union[None, bytes]) - 用于解密的字节类型密钥。
        - **dec_mode** (str) - 当dec_key不设置为None时，此参数才有效。指定解密模式为"AES-GCM"和"AES-CBC"。默认值："AES-GCM"。