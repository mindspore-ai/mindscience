import numpy as np

def print_save_tensor(x, npy_name):
    print("----------------")
    print(f"{npy_name}.shape: ", x.shape)
    print(f"{npy_name}.dtype: ", x.dtype)
    print(f"{npy_name}: ", x)
    print("----------------")
    np.save(f"./dump_ms/{npy_name}_ms.npy", x.float().asnumpy())