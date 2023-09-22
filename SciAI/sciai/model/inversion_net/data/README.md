## Prepare Data

First download any dataset from our [website](https://openfwi-lanl.github.io/docs/data.html#vel) and unzip it into your local directory.

### Load a pair of velocity map and seismic data

For any dataset in _Vel, Fault, Style_  family, the data is saved as `.npy` files, each file contains a batch of 500 samples. `datai.npy` refers to the `i-th` sample of seismic data. To load data and check:

```bash
import numpy as np
# load seismic data
seismic_data = np.load('data1.npy')
print_log(seismic_data.shape) #(500,5,1000,70)
# load velocity map
velocity_map = np.load('model1.npy')
print_log(velocity_map.shape) #(500,1,70,70)
```

### Prepare training and testing set

Note that there are many ways of organizing training and testing dataset, as long as it is compatible with the data loaded module . Whichever way you choose, please refer to the following table for the train/test split.

| Dataset      | Train / test Split | Corresponding `.npy` files                    |
| ------------ | ------------------ | --------------------------------------------- |
| Vel Family   | 24k / 6k           | data(model)1-48.npy / data(model)49-60.npy    |
| Fault Family | 48k / 6k           | data(model)1-96.npy / data(model)97-108.npy   |
| Style Family | 60k / 7k           | data(model)1-120.npy / data(model)121-134.npy |

A convenient way of loading the data is to use a `.txt` file containing the _location+filename_ of all `.npy` files. Take **flatvel-A** as an example, we create `flatvel-a-train.txt`, organized as the follow, and same for `flatvel-a-test.txt`.

```bash
Dataset_directory/data1.npy
Dataset_directory/data2.npy
...
Dataset_directory/data48.npy
```

**To save time, you can download all the text files from the `splitting_files` folder and change to your own directory.**