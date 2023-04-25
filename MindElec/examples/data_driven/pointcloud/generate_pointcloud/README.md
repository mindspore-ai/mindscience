# **Contents**

- [**Contents**](#contents)
- [**Generation of Point Cloud Data**](#generation-of-point-cloud-data)
- [**Requirements**](#requirements)
- [**Script Description**](#script-description)
    - [**Script and Code Sample**](#script-and-code-sample)
    - [**Exporting json and stp Files**](#exporting-json-and-stp-files)
    - [**Generating Point Cloud Data**](#generating-point-cloud-data)
- [**MindScience Home Page**](#mindscience-home-page)

# **Generation of Point Cloud Data**

In order to convert the electromagnetic simulation model into a pattern that can be recognized by the neural network, we provide a point cloud generation tool to convert the model into point cloud data. The process can be divided into two steps:

## **Exporting Information of Geometry and Material**

MindSpore Elec provides two types of automatic execution scripts for converting CST files into STP files that can be read by Python. The scripts can be used to convert data in batches to implement large-scale electromagnetic simulation.

- **The CST VBA API automatically calls and exports the JSON and STP files**: Open the VBA Macros Editor of the CST software, import the `export_stp.bas` file in the `generate_pointcloud` directory, change the paths of the JSON and STP files to the desired ones, and click `Run` to export the JSON and STP files. The JSON file contains the model port location and the material information corresponding to the STP file.
- **For CST 2019 or later, you can use Python to directly call CST**: Directly call the `export_stp.py` file in the `generate_pointcloud` directory.

## Generating Point Cloud Data

The STP file cannot be directly used as the input of the neural network. It needs to be converted into regular tensor data. MindSpore Elec provides an API for efficiently converting the STP file into the point cloud tensor data. The `generate_cloud_point.py` file in the `generate_pointcloud` directory provides the API calling example.

# **Requirements**

- Hardware (Ascend)
    - Prepare the Ascend AI Processor to set up the hardware environment.
- Framework
    - [MindSpore Elec](https://gitee.com/mindspore/mindscience/tree/master/MindElec)
- For more information, see the following resources:
    - [MindSpore Elec Tutorial](https://www.mindspore.cn/mindelec/docs/en/master/intro_and_install.html)
    - [MindSpore Elec Python API](https://www.mindspore.cn/mindelec/docs/en/master/mindelec.architecture.html)

# **Script Description**

## **Script and Code Sample**

```path
.
└─generete_pointcloud
  ├─README.md
  ├─export_stp.py                             # export json and stp file by python
  ├─export_stp.bas                            # export json and stp file by VAB
  ├─generate_cloud_point.py                   # generate cloud point data
```

## **Exporting json and stp Files**

```shell  
python export_stp.py --cst_path CST_PATH
                     --stp_path STP_PATH
                     --json_path JSON_PATH
```

In the preceding command, `cst_path` specifies the path of the CST file to be exported as the STP file, and `stp_path` and `json_path` specify the paths for storing the exported STP and JSON files, respectively.

## **Generating Point Cloud Data**

```shell  
python generate_cloud_point.py --stp_path STP_PATH
                               --json_path JSON_PATH
                               --material_dir MATERIAL_DIR
                               --sample_nums (500, 2000, 80)
                               --bbox_args (-40., -80., -5., 40., 80., 5.)
```

When using this module, `stp_path` and `json_path` can be configured to specify the paths of the STP and JSON files used to generate the point cloud. `material_dir` specifies the path of the material information corresponding to the STP. The material information is directly exported from the CST software. `sample_nums` specifies the number of point cloud data records generated from the x, y, and z dimensions. `bbox_args` specifies the region where the point cloud data is generated, that is, (x_min, y_min, z_min, x_max, y_max, z_max).

# **MindScience Home Page**

Visit the official website [home page](https://gitee.com/mindspore/mindscience).

