## Generating the Input Data of the GraphCast Training Model

### 1. Automatically generating GraphCast model structure data

```shell
python graph_processing_main.py
```

#### Output File and Description

`Output directory`: The output file is stored in the `geometry` directory parallel to the `input_data` directory.

Example: `level=4` and `resolution=0.25` is used to describe the output file and its meaning.
|output file|file Description|Data Format|
|:--------- |:-------------- |:--------- |
|grid_node_features_r0.25.npy| grid node's features|shape of (cos_lat,sin_lon,cos_lon)|
|mesh_node_features_level4_r0.25.npy|mesh node's features whose level is 4| shape of (cos_lat,sin_lon,cos_lon)|
|mesh_edge_normalization_level0_4_r0.25.npy|mesh edge's features whose level is 4| shape of (
length,diff_x,diff_y,diff_z)|
|mesh_edge_sender_level0_4_r0.25.npy|mesh2mesh edge's sender whose level is 4| shape of (idx_mesh,)|
|mesh_edge_receiver_level0_4_r0.25.npy|mesh2mesh edge's receiver whose level is 4| shape of (idx_mesh,)|
|g2m_edge_normalization_level4_r0.25.npy|grid2mesh edge's features whose level is 4| shape of (
length,diff_x,diff_y,diff_z)|
|g2m_sender_level4_r0.25.npy| grid2mesh edge's sender whose level is 4 | shape of (idx_grid,)|
|g2m_receiver_level4_r0.25.npy| grid2mesh edge's receiver whose level is 4 | shape of (idx_mesh,)|
|m2g_edge_normalization_level4_r0.25.npy|mesh2grid edge's features whose level is 4| shape of (
length,diff_x,diff_y,diff_z)|
|m2g_sender_level4_r0.25.npy| mesh2grid edge's sender whose level is 4 | shape of (idx_mesh,)|
|m2g_receiver_level4_r0.25.npy| mesh2grid edge's receiver whose level is 4 | shape of (idx_grid,)|

### 2. Configuration File Description

<p> Default Configuration File Location：
MindEarth/applications/graphcast/graph_processing/graph_construct.yml

#### [Required Configurations]

(1) `level` configuration: a R-refined icosahedral mesh `M^R`. level value range: `0-6`

(2) `input_data` configuration: the directory of input data.

(3) `resolution` configuration: indicates the resolution. The options are `0.25°` or `1.4°`.

(4) `resolution_file` configuration: Name of the grid longitude and latitude data file,
which must be stored in the workdir configuration directory.

(5) `mesh_node` configuration: name of a mesh node file in npy format, The npy file with shape of
`("mesh_lon", "mesh_lat")`. the input file names of different mesh levels are distinguished by the `{level}` format.

For example, if the names of mesh nodes of level `0` and level `1` are `mesh_node_0.npy`, `mesh_node_1.npy`,
the configuration item `mesh_node` should be set to `mesh_node_{level}.npy`.

(6) `mesh_edge` configuration: name of a mesh edge file in npy format, The npy file with shape of
`(mesh_lon1","mesh_lat1","mesh_lon2","mesh_lat2")`. the input file names of different mesh levels are
distinguished by the `{level}` format.

For example, if the names of mesh edge of level `0` and level `1` are `mesh_edge_0.npy`, `mesh_edge_1.npy`,
the configuration item `mesh_edge` should be set to `mesh_edge_{level}.npy`.

#### [Optional Configuration]

(7)  `parallel_configuration_enabled` and `number_of_parallels` configuration:
Sets the degree of parallelism. By default, parallelism is enabled. Set 100 processes for execution,
which can be adjusted as required.

(8) `geometry` configuration: Directory for storing the preprocessed data of the model structure.
Save the default values without modifying them.
</p>
