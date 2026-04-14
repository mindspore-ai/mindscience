# ParaView Readers Filters And Pipeline

## Purpose

Use this reference when selecting readers, applying filters, or repairing pipeline structure.

## Core pipeline chain

Typical chain:

1. reader
2. optional data selection or cleanup
3. one or more analysis or transformation filters
4. representation setup
5. export or render step

## Reader discipline

Rules:

- use a reader compatible with the real dataset type, not only to filename
- inspect available arrays and time steps before building a heavy pipeline
- keep one clear source object per dataset branch

## Supported data formats

| Format | Reader | Typical Use Case | Notes |
|---------|--------|-----------------|-------|
| VTK (.vtk, .vtu) | Legacy VTK files | Legacy format, unstructured grids |
| VTK XML (.vti, .pvtu) | XML-based VTK files | Modern VTK, supports parallel data |
| Exodus II (.e, .ex2) | FEM results from codes | Supports time steps, element data |
| STL (.stl) | 3D printing | Surface meshes only, no field data |
| OBJ (.obj) | 3D graphics | Surface meshes only, no field data |
| PLY (.ply) | 3D point clouds | Point clouds and surface meshes |
| CGNS (.cgns) | CFD simulations | Multi-block structured data |
| EnSight (.case, .ensight) | CFD simulations | Binary format, large datasets |
| OpenFOAM (OpenFOAM reader) | CFD simulations | Direct OpenFOAM case reading |
| XDMF (.xdmf) | Multi-block data | Multi-block structured grids |
| HDF5 (.h5) | Scientific data | Hierarchical data format |
| NetCDF (.nc) | Climate/weather data | Self-describing scientific data |
| DICOM series (.dcm) | Medical imaging | 3D medical images and volumes |
| PNG/JPEG/TIFF | Image files | 2D/3D images as data |

## Common readers

| Reader | Key Features | When to Use |
|--------|--------------|-------------|
| `OpenDataFile` | Auto-detects format, handles most common formats | General purpose, unknown format |
| `ExodusIIReader` | Time steps, element data, point data | FEM analysis, time-series data |
| `CGNSReader` | Multi-block, structured grids | CFD results, multi-block data |
| `EnSightReader` | Binary, large datasets | Large CFD simulations |
| `OpenFOAMReader` | Direct case reading | OpenFOAM post-processing |
| `LegacyVTKFileReader` | VTK files | Legacy VTK data |
| `XMLUnstructuredGridReader` | VTK XML files | Modern VTK data |
| `DICOMImageReader` | Medical imaging | Medical visualization |
| `PLOT3DReader` | CFD structured grids | Multi-block structured grids from aerodynamic simulations |

## Filter discipline

Common filter families:

- clipping and slicing
- thresholding and contouring
- resampling and probing
- plotting or data extraction

### Clipping and slicing filters

| Filter | Purpose | Key Parameters |
|--------|---------|----------------|
| `Clip` | Spatial clipping | Clip Type, Value, Inside Out |
| `Slice` | Extract planes/slices | Slice Type, Slice Normal, Slice Origin |
| `Threshold` | Value-based selection | Scalars, Threshold Between, All Scalars |
| `ExtractSelection` | Extract subset | Selection Source |

### Thresholding and contouring filters

| Filter | Purpose | Key Parameters |
|--------|---------|----------------|
| `Contour` | Generate iso-surfaces | Contour By, Isosurfaces, Compute Normals |
| `ElevationFilter` | Surface extraction | Low Point, High Point |
| `Sample` | Point sampling | Sampling Mode, Proportional Number |

### Resampling and probing filters

| Filter | Purpose | Key Parameters |
|--------|---------|----------------|
| `ResampleToImage` | Convert to image | Sampling Dimensions, Use Input For Sampling |
| `ProbeLocation` | Probe at points | Source, Probe Type |
| `CellSizeToPoint` | Convert cells to points | Vertex Cells |

### Plotting and data extraction filters

| Filter | Purpose | Key Parameters |
|--------|---------|----------------|
| `PlotOverLine` | Plot along line | X Axis, Y Axis, Plot Over |
| `PlotOverTime` | Plot time series | X Axis, Y Axis, Plot Over |
| `ExtractSelection` | Extract to new dataset | Selection Source |

## Pipeline checks

Before exporting or rendering:

- confirm that active arrays are the intended ones
- confirm that the view is showing the intended object
- confirm that the filter output type supports the downstream writer or representation

## Pipeline performance optimization

### For large datasets

- Use `GenerateProcessIds` to identify and process only relevant cells
- Use `GhostCellsGenerator` to create ghost cells for distributed rendering
- Enable `Decimate` to reduce polygon count for visualization
- Use `TriangleStrips` for efficient surface rendering

### For time-series data

- Use `TemporalCache` to cache time steps for faster navigation
- Use `TemporalStatistics` to compute statistics across time
- Consider `TemporalSnapToTimeStep` for precise time control

### For distributed rendering

- Use `D3` filters for distributed processing
- Configure `KDTREE` or `LOD` for efficient spatial queries
- Use `MPI` or `Catalyst` parallel backends

## Failure patterns

| Symptom | Likely cause | First repair |
|----------|--------------|-------------|
| data opens but looks wrong | wrong reader or wrong active arrays | inspect the reader output and array selection |
| a filter is unavailable or fails | upstream data type is incompatible | verify the pipeline object type before adding a filter |
| export format is missing | current data model is incompatible with writer | adjust the pipeline to target output type |