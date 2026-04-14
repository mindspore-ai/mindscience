# GRIB Keys Reference

This document provides a comprehensive reference of GRIB keys organized by category.

## Identification Keys

### Basic Identification
- `shortName`: Short name of the parameter (e.g., 't', 'u', 'v')
- `name`: Full name of the parameter (e.g., 'Temperature', 'U-component of wind')
- `units`: Units of the parameter (e.g., 'K', 'm s-1')
- `cfName`: CF-compliant name (Climate and Forecast conventions)
- `cfVarName`: CF-compliant variable name
- `paramId`: Parameter ID number
- `localTablesVersion`: Local tables version

### GRIB Edition
- `edition`: GRIB edition (1 or 2)
- `grib2LocalSectionNumber`: GRIB2 local section number

### Origin Information
- `centre`: Originating centre (e.g., 98 for ECMWF, 7 for NCEP)
- `centreDescription`: Description of originating centre
- `subCentre`: Sub-centre
- `generatingProcessIdentifier`: Generating process ID
- `generatingProcessDefinedAs`: Description of generating process

## Temporal Keys

### Reference Time
- `dataDate`: Reference date in YYYYMMDD format
- `dataTime`: Reference time in HHMM format
- `refDate`: Reference date (alternative key)
- `refTime`: Reference time (alternative key)

### Forecast Time
- `step`: Forecast step in hours
- `stepType`: Type of step (0: forecast, 1: analysis, 2: forecast from analysis)
- `stepUnits`: Units of step (1: hour, 2: day, etc.)
- `forecastTime`: Forecast time (alternative key)
- `startStep`: Start step for time intervals
- `endStep`: End step for time intervals

### Valid Time
- `validityDate`: Validity date in YYYYMMDD format
- `validityTime`: Validity time in HHMM format
- `validityDateTime`: Combined validity date and time

### Time Processing
- `indicatorOfUnitOfTimeRange`: Unit of time range indicator
- `forecastTimeUnit`: Forecast time unit
- `timeRangeIndicator`: Time range indicator
- `p1`: P1 time value
- `p2`: P2 time value
- `timeInterval`: Time interval

## Spatial Keys

### Grid Dimensions
- `Ni`: Number of points in i-direction (longitude)
- `Nj`: Number of points in j-direction (latitude)
- `numberOfPoints`: Total number of grid points
- `numberOfDataPoints`: Number of data points (may differ due to bitmap)

### Grid Extents
- `latitudeOfFirstGridPoint`: Latitude of first grid point
- `longitudeOfFirstGridPoint`: Longitude of first grid point
- `latitudeOfLastGridPoint`: Latitude of last grid point
- `longitudeOfLastGridPoint`: Longitude of last grid point

### Grid Resolution
- `iDirectionIncrement`: Longitude increment in degrees
- `jDirectionIncrement`: Latitude increment in degrees
- `iScansNegatively`: Whether i-direction scans negatively (0: no, 1: yes)
- `jScansPositively`: Whether j-direction scans positively (0: no, 1: yes)
- `jPointsAreConsecutive`: Whether j-points are consecutive (0: no, 1: yes)

### Grid Type
- `gridType`: Type of grid (e.g., 'regular_ll', 'gaussian', 'lambert')
- `gridTypeDescription`: Description of grid type
- `typeOfGrid`: Type of grid (numeric code)

### Projection Parameters (for projected grids)
- `LaD`: Latitude at which the projection plane cuts the earth
- `LoV`: Longitude of meridian parallel to y-axis
- `Dx`: X-direction grid spacing
- `Dy`: Y-direction grid spacing
- `Latin1`: First standard parallel
- `Latin2`: Second standard parallel
- `OrientatedLon`: Oriented longitude
- `OrientatedLat`: Oriented latitude

### Gaussian Grid Parameters
- `N`: Number of latitude points between pole and equator
- `numberOfOctectsForNumberOfPoints`: Number of octets for number of points
- `pl`: Array of number of points per latitude (for reduced Gaussian)

## Level Keys

### Level Information
- `typeOfLevel`: Type of level (e.g., 'surface', 'heightAboveGround', 'isobaricInhPa')
- `level`: Level value
- `topLevel`: Top level (for layers)
- `bottomLevel`: Bottom level (for layers)
- `levelType`: Level type (numeric code)

### Common Level Types
- `surface`: Surface level
- `heightAboveGround`: Height above ground in meters
- `heightAboveSeaLevel`: Height above sea level in meters
- `isobaricInhPa`: Isobaric level in hPa
- `pressureFromGroundLayer`: Pressure level in hPa
- `sigma`: Sigma level (dimensionless)
- `hybrid`: Hybrid model level
- `depthBelowLand`: Depth below land surface in meters
- `depthBelowSea`: Depth below sea level in meters
- `entireAtmosphere`: Entire atmosphere
- `entireOcean`: Entire ocean
- `tropopause`: Tropopause level
- `maxWind`: Maximum wind level
- `tropopauseLevel`: Tropopause level

## Data Representation Keys

### Data Values
- `values`: Array of data values
- `numberOfValues`: Number of values
- `missingValue`: Value representing missing data
- `bitmapPresent`: Whether bitmap is present (0: no, 1: yes)

### Packing Information
- `binaryScaleFactor`: Binary scale factor (power of 2)
- `decimalScaleFactor`: Decimal scale factor (power of 10)
- `referenceValue`: Reference value for packing
- `bitsPerValue`: Number of bits per value
- `typeOfPacking`: Type of packing method
- `packingType`: Description of packing type

### Compression
- `compressionType`: Type of compression
- `uncompressedDataLength`: Length of uncompressed data
- `compressedDataLength`: Length of compressed data
- `compressionRatio`: Compression ratio

## Product Definition Keys

### Discipline (GRIB2 only)
- `discipline`: Discipline (0: meteorology, 1: oceanography, etc.)
- `disciplineName`: Name of discipline

### Parameter Category (GRIB2 only)
- `parameterCategory`: Parameter category
- `parameterCategoryName`: Name of parameter category
- `parameterNumber`: Parameter number within category

### GRIB1 Parameters
- `table2Version`: Parameter table version
- `indicatorOfParameter`: Parameter indicator
- `parameterIndicator`: Parameter indicator (alternative)

### Statistical Processing
- `statisticalProcessIndicator`: Statistical process indicator
- `statisticalProcessingType`: Type of statistical processing
- `numberOfTimeRange`: Number of time ranges
- `missingDataValues`: Number of missing data values

## Model Keys

### Model Information
- `modelId`: Model identification
- `modelName`: Model name
- `modelNameDescription`: Description of model name
- `productDefinitionTemplateNumber`: Product definition template number

### Analysis/Forecast
- `typeOfProcessedData`: Type of processed data (0: analysis, 1: forecast)
- `productionStatusOfProcessedData`: Production status (0: operational, 1: test, 2: research)
- `typeOfGeneratingProcess`: Type of generating process (0: analysis, 1: initialization, 2: forecast, 3: other)

## Satellite Keys

### Satellite Information
- `satelliteIdentifier`: Satellite identifier
- `satelliteSeries`: Satellite series
- `instrumentIdentifier`: Instrument identifier
- `instrumentType`: Instrument type
- `numberOfSpectralChannels`: Number of spectral channels
- `spectralChannel`: Spectral channel number

### Satellite Grid
- `satelliteGrid`: Satellite grid type
- `numberOfPointsAlongXAxis`: Points along X-axis
- `numberOfPointsAlongYAxis`: Points along Y-axis
- `directionOfXAxis`: Direction of X-axis
- `directionOfYAxis`: Direction of Y-axis

## Ensemble Keys

### Ensemble Information
- `perturbationNumber`: Perturbation number (0: control, 1+: ensemble members)
- `numberOfForecastsInEnsemble`: Number of ensemble members
- `typeOfEnsemble`: Type of ensemble (0: perturbed, 1: multi-model, etc.)
- `ensembleType`: Description of ensemble type

### Ensemble Processing
- `forecastNumber`: Forecast number
- `totalNumberOfForecasts`: Total number of forecasts

## Derived Keys

### Calculated Values
- `min`: Minimum value in data
- `max`: Maximum value in data
- `avg`: Average value in data
- `numberOfMissing`: Number of missing values
- `numberOfCodedValues`: Number of coded values

### Grid Coordinates
- `latitudes`: Array of latitude values
- `longitudes`: Array of longitude values
- `distinctLatitudes`: Array of distinct latitude values
- `distinctLongitudes`: Array of distinct longitude values

## Geographic Keys

### Geographic Area
- `geographicArea`: Geographic area name
- `north`: Northern boundary latitude
- `south`: Southern boundary latitude
- `east`: Eastern boundary longitude
- `west`: Western boundary longitude
- `area`: Geographic area code

### Resolution
- `resolutionAndComponentFlags`: Resolution and component flags
- `resolution`: Resolution (0: low, 1: high)
- `uComponent`: Whether u-component is present
- `vComponent`: Whether v-component is present

## Local Keys

### Local Definitions
- `localDefinition`: Local definition number
- `localDefinitionDescription`: Description of local definition

### User-Defined Keys
- Keys starting with `local_` are user-defined
- Examples: `local_parameter`, `local_table`, `local_section`

## Usage Examples

### Getting Basic Information
```python
import eccodes

msg_id = eccodes.codes_grib_new_from_file(f)

shortName = eccodes.codes_get(msg_id, 'shortName')
name = eccodes.codes_get(msg_id, 'name')
units = eccodes.codes_get(msg_id, 'units')

print(f"Field: {shortName} ({name}) in {units}")
```

### Getting Temporal Information
```python
dataDate = eccodes.codes_get(msg_id, 'dataDate')
dataTime = eccodes.codes_get(msg_id, 'dataTime')
step = eccodes.codes_get(msg_id, 'step')

print(f"Reference: {dataDate} {dataTime}, Step: {step}h")
```

### Getting Spatial Information
```python
gridType = eccodes.codes_get(msg_id, 'gridType')
Ni = eccodes.codes_get(msg_id, 'Ni')
Nj = eccodes.codes_get(msg_id, 'Nj')

print(f"Grid: {gridType}, Size: {Ni} x {Nj}")
```

### Checking for Key Existence
```python
try:
    value = eccodes.codes_get(msg_id, 'customKey')
    print(f"Value: {value}")
except eccodes.EcCodesError:
    print("Key does not exist")
```

## Key Categories Summary

| Category | Common Keys | Purpose |
|----------|-------------|---------|
| Identification | shortName, name, units | Identify parameter |
| Temporal | dataDate, step, validityDate | Time information |
| Spatial | Ni, Nj, gridType | Grid information |
| Level | typeOfLevel, level | Vertical level |
| Data | values, missingValue | Data values |
| Model | centre, generatingProcessIdentifier | Model information |
| Ensemble | perturbationNumber | Ensemble information |

## Notes

- Some keys are GRIB edition-specific
- Some keys may not exist for all message types
- Use `codes_get_keys()` to list all available keys
- Use `codes_get_native_type()` to get key type
- Local keys may vary between centres