# GRIB Format Reference

## Overview

GRIB (General Regularly-distributed Information in Binary) is a binary data format designed to store and transmit meteorological data. It is the primary format used by the World Meteorological Organization (WMO) and national weather services.

## GRIB Editions

### GRIB Edition 1
- Original format from 1990s
- Limited parameter tables
- Simple grid definitions
- Still widely used for operational data

### GRIB Edition 2
- Enhanced format introduced in 2003
- Extensible parameter tables
- Complex grid definitions
- Better compression support
- Preferred for new applications

## Message Structure

A GRIB file consists of one or more GRIB messages. Each message contains:

### Sections

**Section 0: Indicator Section**
- Identifies the message as GRIB
- Contains message length
- Specifies GRIB edition (1 or 2)

**Section 1: Product Definition Section**
- Parameter identification (discipline, parameter category, parameter number)
- Temporal information (reference time, forecast time)
- Spatial information (type of level, level value)
- Generating process information

**Section 2: Grid Definition Section**
- Grid type (regular, Gaussian, reduced, etc.)
- Grid dimensions (Ni, Nj)
- Grid resolution
- Scanning mode

**Section 3: Bitmap Section (Optional)**
- Indicates which grid points contain valid data
- Used for irregular data coverage

**Section 4: Data Representation Section**
- Number of data points
- Data representation template
- Binary scale factor, decimal scale factor
- Reference value

**Section 5: Data Section**
- Packed data values
- Compressed using various methods

**Section 6: Bit-Map Section (Optional)**
- Actual bitmap data

**Section 7: End Section**
- Marks end of message (7777 in ASCII)

## Common Grid Types

### Regular Latitude-Longitude Grid (regular_ll)
- Equally spaced points in latitude and longitude
- Defined by:
  - Ni: Number of points in longitude direction
  - Nj: Number of points in latitude direction
  - latitudeOfFirstGridPoint: First latitude
  - longitudeOfFirstGridPoint: First longitude
  - iDirectionIncrement: Longitude increment
  - jDirectionIncrement: Latitude increment

### Gaussian Grid (gaussian)
- Latitude points are Gaussian latitudes
- Used in spectral models
- Defined by:
  - N: Number of latitude points between pole and equator
  - Ni: Number of longitude points

### Reduced Gaussian Grid (reduced_gg)
- Number of longitude points varies with latitude
- Efficient for spectral models
- Defined by:
  - N: Number of latitude points between pole and equator
  - pl: Array of number of points per latitude

### Lambert Conformal Conic (lambert)
- Conformal projection for mid-latitude regions
- Defined by:
  - LaD: Latitude at which the cone is tangent
  - LoV: Longitude of meridian parallel to y-axis
  - Dx, Dy: Grid spacing
  - Latin1, Latin2: Two standard parallels

### Polar Stereographic (polar_stereographic)
- Conformal projection for polar regions
- Defined by:
  - LaD: Latitude at which the plane cuts the earth
  - LoV: Longitude of meridian parallel to y-axis
  - Dx, Dy: Grid spacing

## Parameter Identification

### GRIB1 Parameter Tables
- Table 2: Parameter version
- Table 128: WMO parameter table
- Table 129: NCEP parameter table
- Table 130: ECMWF parameter table

### GRIB2 Parameter Tables
- Discipline: Meteorology (0), Oceanography (1), etc.
- Parameter Category: Temperature (0), Moisture (1), etc.
- Parameter Number: Specific parameter within category

## Temporal Information

### Reference Time
- `dataDate`: Reference date (YYYYMMDD)
- `dataTime`: Reference time (HHMM)

### Forecast Time
- `step`: Forecast step in hours
- `stepType`: Type of step (forecast, analysis, etc.)

### Valid Time
- Calculated as reference time + forecast step
- Can be retrieved using `validityDate` and `validityTime` keys

## Level Types

Common level types include:
- `surface`: Surface level
- `heightAboveGround`: Height above ground (meters)
- `pressureFromGroundLayer`: Pressure levels (hPa)
- `altitudeAboveMeanSeaLevel`: Altitude above MSL (meters)
- `depthBelowLand`: Depth below land surface (meters)
- `depthBelowSea`: Depth below sea level (meters)
- `isobaricInhPa`: Isobaric levels (hPa)
- `sigma`: Sigma levels (dimensionless)
- `hybrid`: Hybrid model levels

## Data Representation

### Packing Methods
- Simple packing: Original GRIB packing method
- Complex packing: Enhanced compression
- JPEG 2000: Wavelet-based compression
- PNG: Lossless compression

### Scale Factors
- `binaryScaleFactor`: Power of 2 scaling
- `decimalScaleFactor`: Power of 10 scaling
- Used to optimize precision and file size

## Best Practices

1. **Use GRIB2 for new applications** - Better extensibility and features
2. **Validate grid definitions** - Ensure grid parameters are consistent
3. **Check parameter tables** - Verify parameter identification
4. **Handle missing data** - Use bitmaps or special values
5. **Consider compression** - Use appropriate compression for data type
6. **Document custom tables** - If using local parameter tables, document them

## Troubleshooting

### Common Issues

**"Invalid GRIB message"**
- Check file integrity
- Verify GRIB edition
- Check section lengths

**"Unknown parameter"**
- Verify parameter table
- Check discipline/category/number
- Ensure correct local tables

**"Grid mismatch"**
- Verify grid type
- Check grid dimensions
- Validate grid increments

**"Data packing errors"**
- Check scale factors
- Verify reference value
- Ensure correct packing method

## References

- WMO GRIB Documentation: https://www.wmo.int/pages/prog/www/WMOCodes.html
- ECMWF GRIB API: https://confluence.ecmwf.int/display/UDOC/GRIB+API+documentation
- NCEP GRIB Documentation: https://www.nco.ncep.noaa.gov/pmb/docs/grib2/