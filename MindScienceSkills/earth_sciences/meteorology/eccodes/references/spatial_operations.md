# Spatial Operations Guide

This guide covers spatial processing techniques for GRIB data including subsetting, regridding, and coordinate transformations.

## Spatial Subsetting

### Extracting Geographic Regions

Extract a rectangular region from a GRIB file:

```python
import eccodes
import numpy as np

def extract_region(input_file, output_file, lat_north, lat_south, lon_west, lon_east):
    """Extract a geographic region from a GRIB file."""
    
    with open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            while True:
                msg_id = eccodes.codes_grib_new_from_file(f_in)
                if msg_id is None:
                    break
                
                grid_type = eccodes.codes_get(msg_id, 'gridType', default='unknown')
                
                if grid_type == 'regular_ll':
                    # Get grid information
                    Ni = eccodes.codes_get(msg_id, 'Ni')
                    Nj = eccodes.codes_get(msg_id, 'Nj')
                    lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint')
                    lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint')
                    lat_inc = eccodes.codes_get(msg_id, 'jDirectionIncrement')
                    lon_inc = eccodes.codes_get(msg_id, 'iDirectionIncrement')
                    
                    # Get values
                    values = eccodes.codes_get_values(msg_id)
                    values = values.reshape((Nj, Ni))
                    
                    # Find indices for region
                    lat_indices = []
                    for j in range(Nj):
                        lat = lat_first - j * lat_inc
                        if lat_south <= lat <= lat_north:
                            lat_indices.append(j)
                    
                    lon_indices = []
                    for i in range(Ni):
                        lon = lon_first + i * lon_inc
                        if lon_west <= lon <= lon_east:
                            lon_indices.append(i)
                    
                    if lat_indices and lon_indices:
                        # Extract subset
                        subset = values[np.ix_(lat_indices, lon_indices)]
                        subset = subset.flatten()
                        
                        # Update message
                        eccodes.codes_set(msg_id, 'Ni', len(lon_indices))
                        eccodes.codes_set(msg_id, 'Nj', len(lat_indices))
                        eccodes.codes_set_values(msg_id, subset)
                        
                        # Update grid coordinates
                        if lat_indices:
                            eccodes.codes_set(msg_id, 'latitudeOfFirstGridPoint',
                                            lat_first - lat_indices[0] * lat_inc)
                            eccodes.codes_set(msg_id, 'latitudeOfLastGridPoint',
                                            lat_first - lat_indices[-1] * lat_inc)
                        
                        if lon_indices:
                            eccodes.codes_set(msg_id, 'longitudeOfFirstGridPoint',
                                            lon_first + lon_indices[0] * lon_inc)
                            eccodes.codes_set(msg_id, 'longitudeOfLastGridPoint',
                                            lon_first + lon_indices[-1] * lon_inc)
                        
                        # Write
                        eccodes.codes_write(msg_id, f_out)
                
                eccodes.codes_release(msg_id)
```

### Extracting by Coordinates

Extract data for specific coordinate points:

```python
import eccodes
import numpy as np

def extract_points(input_file, points):
    """Extract data for specific coordinate points.
    
    Args:
        input_file: Input GRIB file
        points: List of (lat, lon) tuples
    
    Returns:
        List of values at specified points
    """
    
    results = []
    
    with open(input_file, 'rb') as f:
        msg_id = eccodes.codes_grib_new_from_file(f)
        
        if msg_id is None:
            return results
        
        grid_type = eccodes.codes_get(msg_id, 'gridType', default='unknown')
        
        if grid_type == 'regular_ll':
            Ni = eccodes.codes_get(msg_id, 'Ni')
            Nj = eccodes.codes_get(msg_id, 'Nj')
            lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint')
            lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint')
            lat_inc = eccodes.codes_get(msg_id, 'jDirectionIncrement')
            lon_inc = eccodes.codes_get(msg_id, 'iDirectionIncrement')
            
            values = eccodes.codes_get_values(msg_id)
            values = values.reshape((Nj, Ni))
            
            for lat, lon in points:
                # Find nearest grid point
                j = int(round((lat_first - lat) / lat_inc))
                i = int(round((lon - lon_first) / lon_inc))
                
                # Check bounds
                if 0 <= j < Nj and 0 <= i < Ni:
                    value = values[j, i]
                    results.append(value)
                else:
                    results.append(None)
        
        eccodes.codes_release(msg_id)
    
    return results
```

## Coordinate Transformations

### Latitude/Longitude to Grid Indices

Convert geographic coordinates to grid indices:

```python
import eccodes

def latlon_to_indices(msg_id, lat, lon):
    """Convert latitude/longitude to grid indices."""
    
    grid_type = eccodes.codes_get(msg_id, 'gridType', default='unknown')
    
    if grid_type == 'regular_ll':
        Ni = eccodes.codes_get(msg_id, 'Ni')
        Nj = eccodes.codes_get(msg_id, 'Nj')
        lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint')
        lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint')
        lat_inc = eccodes.codes_get(msg_id, 'jDirectionIncrement')
        lon_inc = eccodes.codes_get(msg_id, 'iDirectionIncrement')
        
        # Calculate indices
        j = (lat_first - lat) / lat_inc
        i = (lon - lon_first) / lon_inc
        
        return i, j
    
    else:
        raise ValueError(f"Unsupported grid type: {grid_type}")
```

### Grid Indices to Latitude/Longitude

Convert grid indices to geographic coordinates:

```python
import eccodes

def indices_to_latlon(msg_id, i, j):
    """Convert grid indices to latitude/longitude."""
    
    grid_type = eccodes.codes_get(msg_id, 'gridType', default='unknown')
    
    if grid_type == 'regular_ll':
        lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint')
        lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint')
        lat_inc = eccodes.codes_get(msg_id, 'jDirectionIncrement')
        lon_inc = eccodes.codes_get(msg_id, 'iDirectionIncrement')
        
        # Calculate coordinates
        lat = lat_first - j * lat_inc
        lon = lon_first + i * lon_inc
        
        return lat, lon
    
    else:
        raise ValueError(f"Unsupported grid type: {grid_type}")
```

### Get Grid Coordinates

Get arrays of latitude and longitude coordinates:

```python
import eccodes
import numpy as np

def get_grid_coordinates(msg_id):
    """Get arrays of latitude and longitude coordinates."""
    
    grid_type = eccodes.codes_get(msg_id, 'gridType', default='unknown')
    
    if grid_type == 'regular_ll':
        Ni = eccodes.codes_get(msg_id, 'Ni')
        Nj = eccodes.codes_get(msg_id, 'Nj')
        lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint')
        lon_first = eccodes[.]codes_get(msg_id, 'longitudeOfFirstGridPoint')
        lat_inc = eccodes.codes_get(msg_id, 'jDirectionIncrement')
        lon_inc = eccodes.codes_get(msg_id, 'iDirectionIncrement')
        
        # Create coordinate arrays
        lats = lat_first - np.arange(Nj) * lat_inc
        lons = lon_first + np.arange(Ni) * lon_inc
        
        return lats, lons
    
    elif grid_type == 'gaussian':
        Ni = eccodes.codes_get(msg_id, 'Ni')
        Nj = eccodes.codes_get(msg_id, 'Nj')
        
        # Gaussian latitudes (simplified)
        lats = np.linspace(90, -90, Nj)
        lons = np.linspace(0, 359.999, Ni)
        
        return lats, lons
    
    else:
        raise ValueError(f"Unsupported grid type: {grid_type}")
```

## Distance Calculations

### Haversine Distance

Calculate great-circle distance between two points:

```python
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6371.0):
    """Calculate great-circle distance between two points.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
        earth_radius: Earth radius in km (default: 6371 km)
    
    Returns:
        Distance in km
    """
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance = earth_radius * c
    
    return distance
```

### Find Nearest Grid Point

Find the nearest grid point to a given location:

```python
import eccodes
import numpy as np

def find_nearest_point(msg_id, target_lat, target_lon):
    """Find the nearest grid point to a target location."""
    
    lats, lons = get_grid_coordinates(msg_id)
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Calculate distances
    distances = haversine_distance(lat_grid, lon_grid, target_lat, target_lon)
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    j, i = min_idx
    
    return i, j, distances[min_idx]
```

## Regridding

### Nearest Neighbor Regridding

Simple regridding using nearest-neighbour interpolation:

```python
import eccodes
import numpy as np

def regrid_nearest_neighbor(msg_id, new_Ni, new_Nj):
    """Regrid data using nearest-neighbor interpolation."""
    
    # Get original grid
    old_Ni = eccodes.codes_get(msg_id, 'Ni')
    old_Nj = eccodes.codes_get(msg_id, 'Nj')
    
    # Get values
    old_values = eccodes.codes_get_values(msg_id)
    old_values = old_values.reshape((old_Nj, old_Ni))
    
    # Create new grid
    new_values = np.zeros((new_Nj, new_Ni))
    
    # Calculate scaling factors
    scale_i = old_Ni / new_Ni
    scale_j = old_Nj / new_Nj
    
    # Nearest-neighbor interpolation
    for j in range(new_Nj):
        for i in range(new_Ni):
            old_j = int(j * scale_j)
            old_i = int(i * scale_i)
            
            # Clip to bounds
            old_j = min(old_j, old_Nj - 1)
            old_i = min(old_i, old_Ni - 1)
            
            new_values[j, i] = old_values[old_j, old_i]
    
    # Update message
    eccodes.codes_set(msg_id, 'Ni', new_Ni)
    eccodes.codes_set(msg_id, 'Nj', new_Nj)
    eccodes.codes_set_values(msg_id, new_values.flatten())
    
    return new_values
```

### Bilinear Interpolation

Regridding using bilinear interpolation:

```python
import numpy as np

def regrid_bilinear(msg_id, new_Ni, new_Nj):
    """Regrid data using bilinear interpolation."""
    
    # Get original grid
    old_Ni = eccodes.codes_get(msg_id, 'Ni')
    old_Nj = eccodes.codes_get(msg_id, 'Nj')
    
    # Get values
    old_values = eccodes.codes_get_values(msg_id)
    old_values = old_values.reshape((old_Nj, old_Ni))
    
    # Create new grid
    new_values = np.zeros((new_Nj, new_Ni))
    
    # Calculate scaling factors
    scale_i = (old_Ni - 1) / (new_Ni - 1) if new_Ni > 1 else 0
    scale_j = (old_Nj - 1) / (new_Nj - 1) if new_Nj > 1 else 0
    
    # Bilinear interpolation
    for j in range(new_Nj):
        for i in range(new_Ni):
            # Calculate position in old grid
            old_j = j * scale_j
            old_i = i * scale_i
            
            # Get surrounding indices
            j0 = int(old_j)
            i0 = int(old_i)
            j1 = min(j0 + 1, old_Nj - 1)
            i1 = min(i0 + 1, old_Ni - 1)
            
            # Calculate weights
            dj = old_j - j0
            di = old_i - i0
            
            # Bilinear interpolation
            v00 = old_values[j0, i0]
            v01 = old_values[j0, i1]
            v10 = old_values[j1, i0]
            v11 = old_values[j1, i1]
            
            v0 = v00 * (1 - di) + v01 * di
            v1 = v10 * (1 - di) + v11 * di
            
            new_values[j, i] = v0 * (1 - dj) + v1 * dj
    
    # Update message
    eccodes.codes_set(msg_id, 'Ni', new_Ni)
    eccodes.codes_set(msg_id, 'Nj', new_Nj)
    eccodes.codes_set_values(msg_id, new_values.flatten())
    
    return new_values
```

## Spatial Statistics

### Calculate Area Statistics

Calculate statistics for a geographic region:

```python
import numpy as np

def region_statistics(msg_id, lat_north, lat_south, lon_west, lon_east):
    """Calculate statistics for a geographic region."""
    
    # Extract region
    subset = extract_region(msg_id, lat_north, lat_south, lon_west, lon_east)
    
    # Calculate statistics
    stats = {
        'mean': np.mean(subset),
        'std': np.std(subset),
        'min': np.min(subset),
        'max': np.max(subset),
        'median': np.median(subset),
        'count': len(subset),
        'missing': np.sum(np.isnan(subset))
    }
    
    return stats
```

### Find Extremes

Find locations of minimum and maximum values:

```python
import numpy as np

def find_extremes(msg_id):
    """Find locations of minimum and maximum values."""
    
    # Get values
    values = eccodes.codes_get_values(msg_id)
    Ni = eccodes.codes_get(msg_id, 'Ni')
    Nj = eccodes.codes_get(msg_id, 'Nj')
    values = values.reshape((Nj, Ni))
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(values(values)), values.shape)
    min_j, min_i = min_idx
    
    # Find maximum
    max_idx = np.unravel_index(np.argmax(values(values)), values.shape)
    max_j, max_i = max_idx
    
    # Convert to coordinates
    min_lat, min_lon = indices_to_latlon(msg_id, min_i, min_j)
    max_lat, max_lon = indices_to_latlon(msg_id, max_i, max_j)
    
    return {
        'minimum': {
            'value': values[min_idx],
            'latitude': min_lat,
            'longitude': min_lon
        },
        'maximum': {
            'value': values[max_idx],
            'latitude': max_lat,
            'longitude': max_lon
        }
    }
```

## Masking

### Create Land-Sea Mask

Create a mask for land or sea points:

```python
import numpy as np

def create_land_sea_mask(msg_id, land_mask_file):
    """Create a land-sea mask from a land mask file."""
    
    # Load land mask (1 = land, 0 = sea)
    land_mask = np.load(land_mask_file)
    
    # Get grid dimensions
    Ni = eccodes.codes_get(msg_id, 'Ni')
    Nj = eccodes.codes_get(msg_id, 'Nj')
    
    # Resize if necessary
    if land_mask.shape != (Nj, Ni):
        # Simple resizing (use proper interpolation in production)
        from scipy.ndimage import zoom
        scale_j = Nj / land_mask.shape[0]
        scale_i = Ni / land_mask.shape[1]
        land_mask = zoom(land_mask, (scale_j, scale_i), order=0)
    
    return land_mask.astype(bool)
```

### Apply Mask

Apply a mask to GRIB data:

```python
import numpy as np

def apply_mask(msg_id, mask, mask_value=np.nan):
    """Apply a mask to GRIB data.
    
    Args:
        msg_id: GRIB message ID
        mask: Boolean mask (True = keep, False = mask)
        mask_value: Value to use for masked points
    """
    
    # Get values
    values = eccodes.codes_get_values(msg_id)
    
    # Apply mask
    values[~mask] = mask_value
    
    # Set values back
    eccodes.codes_set_values(msg_id, values)
```

## Best Practices

### 1. Check Grid Type

Always verify grid type before spatial operations:

```python
grid_type = eccodes.codes_get(msg_id, 'gridType', default='unknown')

if grid_type != 'regular_ll':
    raise ValueError(f"Unsupported grid type: {grid_type}")
```

### 2. Handle Edge Cases

Handle cases where target region is outside grid:

```python
if not lat_indices or not lon_indices:
    print("Warning: No overlap between region and grid")
    continue
```

### 3. Preserve Metadata

When modifying grids, update related metadata:

```python
# Update grid dimensions
eccodes.codes_set(msg_id, 'Ni', new_Ni)
eccodes.codes_set(msg_id, 'Nj', new_Nj)

# Update grid coordinates
eccodes.codes_set(msg_id, 'latitudeOfFirstGridPoint', new_lat_first)
eccodes.codes_set(msg_id, 'longitudeOfFirstGridPoint', new_lon_first)
```

### 4. Use Appropriate Interpolation

Choose interpolation method based on data type:
- Nearest-neighbor for categorical data
- Bilinear for continuous data
- Conservative for flux data

### 5. Validate Results

After spatial operations, validate results:

```python
# Check for NaN values
if np.any(np.isnan(new_values)):
    print("Warning: NaN values present after operation")

# Check value range
if np.min(new_values) < threshold or np.max(new_values) > max_threshold:
    print("Warning: Values outside expected range")
```

## Common Use Cases

### Extract Regional Average

Calculate average temperature for a region:

```python
def regional_average(input_file, lat_north, lat_south, lon_west, lon_east):
    """Calculate regional average."""
    
    with open(input_file, 'rb') as f:
        msg_id = eccodes.codes_grib_new_from_file(f)
        
        # Extract region
        subset = extract_region(msg_id, lat_north, lat_south, lon_west, lon_east)
        
        # Calculate average
        average = np.mean(subset[~np.isnan(subset)])
        
        eccodes.codes_release(msg_id)
    
    return average
```

### Create Time Series

Extract time series for a location:

```python
def extract_time_series(input_file, lat, lon):
    """Extract time series for a location."""
    
    time_series = []
    
    with open(input_file, 'rb') as f:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            # Get time
            step = eccodes.codes_get(msg_id, 'step')
            
            # Extract value at location
            i, j, _ = find_nearest_point(msg_id, lat, lon)
            
            values = eccodes.codes_get_values(msg_id)
            Ni = eccodes.codes_get(msg_id, 'Ni')
            value = values[j * Ni + i]
            
            time_series.append((step, value))
            
            eccodes.codes_release(msg_id)
    
    return time_series
```

## References

- ECMWF GRIB API: https://confluence.ecmwf.int/display/UDOC/GRIB+API+documentation
- CDMS Documentation: https://cdms.readthedocs.io/
- xarray Documentation: https://xarray.pydata.org/