# Common Use Cases

This document provides real-world use cases and patterns for common meteorological data processing tasks with ecCodes.

## Use Case 1: Weather Forecast Analysis

Extract and analyze weather forecast data from GRIB files.

```python
import eccodes
import numpy as np

def analyze_forecast(forecast_file, parameter, level):
    """Analyze forecast data for a specific parameter and level."""
    
    with open(forecast_file, 'rb') as f:
        results = []
        
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            # Check if this message matches our criteria
            shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
            msg_level = eccodes.codes_get(msg_id, 'level', default=-1)
            
            if shortName == parameter and msg_level == level:
                step = eccodes.codes_get(msg_id, 'step')
                values = eccodes.codes_get_values(msg_id)
                
                # Calculate statistics
                stats = {
                    'step': step,
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
                
                results.append(stats)
            
            eccodes.codes_release(msg_id)
    
    return results

# Example: Analyze 2m temperature forecasts
results = analyze_forecast('forecast.grib', 't2m', 0)

for result in results:
    print(f"Step {result['step']}h: "
          f"Mean={result['mean']:.2f}, "
          f"Min={result['min']:.2f}, "
          f"Max={result['max']:.2f}")
```

## Use Case 2: Regional Climate Analysis

Extract climate data for a specific region and calculate climatological statistics.

```python
import eccodes
import numpy as np

def extract_regional_climate(data_files, lat_north, lat_south, lon_west, lon_east):
    """Extract regional climate data from multiple files."""
    
    regional_data = []
    
    for filename in data_files:
        with open(filename, 'rb') as f:
            msg_id = eccodes.codes_grib_new_from_file(f)
            
            if msg_id is not None:
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
                
                # Find regional indices
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
                
                # Extract regional values
                if lat_indices and lon_indices:
                    regional_values = values[np.ix_(lat_indices, lon_indices)]
                    regional_data.append(regional_values.flatten())
                
                eccodes.codes_release(msg_id)
    
    # Calculate climatological statistics
    if regional_data:
        all_values = np.concatenate(regional_data)
        climatology = {
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values),
            'count': len(all_values)
        }
        return climatology
    
    return None

# Example: Analyze regional temperature climatology
data_files = ['jan_2020.grib', 'jan_2021.grib', 'jan_2022.grib']
climatology = extract_regional_climate(data_files, 50, 30, -130, -110)

print(f"Regional Climatology:")
print(f"  Mean: {climatology['mean']:.2f} K")
print(f"  Std: {climatology['std']:.2f} K")
print(f"  Range: {climatology['min']:.2f} - {climatology['max']:.2f} K")
```

## Use Case 3: Satellite Data Processing

Process satellite radiance data from GRIB files.

```python
import eccodes
import numpy as np

def process_satellite_data(radiance_file):
    """Process satellite radiance data."""
    
    with open(radiance_file, 'rb') as f:
        results = []
        
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            # Get channel information
            try:
                channel = eccodes.codes_get(msg_id, 'satelliteSpectralChannel')
            except:
                channel = None
            
            if channel is not None:
                # Get radiance values
                values = eccodes.codes_get_values(msg_id)
                
                # Convert to brightness temperature (simplified)
                # In practice, use proper radiance-to-bt conversion
                bt = 1.0 / (np.log(1.0 / values + 1.0) * 1.44e-2)
                
                # Calculate statistics
                stats = {
                    'channel': channel,
                    'mean_bt': float(np.mean(bt)),
                    'min_bt': float(np.min(bt)),
                    'max_bt': float(np.max(bt))
                }
                
                results.append(stats)
            
            eccodes.codes_release(msg_id)
    
    return results

# Example: Process satellite radiance data
results = process_satellite_data('satellite_radiances.grib')

for result in results:
    print(f"Channel {result['channel']}: "
          f"BT Mean={result['mean_bt']:.2f} K, "
          f"Range={result['min_bt']:.2f}-{result['max_bt']:.2f} K")
```

## Use Case 4: Observational Data Processing

Process observational data from BUFR files.

```python
import eccodes

def process_observations(obs_file):
    """Process observational data from BUFR file."""
    
    observations = []
    
    with open(obs_file, 'rb') as f:
        while True:
            msg_id = eccodes.codes_bufr_new_from_file(f)
            if msg_id is None:
                break
            
            # Unpack message
            eccodes.codes_set(msg_id, 'unpack', 1)
            
            # Get number of subsets
            num_subsets = eccodes.codes_get(msg_id, 'numberOfSubsets')
            
            for i in range(num_subsets):
                eccodes.codes_set(msg_id, 'subsetNumber', i + 1)
                
                try:
                    obs = {
                        'latitude': eccodes.codes_get(msg_id, 'latitude'),
                        'longitude': eccodes.codes_get(msg_id, 'longitude'),
                        'temperature': eccodes.codes_get(msg_id, 'airTemperature'),
                        'pressure': eccodes.codes_get(msg_id, 'pressure'),
                        'wind_speed': eccodes.codes_get(msg_id, 'windSpeed')
                    }
                    observations.append(obs)
                except:
                    pass
            
            eccodes.codes_release(msg_id)
    
    return observations

# Example: Process surface observations
observations = process_observations('surface_observations.bufr')

print(f"Processed {len(observations)} observations")
for obs in observations[:5]:  # Show first 5
    print(f"  ({obs['latitude']:.2f}, {obs['longitude']:.2f}): "
          f"T={obs['temperature']:.2f} K, "
          f"P={obs['pressure']:.2f} hPa")
```

## Use Case 5: Model Output Statistics

Calculate model output statistics from ensemble forecasts.

```python
import eccodes
import numpy as np

def calculate_ensemble_statistics(ensemble_file, parameter, step):
    """Calculate ensemble statistics for a specific step."""
    
    ensemble_members = []
    
    with open(ensemble_file, 'rb') as f:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            # Check if this message matches our criteria
            shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
            msg_step = eccodes.codes_get(msg_id, 'step', default=-1)
            
            if shortName == parameter and msg_step == step:
                values = eccodes.codes_get_values(msg_id)
                ensemble_members.append(values)
            
            eccodes.codes_release(msg_id)
    
    if ensemble_members:
        # Stack ensemble members
        ensemble_array = np.array(ensemble_members)
        
        # Calculate statistics
        mean = np.mean(ensemble_array, axis=0)
        std = np.std(ensemble_array, axis=0)
        min = np.min(ensemble_array, axis=0)
        max = np.max(ensemble_array, axis=0)
        
        return {
            'mean': mean,
            'std': std,
            'min': min,
            'max': max,
            'num_members': len(ensemble_members)
        }
    
    return None

# Example: Calculate ensemble statistics for 2m temperature at step 6h
stats = calculate_ensemble_statistics('ensemble_forecast.grib', 't2m', 6)

if stats:
    print(f"Ensemble Statistics (Step 6h):")
    print(f"  Number of members: {stats['num_members']}")
    print(f"  Ensemble mean: {np.mean(stats['mean']):.2f} K")
    print(f"  Ensemble spread: {np.mean(stats['std']):.2f} K")
    print(f"  Global min: {np.min(stats['min']):.2f} K")
    print(f"  Global max: {np.max(stats['max']):.2f} K")
```

## Use Case 6: Data Validation

Validate GRIB data against expected ranges and quality criteria.

```python
import eccodes
import numpy as np

def validate_grib_data(filename, expected_params):
    """Validate GRIB data against expected parameters."""
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    with open(filename, 'rb') as f:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
            
            if shortName in expected_params:
                expected = expected_params[shortName]
                
                # Check level
                level = eccodes.codes_get(msg_id, 'level', default=None)
                if level != expected['level']:
                    validation_results['warnings'].append(
                        f"Level mismatch for {shortName}: expected {expected['level']}, got {level}"
                    )
                
                # Check units
                units = eccodes.codes_get(msg_id, 'units', default='unknown')
                if units != expected['units']:
                    validation_results['warnings'].append(
                        f"Units mismatch for {shortName}: expected {expected['units']}, got {units}"
                    )
                
                # Check value range
                values = eccodes.codes_get_values(msg_id)
                value_min = np.min(values)
                value_max = np.max(values)
                
                if value_min < expected['min_value']:
                    validation_results['errors'].append(
                        f"Value below minimum for {shortName}: {value_min:.2f} < {expected['min_value']}"
                    )
                    validation_results['valid'] = False
                
                if value_max > expected['max_value']:
                    validation_results['errors'].append(
                        f"Value above maximum for {shortName}: {value_max:.2f} > {expected['max_value']}"
                    )
                    validation_results['valid'] = False
            
            eccodes.codes_release(msg_id)
    
    return validation_results

# Example: Validate forecast data
expected_params = {
    't2m': {
        'level': 0,
        'units': 'K',
        'min_value': 200.0,
        'max_value': 350.0
    },
    'sp': {
        'level': 0,
        'units': 'Pa',
        'min_value': 50000.0,
        'max_value': 110000.0
    }
}

results = validate_grib_data('forecast.grib', expected_params)

if results['valid']:
    print("✅ Validation passed")
else:
    print("❌ Validation failed")
    for error in results['errors']:
        print(f"  ERROR: {error}")

for warning in results['warnings']:
    print(f"  WARNING: {warning}")
```

## Use Case 7: Time Series Extraction

Extract time series for specific locations from forecast data.

```python
import eccodes
import numpy as np

def extract_time_series(forecast_file, lat, lon, parameter):
    """Extract time series for a specific location."""
    
    time_series = []
    
    with open(forecast_file, 'rb') as f:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
            
            if shortName == parameter:
                # Get grid information
                Ni = eccodes.codes_get(msg_id, 'Ni')
                Nj = eccodes.codes_get(msg_id, 'Nj')
                lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint')
                lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint')
                lat_inc = eccodes.codes_get(msg_id, 'jDirectionIncrement')
                lon_inc = eccodes.codes_get(msg_id, 'iDirectionIncrement')
                
                # Find nearest grid point
                j = int(round((lat_first - lat) / lat_inc))
                i = int(round((lon - lon_first) / lon_inc))
                
                # Check bounds
                if 0 <= j < Nj and 0 <= i < Ni:
                    # Get values
                    values = eccodes.codes_get_values(msg_id)
                    value = values[j * Ni + i]
                    
                    # Get time information
                    step = eccodes.codes_get(msg_id, 'step')
                    
                    time_series.append({
                        'step': step,
                        'value': float(value)
                    })
            
            eccodes.codes_release(msg_id)
    
    return time_series

# Example: Extract 2m temperature time series for New York
time_series = extract_time_series('forecast.grib', 40.7, -74.0, 't2m')

print("Time Series for New York (40.7°N, 74.0°W):")
for ts in time_series:
    print(f"  Step {ts['step']}h: {ts['value']:.2f} K")
```

## Use Case 8: Data Format Conversion

Convert GRIB data to other formats for analysis.

```python
import eccodes
import numpy as np
import json

def grib_to_json_dict(grib_file):
    """Convert GRIB file to JSON-serializable dictionary."""
    
    data = {
        'file': grib_file,
        'messages': []
    }
    
    with open(grib_file, 'rb') as f:
        msg_count = 0
        
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            msg_count += 1
            
            # Get basic metadata
            message = {
                'message_number': msg_count,
                'shortName': eccodes.codes_get(msg_id, 'shortName', default='unknown'),
                'name': eccodes.codes_get(msg_id, 'name', default='unknown'),
                'units': eccodes.codes_get(msg_id, 'units', default='unknown'),
                'dataDate': eccodes.codes_get(msg_id, 'dataDate', default='unknown'),
                'dataTime': eccodes.codes_get(msg_id, 'dataTime', default='unknown'),
                'step': eccodes.codes_get(msg_id, 'step', default='unknown')
            }
            
            # Get value statistics
            values = eccodes.codes_get_values(msg_id)
            message['statistics'] = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
            data['messages'].append(message)
            
            eccodes.codes_release(msg_id)
    
    return data

# Example: Convert GRIB to JSON
data = grib_to_json_dict('forecast.grib')

# Save to JSON file
with open('forecast.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Converted {len(data['messages'])} messages to JSON")
")
```

## Use Case 9: Precipitation Analysis

Analyze precipitation accumulation and intensity.

```python
import eccodes
import numpy as np

def analyze_precipitation(precip_file):
    """Analyze precipitation data."""
    
    with open(precip_file, 'rb') as f:
        results = []
        
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
            
            if shortName == 'tp':  # Total precipitation
                # Get time information
                step = eccodes.codes_get(msg_id, 'step')
                start_step = eccodes.codes_get(msg_id, 'startStep', default=0)
                end_step = eccodes.codes_get(msg_id, 'endStep', default=step)
                
                # Get precipitation values
                values = eccodes.codes_get_values(msg_id)
                
                # Calculate statistics
                total_precip = np.sum(values)
                mean_precip = np.mean(values)
                max_precip = np.max(values)
                
                # Calculate area with significant precipitation (> 1 mm)
                significant_area = np.sum(values > 1.0)
                
                result = {
                    'step': step,
                    'start_step': start_step,
                    'end_step': end_step,
                    'total_precip': float(total_precip),
                    'mean_precip': float(mean_precip),
                    'max_precip': float(max_precip),
                    'significant_area': int(significant_area)
                }
                
                results.append(result)
            
            eccodes.codes_release(msg_id)
    
    return results

# Example: Analyze precipitation
results = analyze_precipitation('precipitation.grib')

for result in results:
    print(f"Step {result['step']}h: "
          f"Total={result['total_precip']:.2f} mm, "
          f"Max={result['max_precip']:.2f} mm, "
          f"Area with >1mm={result['significant_area']} points")
```

## Use Case 10: Vertical Profile Extraction

Extract vertical profiles (e.g., soundings) from GRIB data.

```python
import eccodes
import numpy as np

def extract_vertical_profile(grib_file, lat, lon, parameter):
    """Extract vertical profile for a specific location."""
    
    profile = []
    
    with open(grib_file, 'rb') as f:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
            
            if shortName == parameter:
                # Get level information
                level = eccodes.codes_get(msg_id, 'level', default=None)
                level_type = eccodes.codes_get(msg_id, 'typeOfLevel', default='unknown')
                
                if level_type == 'isobaricInhPa':
                    # Get grid information
                    Ni = eccodes.codes_get(msg_id, 'Ni')
                    Nj = eccodes.codes_get(msg_id, 'Nj')
                    lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint')
                    lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint')
                    lat_inc = eccodes.codes_get(msg_id, 'jDirectionIncrement')
                    lon_inc = eccodes.codes_get(msg_id, 'iDirectionIncrement')
                    
                    # Find nearest grid point
                    j = int(round((lat_first - lat) / lat_inc))
                    i = int(round((lon - lon_first) / lon_inc))
                    
                    # Check bounds
                    if 0 <= j < Nj and 0 <= i < Ni:
                        # Get values
                        values = eccodes.codes_get_values(msg_id)
                        value = values[j * Ni + i]
                        
                        profile.append({
                            'level': level,
                            'value': float(value)
                        })
            
            eccodes.codes_release(msg_id)
    
    # Sort by level (descending)
    profile.sort(key=lambda x: x['level'], reverse=True)
    
    return profile

# Example: Extract temperature sounding
profile = extract_vertical_profile('analysis.grib', 40.0, -100.0, 't')

print("Temperature Sounding:")
for p in profile:
    print(f"  {p['level']:4d} hPa: {p['value']:.2f} K")
```

## Best Practices

### 1. Always Use Context Managers

```python
# Good: Use context managers
with open('file.grib', 'rb') as f:
    msg_id = eccodes.codes_grib_new_from_file(f)
    # Process
    eccodes.codes_release(msg_id)

# Bad: Manual file handling
f = open('file.grib', 'rb')
msg_id = eccodes.codes_grib_new_from_file(f)
# What if an exception occurs?
f.close()
```

### 2. Release Resources Promptly

```python
# Good: Release in finally block
msg_id = eccodes.codes_grib_new_from_file(f)
try:
    process(msg_id)
finally:
    eccodes.codes_release(msg_id)
```

### 3. Validate Input Data

```python
# Always validate input
if not os.path.exists(filename):
    raise FileNotFoundError(f"File not found: {filename}")

# Validate data ranges
if np.min(values) < min_threshold or np.max(values) > max_threshold:
    raise ValueError(f"Values out of range")
```

### 4. Handle Missing Data

```python
# Handle missing data gracefully
try:
    value = eccodes.codes_get(msg_id, 'optional_key')
except eccodes.EcCodesError:
    value = default_value

# Check for NaN values
if np.any(np.isnan(values)):
    print("Warning: NaN values present")
```

### 5. Use Appropriate Data Structures

```python
# Use dictionaries for metadata
metadata = {
    'shortName': shortName,
    'level': level,
    'step': step
}

# Use lists for time series
time_series = [{'step': s, 'value': v} for s, v in zip(steps, values)]

# Use numpy arrays for numerical data
values = np.array(data, dtype=np.float32)
```

## References

- ECMWF GRIB API: https://confluence.ecmwf.int/display/UDOC/GRIB+API+documentation
- WMO Codes: https://www.wmo.int/pages/prog/www/WMOCodes.html
- NumPy Documentation: https://numpy.org/doc/stable/