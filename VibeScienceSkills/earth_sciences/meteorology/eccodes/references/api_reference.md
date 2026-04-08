# ecCodes Python API Reference

This document provides a complete reference of the ecCodes Python API.

## Message Creation and Reading

### codes_grib_new_from_file

Read a GRIB message from a file.

```python
msg_id = eccodes.codes_grib_new_from_file(file_handle)
```

**Parameters:**
- `file_handle`: File handle opened in binary mode

**Returns:**
- `msg_id`: Message ID (integer), or `None` if no more messages

**Example:**
```python
with open('file.grib', 'rb') as f:
    msg_id = eccodes.codes_grib_new_from_file(f)
    if msg_id is not None:
        # Process message
        eccodes.codes_release(msg_id)
```

### codes_bufr_new_from_file

Read a BUFR message from a file.

```python
msg_id = eccodes.codes_bufr_new_from_file(file_handle)
```

**Parameters:**
- `file_handle`: File handle opened in binary mode

**Returns:**
- `msg_id`: Message ID (integer), or `None` if no more messages

**Example:**
```python
with open('file.bufr', 'rb') as f:
    msg_id = eccodes.codes_bufr_new_from_file(f)
    if msg_id is not None:
        # Process message
        eccodes.codes_release(msg_id)
```

### codes_grib_new_from_samples

Create a new GRIB message from a sample template.

```python
msg_id = eccodes.codes_grib_new_from_samples(sample_name)
```

**Parameters:**
- `sample_name`: Name of sample template (e.g., 'GRIB2', 'regular_ll_sfc_grib2')

**Returns:**
- `msg_id`: Message ID (integer)

**Common Samples:**
- `GRIB2`: Basic GRIB2 message
- `regular_ll_sfc_grib2`: Regular lat-lon grid, surface data
- `gg_sfc_grib2`: Gaussian grid, surface data
- `reduced_gg_sfc_grib2`: Reduced Gaussian grid, surface data
- `lambert_sfc_grib2`: Lambert conformal grid, surface data
- `polar_stereographic_sfc_grib2`: Polar stereographic grid, surface data

**Example:**
```python
msg_id = eccodes.codes_grib_new_from_samples('GRIB2')
eccodes.codes_set(msg_id, 'Ni', 360)
eccodes.codes_set(msg_id, 'Nj', 181)
```

### codes_bufr_new_from_samples

Create a new BUFR message from a sample template.

```python
msg_id = eccodes.codes_bufr_new_from_samples(sample_name)
```

**Parameters:**
- `sample_name`: Name of sample template (e.g., 'BUFR4')

**Returns:**
- `msg_id`: Message ID (integer)

**Example:**
```python
msg_id = eccodes.codes_bufr_new_from_samples('BUFR4')
eccodes.codes_set(msg_id, 'dataCategory', 0)
```

## Message Writing

### codes_write

Write a message to a file.

```python
eccodes.codes_write(msg_id, file_handle)
```

**Parameters:**
- `msg_id`: Message ID
- `file_handle`: File handle opened in binary write mode

**Example:**
```python
with open('output.grib', 'wb') as f:
    eccodes.codes_write(msg_id, f)
```

## Message Management

### codes_release

Release a message and free memory.

```python
eccodes.codes_release(msg_id)
```

**Parameters:**
- `msg_id`: Message ID

**Example:**
```python
msg_id = eccodes.codes_grib_new_from_file(f)
try:
    process(msg_id)
finally:
    eccodes.codes_release(msg_id)
```

### codes_clone

Clone a message.

```python
clone_id = eccodes.codes_clone(msg_id)
```

**Parameters:**
- `msg_id`: Message ID to clone

**Returns:**
- `clone_id`: Message ID of cloned message

**Example:**
```python
msg_id = eccodes.codes_grib_new_from_file(f)
clone_id = eccodes.codes_clone(msg_id)

# Modify clone
eccodes.codes_set(clone_id, 'shortName', 'modified')

eccodes.codes_release(msg_id)
eccodes.codes_release(clone_id)
```

## Getting Values

### codes_get

Get a value from a message.

```python
value = eccodes.codes_get(msg_id, key, default=None)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name (string)
- `default`: Default value if key not found (optional)

**Returns:**
- `value`: Value associated with key

**Example:**
```python
shortName = eccodes.codes_get(msg_id, 'shortName')
step = eccodes.codes_get(msg_id, 'step', default=0)
```

### codes_get_long

Get a long integer value.

```python
value = eccodes.codes_get_long(msg_id, key)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name

**Returns:**
- `value`: Long integer value

**Example:**
```python
step = eccodes.codes_get_long(msg_id, 'step')
```

### codes_get_double

Get a double precision floating-point value.

```python
value = eccodes.codes_get_double(msg_id, key)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name

**Returns:**
- `value`: Double precision value

**Example:**
```python
latitude = eccodes.codes_get_double(msg_id, 'latitudeOfFirstGridPoint')
```

### codes_get_string

Get a string value.

```python
value = eccodes.codes_get_string(msg_id, key)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name

**Returns:**
- `value`: String value

**Example:**
```python
shortName = eccodes.codes_get_string(msg_id, 'shortName')
```

### codes_get_values

Get all data values from a message.

```python
values = eccodes.codes_get_values(msg_id)
```

**Parameters:**
- `msg_id`: Message ID

**Returns:**
- `values`: NumPy array of values

**Example:**
```python
import numpy as np

values = eccodes.codes_get_values(msg_id)
print(f"Shape: {values.shape}")
print(f"Mean: {np.mean(values)}")
```

### codes_get_double_array

Get an array of double precision values.

```python
values = eccodes.codes_get_double_array(msg_id, key)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name

**Returns:**
- `values`: List of double precision values

**Example:**
```python
lats = eccodes.codes_get_double_array(msg_id, 'latitudes')
```

### codes_get_long_array

Get an array of long integer values.

```python
values = eccodes.codes_get_long_array(msg_id, key)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name

**Returns:**
- `values`: List of long integer values

**Example:**
```python
bitmap = eccodes.codes_get_long_array(msg_id, 'bitmap')
```

## Setting Values

### codes_set

Set a value in a message.

```python
eccodes.codes_set(msg_id, key, value)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name (string)
- `value`: Value to set

**Example:**
```python
eccodes.codes_set(msg_id, 'shortName', 't2m')
eccodes.codes_set(msg_id, 'step', 6)
```

### codes_set_long

Set a long integer value.

```python
eccodes.codes_set_long(msg_id, key, value)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name
- `value`: Long integer value

**Example:**
```python
eccodes.codes_set_long(msg_id, 'step', 6)
```

### codes_set_double

Set a double precision floating-point value.

```python
eccodes.codes_set_double(msg_id, key, value)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name
- `value`: Double precision value

**Example:**
```python
eccodes.codes_set_double(msg_id, 'latitudeOfFirstGridPoint', 90.0)
```

### codes_set_string

Set a string value.

```python
eccodes.codes_set_string(msg_id, key, value)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name
- `value`: String value

**Example:**
```python
eccodes.codes_set_string(msg_id, 'shortName', 't2m')
```

### codes_set_values

Set all data values in a message.

```python
eccodes.codes_set_values(msg_id, values)
```

**Parameters:**
- `msg_id`: Message ID
- `values`: NumPy array or list of values

**Example:**
```python
import numpy as np

values = np.random.rand(360 * 181)
eccodes.codes_set_values(msg_id, values)
```

### codes_set_double_array

Set an array of double precision values.

```python
eccodes.codes_set_double_array(msg_id, key, values)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name
- `values`: List of double precision values

**Example:**
```python
lats = [90.0, 89.0, 88.0, ...]
eccodes.codes_set_double_array(msg_id, 'latitudes', lats)
```

### codes_set_long_array

Set an array of long integer values.

```python
eccodes.codes_set_long_array(msg_id, key, values)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name
- `values`: List of long integer values

**Example:**
```python
bitmap = [1, 1, 1, 0, 0, ...]
eccodes.codes_set_long_array(msg_id, 'bitmap', bitmap)
```

## Key Information

### codes_get_keys

Get all keys from a message.

```python
keys = eccodes.codes_get_keys(msg_id)
```

**Parameters:**
- `msg_id`: Message ID

**Returns:**
- `keys`: List of key names

**Example:**
```python
keys = eccodes.codes_get_keys(msg_id)
print(f"Number of keys: {len(keys)}")
print(f"First 10 keys: {keys[:10]}")
```

### codes_get_count

Get the number of keys in a message.

```python
count = eccodes.codes_get_count(msg_id)
```

**Parameters:**
- `msg_id`: Message ID

**Returns:**
- `count`: Number of keys

**Example:**
```python
num_keys = eccodes.codes_get_count(msg_id)
print(f"Number of keys: {num_keys}")
```

### codes_get_size

Get the size (number of elements) of a key.

```python
size = eccodes.codes_get_size(msg_id, key)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name

**Returns:**
- `size`: Number of elements

**Example:**
```python
size = eccodes.codes_get_size(msg_id, 'values')
print(f"Number of values: {size}")
```

### codes_get_native_type

Get the native type of a key.

```python
type_name = eccodes.codes_get_native_type(msg_id, key)
```

**Parameters:**
- `msg_id`: Message ID
- `key`: Key name

**Returns:**
- `type_name`: Type name ('int', 'float', 'string', etc.)

**Example:**
```python
key_type = eccodes.codes_get_native_type(msg_id, 'step')
if key_type == 'int':
    value = eccodes.codes_get_long(msg_id, 'step')
```

### codes_get_namespace

Get the namespace of a message.

```python
namespace = eccodes.codes_get_namespace(msg_id)
```

**Parameters:**
- `msg_id`: Message ID

**Returns:**
- `namespace`: Namespace name ('grib1', 'grib2', 'bufr')

**Example:**
```python
namespace = eccodes.codes_get_namespace(msg_id)
print(f"Namespace: {namespace}")
```

## Indexing

### codes_index_new_from_file

Create an index for a file.

```python
index_id = eccodes.codes_index_new_from_file(filename, keys)
```

**Parameters:**
- `filename`: File name
- `keys`: Comma-separated list of keys to index

**Returns:**
- `index_id`: Index ID

**Example:**
```python
index_id = eccodes.codes_index_new_from_file('file.grib', 'shortName,step')
```

### codes_index_select

Select messages by key values.

```python
eccodes.codes_index_select(index_id, key, value)
```

**Parameters:**
- `index_id`: Index ID
- `key`: Key name
- `value`: Value to select

**Example:**
```python
eccodes.codes_index_select(index_id, 'shortName', 't2m')
eccodes.codes_index_select(index_id, 'step', 0)
```

### codes_new_from_index

Get a message from an index.

```python
msg_id = eccodes.codes_new_from_index(index_id)
```

**Parameters:**
- `index_id`: Index ID

**Returns:**
- `msg_id`: Message ID

**Example:**
```python
msg_id = eccodes.codes_new_from_index(index_id)
values = eccodes.codes_get_values(msg_id)
eccodes.codes_release(msg_id)
```

### codes_index_release

Release an index.

```python
eccodes.codes_index_release(index_id)
```

**Parameters:**
- `index_id`: Index ID

**Example:**
```python
index_id = eccodes.codes_index_new_from_file('file.grib', 'shortName')
try:
    eccodes.codes_index_select(index_id, 'shortName', 't2m')
    msg_id = eccodes.codes_new_from_index(index_id)
    eccodes.codes_release(msg_id)
finally:
    eccodes.codes_index_release(index_id)
```

## Error Handling

### EcCodesError

Exception raised for ecCodes errors.

```python
try:
    value = eccodes.codes_get(msg_id, 'nonexistent_key')
except eccodes.EcCodesError as e:
    print(f"Error: {e}")
    print(f"Error code: {e.code}")
    print(f"Error message: {e.message}")
```

## Utility Functions

### codes_grib_multi_support_on

Enable multi-field GRIB message support.

```python
eccodes.codes_grib_multi_support_on()
```

**Example:**
```python
eccodes.codes_grib_multi_support_on()
```

### codes_grib_multi_support_off

Disable multi-field GRIB message support.

```python
eccodes.codes_grib_multi_support_off()
```

**Example:**
```python
eccodes.codes_grib_multi_support_off()
```

### codes_get_api_version

Get the ecCodes API version.

```python
version = eccodes.codes_get_api_version()
```

**Returns:**
- `version`: Version string

**Example:**
```python
version = eccodes.codes_get_api_version()
print(f"ecCodes version: {version}")
```

## Common Patterns

### Reading All Messages

```python
import eccodes

with open('file.grib', 'rb') as f:
    while True:
        msg_id = eccodes.codes_grib_new_from_file(f)
        if msg_id is None:
            break
        
        # Process message
        shortName = eccodes.codes_get(msg_id, 'shortName')
        values = eccodes.codes_get_values(msg_id)
        
        eccodes.codes_release(msg_id)
```

### Creating and Writing a Message

```python
import eccodes
import numpy as np

# Create message from sample
msg_id = eccodes.codes_grib_new_from_samples('GRIB2')

# Set metadata
eccodes.codes_set(msg_id, 'Ni', 360)
eccodes.codes_set(msg_id, 'Nj', 181)
eccodes.codes_set(msg_id, 'shortName', 't2m')
eccodes.codes_set(msg_id, 'dataDate', 20240330)
eccodes.codes_set(msg_id, 'dataTime', 1200)

# Set values
values = np.random.rand(360 * 181) * 50 + 250
eccodes.codes_set_values(msg_id, values)

# Write to file
with open('output.grib', 'wb') as f:
    eccodes.codes_write(msg_id, f)

# Release message
eccodes.codes_release(msg_id)
```

### Using Indexes

```python
import eccodes

# Create index
index_id = eccodes.codes_index_new_from_file('file.grib', 'shortName,step')

# Select messages
eccodes.codes_index_select(index_id, 'shortName', 't2m')
eccodes.codes_index_select(index)id, 'step', 0)

# Get message
msg_id = eccodes.codes_new_from_index(index_id)

# Process message
values = eccodes.codes_get_values(msg_id)

# Clean up
eccodes.codes_release(msg_id)
eccodes.codes_index_release(index_id)
```

## Notes

- Always release messages and indexes to avoid memory leaks
- Use try-finally blocks to ensure resources are released
- File handles must be opened in binary mode
- Indexes provide significant performance improvements for selective access

## References

- ECMWF GRIB API: https://confluence.ecmwf.int/display/UDOC/GRIB+API+documentation
- ecCodes GitHub: https://github.com/ecmwf/eccodes