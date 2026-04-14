# Performance Optimization Guide

This guide covers performance optimization techniques for processing GRIB and BUFR files with ecCodes.

## Memory Management

### Process Messages Sequentially

For large files, process messages one at a time rather than loading all into memory:

```python
import eccodes

def process_large_file(input_file, process_func):
    """Process large GRIB files without loading all messages into memory."""
    
    with open(input_file, 'rb') as f:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            try:
                process_func(msg_id)
            finally:
                eccodes.codes_release(msg_id)
```

### Use Efficient Data Types

Choose appropriate numpy data types to reduce memory usage:

```python
import numpy as np

# For temperature data (typically 200-350 K)
values = np.array(data, dtype=np.float32)  # 4 bytes per value

# For flags or small integers
flags = np.array(data, dtype=np.int8)  # 1 byte per value

# For pressure levels (0-1100 hPa)
pressure = np.array(data, dtype=np.int16)  # 2 bytes per value
```

### Release Resources Promptly

Always release messages and indices immediately after use:

```python
import eccodes

# Good: Release in finally block
msg_id = eccodes.codes_grib_new_from_file(f)
try:
    values = eccodes.codes_get_values(msg_id)
finally:
    eccodes.codes_release(msg_id)

# Good: Release indices
index_id = eccodes.codes_index_new_from_file('file.grib', 'shortName')
try:
    # Use index
    pass
finally:
    eccodes.codes.index_release(index_id)
```

## Indexing

### Create Indexes for Fast Access

Indexes provide fast access to specific messages without reading the entire file:

```python
import eccodes

# Create index
index_id = eccodes.codes_index_new_from_file('file.grib', 'shortName,step')

# Select messages by keys
eccodes.codes_index_select(index_id, 'shortName', 't2m')
eccodes.codes_index_select(index_id, 'step', 0)

# Get selected messages
msg_id = eccodes.codes_new_from_index(index_id)

# Process message
values = eccodes.codes_get_values(msg_id)

# Clean up
eccodes.codes_release(msg_id)
eccodes.codes_index_release(index_id)
```

### Index on Multiple Keys

Index on multiple keys for efficient multi-criteria selection:

```python
import eccodes

# Index on multiple keys
index_id = eccodes.codes_index_new_from_file('file.grib', 
                                             'shortName,step,level')

# Select by multiple criteria
eccodes.codes_index_select(index_id, 'shortName', 't')
eccodes.codes_index_select(index_id, 'step', 6)
eccodes.codes_index_select(index_id, 'level', 500)

# Get selected messages
msg_id = eccodes.codes_new_from_index(index_id)

# Clean up
eccodes.codes_release(msg_id)
eccodes.codes_index_release(index_id)
```

### Index Performance Comparison

```python
import time
import eccodes

# Without index
start = time.time()
with open('large_file.grib', 'rb') as f:
    while True:
        msg_id = eccodes.codes_grib_new_from_file(f)
        if msg_id is None:
            break
        
        shortName = eccodes.codes_get(msg_id, 'shortName')
        if shortName == 't2m':
            pass  # Process
        
        eccodes.codes_release(msg_id)
end = time.time()
print(f"Without index: {end - start:.2f} seconds")

# With index
start = time.time()
index_id = eccodes.codes_index_new_from_file('large_file.grib', 'shortName')
eccodes.codes_index_select(index_id, 'shortName', 't2m')
msg_id = eccodes.codes_new_from_index(index_id)
eccodes.codes_release(msg_id)
eccodes.codes_index_release(index_id)
end = time.time()
print(f"With index: {end - start:.2f} seconds")
```

## File I/O Optimization

### Use Binary Mode

Always open files in binary mode for better performance:

```python
# Good: Binary mode
with open('file.grib', 'rb') as f:
    msg_id = eccodes.codes_grib_new_from_file(f)

# Bad: Text mode (slower)
with open('file.grib', 'r') as f:
    msg_id = eccodes.codes_grib_new_from_file(f)
```

### Buffer File Reads

For repeated access to the same file, consider buffering:

```python
import io

# Read entire file into memory
with open('file.grib', 'rb') as f:
    buffer = io.BytesIO(f.read())

# Read from buffer
msg_id = eccodes.codes_grib_new_from_file(buffer)
```

### Use Memory Mapping

For very large files, use memory mapping:

```python
import mmap

with open('large_file.grib', 'rb') as f:
    # Create memory-mapped file
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    # Read from memory-mapped file
    msg_id = eccodes.codes_grib_new_from_file(mm)
    
    mm.close()
```

## Data Access Optimization

### Access Values Efficiently

Use `codes_get_values()` for bulk access rather than individual values:

```python
import numpy as np

# Good: Bulk access
values = eccodes.codes_get_values(msg_id)
mean = np.mean(values)

# Bad: Individual access
Ni = eccodes.codes_get(msg_id, 'Ni')
Nj = eccodes.codes_get(msg_id, 'Nj')
total = [eccodes.codes_get_double_array_element(msg_id, i) 
         for i in range(Ni * Nj)]
mean = np.mean(total)
```

### Cache Metadata

Cache frequently accessed metadata:

```python
import eccodes

# Cache grid information
grid_info = {
    'Ni': eccodes.codes_get(msg_id, 'Ni'),
    'Nj': eccodes.codes_get(msg_id, 'Nj'),
    'lat_first': eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint'),
    'lon_first': eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint'),
    'lat_inc': eccodes.codes_get(msg_id, 'jDirectionIncrement'),
    'lon_inc': eccodes.codes_get(msg_id, 'iDirectionIncrement')
}

# Use cached information
for i in range(grid_info['Ni']):
    lon = grid_info['lon_first'] + i * grid_info['lon_inc']
```

### Use Native Types

Use native types for better performance:

```python
import eccodes

# Get native type
key_type = eccodes.codes_get_native_type(msg_id, 'step')

# Use appropriate get method
if key_type == 'int':
    value = eccodes.codes_get_long(msg_id, 'step')
elif key_type == 'float':
    value = eccodes.codes_get_double(msg_id, 'step')
else:
    value = eccodes.codes_get(msg_id, 'step')
```

## Parallel Processing

### Process Files in Parallel

Process multiple files in parallel using multiprocessing:

```python
import multiprocessing
import eccodes

def process_file(filename):
    """Process a single file."""
    with open(filename, 'rb') as f:
        msg_id = eccodes.codes_grib_new_from_file(f)
        # Process message
        eccodes.codes_release(msg_id)
    return filename

# Process multiple files in parallel
files = ['file1.grib', 'file2.grib', 'file3.grib']

with multiprocessing.Pool() as pool:
    results = pool.map(process_file, files)
```

### Process Messages in Parallel

For multi-message files, process messages in parallel:

```python
import multiprocessing
import eccodes

def process_message(msg_data):
    """Process a single message."""
    # Process message data
    return result

# Read all messages
messages = []
with open('file.grib', 'rb') as f:
    while True:
        msg_id = eccodes.codes_grib_new_from_file(f)
        if msg_id is None:
            break
        
        values = eccodes.codes_get_values(msg_id)
        messages.append(values)
        
        eccodes.codes_release(msg_id)

# Process in parallel
with multiprocessing.Pool() as pool:
    results = pool.map(process_message, messages)
```

## Caching

### Cache Decoded Messages

Cache decoded messages for repeated access:

```python
import functools
import eccodes

@functools.lru_cache(maxsize=128)
def get_cached_message(filename, message_num):
    """Cache decoded messages."""
    with open(filename, 'rb') as f:
        for i in range(message_num):
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                return None
        
        values = eccodes.codes_get_values(msg_id)
        eccodes.codes_release(msg_id)
        
        return values

# Use cached messages
values1 = get_cached_message('file.grib', 0)
values2 = get_cached_message('file.grib', 0)  # Returns cached value
```

### Cache Indexes

Cache indexes for repeated file access:

```python
import functools
import eccodes

@functools.lru_cache(maxsize=32)
def get_file_index(filename, keys):
    """Cache file indexes."""
    return eccodes.codes_index_new_from_file(filename, keys)

# Use cached index
index_id = get_file_index('file.grib', 'shortName,step')
eccodes.codes_index_select(index_id, 'shortName', 't2m')
```

## Compression

### Use Appropriate Compression

Choose compression method based on data characteristics:

```python
import eccodes

# For smooth fields (temperature, pressure)
eccodes.codes_set(msg_id, 'packingType', 'grid_simple')
eccodes.codes_set(msg_id, 'binaryScaleFactor', 0)
eccodes.codes_set(msg_id, 'decimalScaleFactor', 0)

# For fields with high variability
eccodes.codes_set(msg_id, 'packingType', 'grid_complex')
eccodes.codes_set(msg_id, 'binaryScaleFactor', 2)
eccodes.codes_set(msg_id, 'decimalScaleFactor', 1)

# For categorical data
eccodes.codes_set(msg_id, 'packingType', 'grid_simple')
eccodes.codes_set(msg_id, 'bitsPerValue', 8)
```

### Adjust Scale Factors

Optimize scale factors for better compression:

```python
import numpy as np
import eccodes

values = eccodes.codes_get_values(msg_id)

# Calculate optimal scale factors
value_range = np.max(values) - np.min(values)
decimal_scale = int(np.ceil(np.log10(value_range)))

# Set scale factors
eccodes.codes_set(msg_id, 'decimalScaleFactor', decimal_scale)
```

## Batch Processing

### Process Batches of Messages

Process messages in batches to balance memory and performance:

```python
import eccodes

def process_batch(messages, batch_size=100):
    """Process messages in batches."""
    results = []
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        
        # Process batch
        batch_results = [process_message(msg) for msg in batch]
        results.extend(batch_results)
    
    return results
```

### Write Batches to Disk

Write messages in batches to reduce I/O overhead:

```python
import eccodes

def write_messages(messages, output_file, batch_size=100):
    """Write messages in batches."""
    
    with open(output_file, 'wb') as f:
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            
            for msg_id in batch:
                eccodes.codes_write(msg_id, f)
```

## Performance Profiling

### Profile Code Execution

Use profiling to identify bottlenecks:

```python
import time
import cProfile

def profile_processing():
    """Profile GRIB processing."""
    
    pr = cProfile.Profile()
    pr.enable()
    
    # Process file
    with open('file.grib', 'rb') as f:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            values = eccodes.codes_get_values(msg_id)
            
            eccodes.codes_release(msg_id)
    
    pr.disable()
    pr.print_stats(sort='cumtime')
```

### Measure Memory Usage

Monitor memory usage during processing:

```python
import psutil
import os

def measure_memory():
    """Measure memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    print(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    print(f"Virtual memory: {mem_info.vms / 1024 / 1024:.2f} MB")
```

## Best Practices

### 1. Use Indexes for Repeated Access

```python
# Good: Use index
index_id = eccodes.codes_index_new_from_file('file.grib', 'shortName')
eccodes.codes_index_select(index_id, 'shortName', 't2m')
msg_id = eccodes.codes_new_from_index(index_id)

# Bad: Scan entire file
with open('file.grib', 'rb') as f:
    while True:
        msg_id = eccodes.codes_grib_new_from_file(f)
        if msg_id is None:
            break
        
        if eccodes.codes_get(msg_id, 'shortName') == 't2m':
            break
        
        eccodes.codes_release(msg_id)
```

### 2. Process Messages Sequentially

```python
# Good: Sequential processing
with open('file.grib', 'rb') as f:
    while True:
        msg_id = eccodes.codes_grib_new_from_file(f)
        if msg_id is None:
            break
        
        process(msg_id)
        eccodes.codes_release(msg_id)

# Bad: Load all messages
messages = []
with open('file.grib', 'rb') as f:
    while True:
        msg_id = eccodes.codes_grib_new_from_file(f)
        if msg_id is None:
            break
        
        messages.append(msg_id)

for msg_id in messages:
    process(msg_id)
    eccodes.codes_release(msg_id)
```

### 3. Release Resources Promptly

```python
# Good: Release in finally
msg_id = eccodes.codes_grib_new_from_file(f)
try:
    process(msg_id)
finally:
    eccodes.codes_release(msg_id)

# Bad: May leak resources
msg_id = eccodes.codes_grib_new_from_file(f)
process(msg_id)
# What if process() raises an exception?
eccodes.codes_release(msg_id)
```

### 4. Use Appropriate Data Types

```python
# Good: Use efficient data types
values = np.array(data, dtype=np.float32)

# Bad: Use default (float64)
values = np.array(data)
```

### 5. Cache Frequently Used Data

```python
# Good: Cache metadata
grid_info = get_grid_info(msg_id)

# Bad: Repeatedly access metadata
for i in range(1000):
    Ni = eccodes.codes_get(msg_id, 'Ni')
```

## Performance Tips

1. **Use indexes** - For repeated access to specific messages
2. **Process sequentially** - Avoid loading entire files into memory
3. **Release resources** - Always release messages and indices
4. **Use efficient data types** - Choose appropriate numpy dtypes
5. **Cache metadata** - Cache frequently accessed information
6. **Batch operations** - Process in batches for large datasets
7. **Profile code** - Identify and optimize bottlenecks
8. **Monitor memory** - Track memory usage during processing

## Common Performance Issues

### Issue: High Memory Usage

**Solution:** Process messages sequentially and release resources promptly.

```python
# Process one message at a time
with open('file.grib', 'rb') as f:
    while True:
        msg_id = eccodes.codes_grib_new_from_file(f)
        if msg_id is None:
            break
        
        process(msg_id)
        eccodes.codes_release(msg_id)
```

### Issue: Slow File Access

**Solution:** Use indexes for selective access.

```python
# Create index for fast access
index_id = eccodes.codes_index_new_from_file('file.grib', 'shortName')
eccodes.codes_index_select(index_id, 'shortName', 't2m')
msg_id = eccodes.codes_new_from_index(index_id)
```

### Issue: Repeated Metadata Access

**Solution:** Cache metadata in variables.

```python
# Cache metadata
Ni = eccodes.codes_get(msg_id, 'Ni')
Nj = eccodes.codes_get(msg_id, 'Nj')

# Use cached values
for i in range(Ni * Nj):
    pass
```

## References

- ECMWF GRIB API: https://confluence.ecmwf.int/display/UDOC/GRIB+API+documentation
- Python Performance Tips: https://wiki.python.org/moin/PythonSpeed/PerformanceTips
- NumPy Performance: https://numpy.org/doc/stable/user/basics.performance.html