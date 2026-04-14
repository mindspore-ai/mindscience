#!/usr/bin/env python3
"""
Comprehensive inspection of GRIB message structure, keys, and metadata.

Usage:
    python inspect_grib.py <input.grib> [--message <number>] [--keys] [--values]
    
Examples:
    python inspect_grib.py data.grib
    python inspect_grib.py data.grib --message 1
    python inspect_grib.py data.grib --keys
    python inspect_grib.py data.grib --values
"""

import sys
import argparse
import numpy as np
import eccodes


def inspect_message_structure(msg_id, message_num):
    """Inspect the basic structure of a GRIB message."""
    print(f"\n=== Message {message_num} Structure ===")
    
    try:
        namespace = eccodes.codes_get_namespace(msg_id)
        print(f"Namespace: {namespace}")
        
        edition = eccodes.codes_get(msg_id, 'edition', default='unknown')
        print(f"Edition: GRIB{edition}")
        
        centre = eccodes.codes_get(msg_id, 'centre', default='unknown')
        print(f"Centre: {centre}")
        
        num_keys = eccodes.codes_get_count(msg_id)
        print(f"Number of keys: {num_keys}")
    
    except Exception as e:
        print(f"Error inspecting structure: {e}")


def inspect_keys(msg_id, message_num):
    """Inspect all keys in a GRIB message."""
    print(f"\n=== Message {message_num} Keys ===")
    
    try:
        keys = eccodes.codes_get_keys(msg_id)
        
        print(f"Total keys: {len(keys)}")
        print("\nKeys:")
        
        for i, key in enumerate(keys, 1):
            try:
                value = eccodes.codes_get(msg_id, key)
                key_type = eccodes.codes_get_native_type(msg_id, key)
                
                if isinstance(value, (list, np.ndarray)):
                    value_str = f"<array of {len(value)} elements>"
                else:
                    value_str = str(value)
                
                print(f"  {i:3d}. {key:30s} ({key_type:5s}): {value_str}")
            
            except Exception as e:
                print(f"  {i:3d}. {key:30s}: <error: {e}>")
    
    except Exception as e:
        print(f"Error inspecting keys: {e}")


def inspect_values(msg_id, message_num):
    """Inspect the data values in a GRIB message."""
    print(f"\n=== Message {message_num} Values ===")
    
    try:
        shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
        name = eccodes.codes_get(msg_id, 'name', default='unknown')
        units = eccodes.codes_get(msg_id, 'units', default='unknown')
        
        print(f"Field: {shortName} ({name})")
        print(f"Units: {units}")
        
        values = eccodes.codes_get_values(msg_id)
        
        print(f"\nValue statistics:")
        print(f"  Count: {len(values)}")
        print(f"  Min: {np.min(values):.6f}")
        print(f"  Max: {np.max(values):.6f}")
        print(f"  Mean: {np.mean(values):.6f}")
        print(f"  Std: {np.std(values):.6f}")
        
        print(f"\nFirst 20 values:")
        print(f"  {values[:20]}")
        
        print(f"\nLast 20 values:")
        print(f"  {values[-20:]}")
        
        missing_values = np.sum(np.isnan(values))
        if missing_values > 0:
            print(f"\nMissing values: {missing_values}")
    
    except Exception as e:
        print(f"Error inspecting values: {e}")


def inspect_metadata(msg_id, message_num):
    """Inspect common metadata keys."""
    print(f"\n=== Message {message_num} Metadata ===")
    
    metadata_keys = [
        'shortName', 'name', 'units',
        'dataDate', 'dataTime', 'step', 'stepType',
        'Ni', 'Nj', 'numberOfPoints',
        'latitudeOfFirstGridPoint', 'longitudeOfFirstGridPoint',
        'latitudeOfLastGridPoint', 'longitudeOfLastGridPoint',
        'iDirectionIncrement', 'jDirectionIncrement',
        'centre', 'generatingProcessIdentifier',
        'typeOfLevel', 'level'
    ]
    
    for key in metadata_keys:
        try:
            value = eccodes.codes_get(msg_id, key, default='<not set>')
            print(f"  {key:30s}: {value}")
        except Exception:
            print(f"  {key:30s}: <not available>")


def main():
    parser = argparse.ArgumentParser(description='Inspect GRIB message structure and content')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('--message', type=int, help='Inspect specific message number')
    parser.add_argument('--keys', action='store_true', help='Show all keys')
    parser.add_argument('--values', action='store_true', help='Show value statistics')
    
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'rb') as f:
            message_count = 0
            target_message = args.message if args.message else None
            
            while True:
                msg_id = eccodes.codes_grib_new_from_file(f)
                if msg_id is None:
                    break
                
                message_count += 1
                
                if target_message is None or message_count == target_message:
                    inspect_message_structure(msg_id, message_count)
                    inspect_metadata(msg_id, message_count)
                    
                    if args.keys:
                        inspect_keys(msg_id, message_count)
                    
                    if args.values:
                        inspect_values(msg_id, message_count)
                    
                    if target_message is not None:
                        break
                
                eccodes.codes_release(msg_id)
            
            if target_message is not None and message_count < target_message:
                print(f"Error: Message {target_message} not found (file has {message_count} messages)")
                sys.exit(1)
            
            print(f"\nTotal messages in file: {message_count}")
    
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()