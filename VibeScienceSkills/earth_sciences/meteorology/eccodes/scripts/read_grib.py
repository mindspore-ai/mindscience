#!/usr/bin/env python3
"""
Read and display GRIB file contents with options for summary, detailed view, or specific field extraction.

Usage:
    python read_grib.py <input.grib> [--summary] [--detailed] [--field <shortName>]
    
Examples:
    python read_grib.py data.grib --summary
    python read_grib.py data.grib --detailed
    python read_grib.py data.grib --field t2m
"""

import sys
import argparse
import eccodes


def print_summary(msg_id):
    """Print summary information about a GRIB message."""
    try:
        shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
        name = eccodes.codes_get(msg_id, 'name', default='unknown')
        units = eccodes.codes_get(msg_id, 'units', default='unknown')
        dataDate = eccodes.codes_get(msg_id, 'dataDate', default='unknown')
        dataTime = eccodes.codes_get(msg_id, 'dataTime', default='unknown')
        step = eccodes.codes_get(msg_id, 'step', default='unknown')
        
        values = eccodes.codes_get_values(msg_id)
        value_count = len(values)
        value_min = float(values.min()) if value_count > 0 else 'N/A'
        value_max = float(values.max()) if value_count > 0 else 'N/A'
        
        print(f"  Field: {shortName} ({name})")
        print(f"  Units: {units}")
        print(f"  Date: {dataDate}, Time: {dataTime}, Step: {step}")
        print(f"  Values: {value_count} points, Min: {value_min}, Max: {value_max}")
        print()
    except Exception as e:
        print(f"  Error reading message: {e}")


def print_detailed(msg_id):
    """Print detailed information about a GRIB message."""
    try:
        keys = eccodes.codes_get_keys(msg_id)
        
        print(f"  Total keys: {len(keys)}")
        print()
        
        for key in keys:
            try:
                value = eccodes.codes_get(msg_id, key)
                print(f"  {key}: {value}")
            except Exception:
                print(f"  {key}: <unable to read>")
        
        print()
    except Exception as e:
        print(f"  Error reading message: {e}")


def print_field(msg_id, target_shortName):
    """Print information for a specific field."""
    try:
        shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
        
        if shortName == target_shortName:
            print(f"Found field: {shortName}")
            
            name = eccodes.codes_get(msg_id, 'name', default='unknown')
            units = eccodes.codes_get(msg_id, 'units', default='unknown')
            dataDate = eccodes.codes_get(msg_id, 'dataDate', default='unknown')
            dataTime = eccodes.codes_get(msg_id, 'dataTime', default='unknown')
            step = eccodes.codes_get(msg_id, 'step', default='unknown')
            
            values = eccodes.codes_get_values(msg_id)
            
            print(f"  Full name: {name}")
            print(f"  Units: {units}")
            print(f"  Date: {dataDate}, Time: {dataTime}, Step: {step}")
            print(f"  Number of values: {len(values)}")
            print(f"  First 10 values: {values[:10]}")
            print()
            return True
    
    except Exception as e:
        print(f"  Error reading field: {e}")
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Read and display GRIB file contents')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('--summary', action='store_true', help='Print summary of each message')
    parser.add_argument('--detailed', action='store_true', help='Print detailed information')
    parser.add_argument('--field', help='Extract specific field by shortName')
    
    args = parser.parse_args()
    
    if not (args.summary or args.detailed or args.field):
        args.summary = True
    
    try:
        with open(args.input_file, 'rb') as f:
            message_count = 0
            field_found = False
            
            while True:
                msg_id = eccodes.codes_grib_new_from_file(f)
                if msg_id is None:
                    break
                
                message_count += 1
                
                if args.field:
                    if print_field(msg_id, args.field):
                        field_found = True
                elif args.detailed:
                    print(f"Message {message_count}:")
                    print_detailed(msg_id)
                elif args.summary:
                    print(f"Message {message_count}:")
                    print_summary(msg_id)
                
                eccodes.codes_release(msg_id)
            
            if args.field and not field_found:
                print(f"Field '{args.field}' not found in file")
                sys.exit(1)
            
            print(f"Total messages processed: {message_count}")
    
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()