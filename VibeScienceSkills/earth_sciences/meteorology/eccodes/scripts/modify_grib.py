#!/usr/bin/env python3
"""
Modify GRIB file metadata or values.

Usage:
    python modify_grib.py <input.grib> <output.grib> --set-key <key> --set-value <value>
    python modify_grib.py <input.grib> <output.grib> --scale-values <factor>
    
Examples:
    python modify_grib.py input.grib output.grib --set-key shortName --set-value t2m
    python modify_grib.py input.grib output.grib --scale-values 1.5
"""

import sys
import argparse
import numpy as np
import eccodes


def modify_metadata(input_file, output_file, key, value):
    """Modify metadata key in GRIB file."""
    try:
        with open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                message_count = 0
                modified_count = 0
                
                while True:
                    msg_id = eccodes.codes_grib_new_from_file(f_in)
                    if msg_id is None:
                        break
                    
                    message_count += 1
                    
                    try:
                        old_value = eccodes.codes_get(msg_id, key, default='<not set>')
                        eccodes.codes_set(msg_id, key, value)
                        
                        shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
                        eccodes.codes_write(msg_id, f_out)
                        
                        modified_count += 1
                        print(f"Modified message {message_count}: {key} = {old_value} → {value}")
                    
                    except Exception as e:
                        print(f"Error modifying message {message_count}: {e}")
                    
                    finally:
                        eccodes.codes_release(msg_id)
                
                print(f"Total messages processed: {message_count}")
                print(f"Messages modified: {modified_count}")
                return modified_count > 0
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def scale_values(input_file, output_file, factor):
    """Scale data values in GRIB file."""
    try:
        with open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                message_count = 0
                modified_count = 0
                
                while True:
                    msg_id = eccodes.codes_grib_new_from_file(f_in)
                    if msg_id is None:
                        break
                    
                    message_count += 1
                    
                    try:
                        shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
                        values = eccodes.codes_get_values(msg_id)
                        
                        values = values * factor
                        eccodes.codes_set_values(msg_id, values)
                        
                        eccodes.codes_write(msg_id, f_out)
                        
                        modified_count += 1
                        print(f"Scaled values in message {message_count}: {shortName} (factor: {factor})")
                    
                    except Exception as e:
                        print(f"Error modifying message {message_count}: {e}")
                    
                    finally:
                        eccodes.codes_release(msg_id)
                
                print(f"Total messages processed: {message_count}")
                print(f"Messages modified: {modified_count}")
                return modified_count > 0
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Modify GRIB file metadata or values')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('output_file', help='Output GRIB file')
    parser.add_argument('--set-key', help='Key to modify')
    parser.add_argument('--set-value', help='New value for key')
    parser.add_argument('--scale-values', type=float, help='Scale factor for values')
    
    args = parser.parse_args()
    
    success = False
    
    if args.set_key and args.set_value is not None:
        success = modify_metadata(args.input_file, args.output_file, args.set_key, args.set_value)
    elif args.scale_values is not None:
        success = scale_values(args.input_file, args.output_file, args.scale_values)
    else:
        print("Error: Must specify either --set-key/--set-value or --scale-values")
        sys.exit(1)
    
    if not success:
        print("Modification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()