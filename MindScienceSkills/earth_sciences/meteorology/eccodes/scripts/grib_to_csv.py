#!/usr/bin/env python3
"""
Export GRIB data to CSV format with configurable output options.

Usage:
    python grib_to_csv.py <input.grib> <output.csv> [--field <shortName>] [--delimiter <char>]
    
Examples:
    python grib_to_csv.py input.grib output.csv
    python grib_to_csv.py input.grib output.csv --field t2m
    python grib_to_csv.py input.grib output.csv --delimiter ","
"""

import sys
import argparse
import csv
import numpy as np
import eccodes


def export_to_csv(input_file, output_file, field=None, delimiter=','):
    """Export GRIB data to CSV format."""
    try:
        with open(input_file, 'rb') as f_in:
            with open(output_file, 'w', newline='') as f_out:
                writer = csv.writer(f_out, delimiter=delimiter)
                
                message_count = 0
                exported_count = 0
                
                while True:
                    msg_id = eccodes.codes_grib_new_from_file(f_in)
                    if msg_id is None:
                        break
                    
                    message_count += 1
                    
                    try:
                        shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
                        
                        if field is None or shortName == field:
                            name = eccodes.codes_get(msg_id, 'name', default='unknown')
                            units = eccodes.codes_get(msg_id, 'units', default='unknown')
                            dataDate = eccodes.codes_get(msg_id, 'dataDate', default='unknown')
                            dataTime = eccodes.codes_get(msg_id, 'dataTime', default='unknown')
                            step = eccodes.codes_get(msg_id, 'step', default='unknown')
                            
                            values = eccodes.codes_get_values(msg_id)
                            
                            writer.writerow(['# Field Information'])
                            writer.writerow(['shortName', shortName])
                            writer.writerow(['name', name])
                            writer.writerow(['units', units])
                            writer.writerow(['dataDate', dataDate])
                            writer.writerow(['dataTime', dataTime])
                            writer.writerow(['step', step])
                            writer.writerow([])
                            writer.writerow(['# Data Values'])
                            writer.writerow(['index', 'value'])
                            
                            for i, val in enumerate(values):
                                writer.writerow([i, float(val)])
                            
                            writer.writerow([])
                            exported_count += 1
                            print(f"Exported field: {shortName}")
                    
                    except Exception as e:
                        print(f"Error processing message: {e}")
                    
                    finally:
                        eccodes.codes_release(msg_id)
                
                print(f"Total messages processed: {message_count}")
                print(f"Total fields exported: {exported_count}")
                return exported_count > 0
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export GRIB data to CSV format')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('output_file', help='Output CSV file')
    parser.add_argument('--field', help='Export specific field by shortName')
    parser.add_argument('--delimiter', default=',', help='CSV delimiter (default: comma)')
    
    args = parser.parse_args()
    
    success = export_to_csv(args.input_file, args.output_file, args.field, args.delimiter)
    
    if not success:
        print("Export failed or no matching fields found")
        sys.exit(1)


if __name__ == "__main__":
    main()