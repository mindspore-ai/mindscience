#!/usr/bin/env python3
"""
Extract specific fields from GRIB files based on shortName, parameter name, or other criteria.

Usage:
    python extract_field.py <input.grib> <output.grib> --field <shortName>
    python extract_field.py <input.grib> <output.grib> --name <parameter_name>
    python extract_field.py <input.grib> <output.grib> --level <level_type>
    
Examples:
    python extract_field.py input.grib output.grib --field t2m
    python extract_field.py input.grib output.grib --name "2 metre temperature"
    python extract_field.py input.grib output.grib --step 0
"""

import sys
import argparse
import eccodes


def extract_by_shortname(input_file, output_file, shortName):
    """Extract messages matching a specific shortName."""
    extracted_count = 0
    
    try:
        with open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                while True:
                    msg_id = eccodes.codes_grib_new_from_file(f_in)
                    if msg_id is None:
                        break
                    
                    try:
                        msg_shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
                        
                        if msg_shortName == shortName:
                            eccodes.codes_write(msg_id, f_out)
                            extracted_count += 1
                            print(f"Extracted: {msg_shortName}")
                    
                    except Exception as e:
                        print(f"Error processing message: {e}")
                    
                    finally:
                        eccodes.codes_release(msg_id)
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print(f"Total messages extracted: {extracted_count}")
    return extracted_count > 0


def extract_by_name(input_file, output_file, name):
    """Extract messages matching a specific parameter name."""
    extracted_count = 0
    
    try:
        with open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                while True:
                    msg_id = eccodes.codes_grib_new_from_file(f_in)
                    if msg_id is None:
                        break
                    
                    try:
                        msg_name = eccodes.codes_get(msg_id, 'name', default='unknown')
                        
                        if msg_name == name:
                            eccodes.codes_write(msg_id, f_out)
                            extracted_count += 1
                            print(f"Extracted: {msg_name}")
                    
                    except Exception as e:
                        print(f"Error processing message: {e}")
                    
                    finally:
                        eccodes.codes_release(msg_id)
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print(f"Total messages extracted: {extracted_count}")
    return extracted_count > 0


def extract_by_step(input_file, output_file, step):
    """Extract messages matching a specific forecast step."""
    extracted_count = 0
    
    try:
        with open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                while True:
                    msg_id = eccodes.codes_grib_new_from_file(f_in)
                    if msg_id is None:
                        break
                    
                    try:
                        msg_step = eccodes.codes_get(msg_id, 'step', default=-1)
                        
                        if msg_step == step:
                            shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
                            eccodes.codes_write(msg_id, f_out)
                            extracted_count += 1
                            print(f"Extracted: {shortName} (step {msg_step})")
                    
                    except Exception as e:
                        print(f"Error processing message: {e}")
                    
                    finally:
                        eccodes.codes_release(msg_id)
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print(f"Total messages extracted: {extracted_count}")
    return extracted_count > 0


def main():
    parser = argparse.ArgumentParser(description='Extract specific fields from GRIB files')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('output_file', help='Output GRIB file')
    parser.add_argument('--field', help='Extract by shortName')
    parser.add_argument('--name', help='Extract by parameter name')
    parser.add_argument('--step', type=int, help='Extract by forecast step')
    
    args = parser.parse_args()
    
    if not (args.field or args.name or args.step is not None):
        print("Error: Must specify one of --field, --name, or --step")
        sys.exit(1)
    
    success = False
    
    if args.field:
        success = extract_by_shortname(args.input_file, args.output_file, args.field)
    elif args.name:
        success = extract_by_name(args.input_file, args.output_file, args.name)
    elif args.step is not None:
        success = extract_by_step(args.input_file, args.output_file, args.step)
    
    if not success:
        print("Extraction failed or no matching messages found")
        sys.exit(1)


if __name__ == "__main__":
    main()