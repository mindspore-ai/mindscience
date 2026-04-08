#!/usr/bin/env python3
"""
Combine multiple fields from different files into a single GRIB file.

Usage:
    python combine_fields.py <output.grib> --input1 <file1.grib> --field1 <shortName1> --input2 <file2.grib> --field2 <shortName2> ...
    
Examples:
    python combine_fields.py output.grib --input1 temp.grib --field1 t2m --input2 pressure.grib --field2 sp
"""

import sys
import argparse
import eccodes


def extract_field_from_file(input_file, shortName):
    """Extract a specific field from a GRIB file."""
    messages = []
    
    try:
        with open(input_file, 'rb') as f:
            while True:
                msg_id = eccodes.codes_grib_new_from_file(f)
                if msg_id is None:
                    break
                
                try:
                    msg_shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
                    
                    if msg_shortName == shortName:
                        messages.append(msg_id)
                        print(f"Found field '{shortName}' in {input_file}")
                    else:
                        eccodes.codes_release(msg_id)
                
                except Exception as e:
                    print(f"Error processing message: {e}")
                    eccodes.codes_release(msg_id)
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []
    
    return messages


def combine_fields(output_file, input_fields):
    """Combine fields from multiple input files."""
    try:
        all_messages = []
        
        for input_file, shortName in input_fields:
            messages = extract_field_from_file(input_file, shortName)
            
            if not messages:
                print(f"Warning: Field '{shortName}' not found in {input_file}")
            else:
                all_messages.extend(messages)
        
        if not all_messages:
            print("Error: No fields found to combine")
            return False
        
        with open(output_file, 'wb') as f_out:
            for msg_id in all_messages:
                try:
                    eccodes.codes_write(msg_id, f_out)
                except Exception as e:
                    print(f"Error writing message: {e}")
        
        print(f"Successfully combined {len(all_messages)} fields into {output_file}")
        
        for msg_id in all_messages:
            eccodes.codes_release(msg_id)
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        for msg_id in all_messages:
            eccodes.codes_release(msg_id)
        return False


def main():
    parser = argparse.ArgumentParser(description='Combine multiple fields into a single GRIB file')
    parser.add_argument('output_file', help='Output GRIB file')
    
    args, unknown = parser.parse_known_args()
    
    input_fields = []
    i = 0
    while i < len(unknown):
        if unknown[i] == '--input1' and i + 1 < len(unknown):
            input1 = unknown[i + 1]
            if i + 3 < len(unknown) and unknown[i + 2] == '--field1':
                field1 = unknown[i + 3]
                input_fields.append((input1, field1))
                i += 4
            else:
                print("Error: --field1 must follow --input1")
                sys.exit(1)
        elif unknown[i] == '--input2' and i + 1 < len(unknown):
            input2 = unknown[i + 1]
            if i + 3 < len(unknown) and unknown[i + 2] == '--field2':
                field2 = unknown[i + 3]
                input_fields.append((input2, field2))
                i += 4
            else:
                print("Error: --field2 must follow --input2")
                sys.exit(1)
        else:
            i += 1
    
    if not input_fields:
        print("Error: Must specify at least one input field pair (--input1/--field1)")
        print("Example: combine_fields.py output.grib --input1 temp.grib --field1 t2m")
        sys.exit(1)
    
    success = combine_fields(args.output_file, input_fields)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()