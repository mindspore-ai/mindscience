#!/usr/bin/env python3
"""
Validate GRIB file integrity and structure.

Usage:
    python validate_grib.py <input.grib> [--detailed]
    
Examples:
    python validate_grib.py data.grib
    python validate_grib.py data.grib --detailed
"""

import sys
import argparse
import eccodes


def validate_grib_file(input_file, detailed=False):
    """Validate GRIB file integrity."""
    try:
        with open(input_file, 'rb') as f:
            message_count = 0
            valid_messages = 0
            errors = []
            
            while True:
                msg_id = eccodes.codes_grib_new_from_file(f)
                if msg_id is None:
                    break
                
                message_count += 1
                message_valid = True
                message_errors = []
                
                try:
                    shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
                    edition = eccodes.codes_get(msg_id, 'edition', default='unknown')
                    
                    if detailed:
                        try:
                            values = eccodes.codes_get_values(msg_id)
                            print(f"Message {message_count}: {shortName} (GRIB{edition}) - {len(values)} values")
                        except Exception as e:
                            message_valid = False
                            message_errors.append(f"Cannot read values: {e}")
                    
                    if detailed:
                        try:
                            Ni = eccodes.codes_get(msg_id, 'Ni', default=0)
                            Nj = eccodes.codes_get(msg_id, 'Nj', default=0)
                            print(f"  Grid: {Ni} x {Nj}")
                        except Exception:
                            pass
                
                except Exception as e:
                    message_valid = False
                    message_errors.append(str(e))
                
                if message_valid:
                    valid_messages += 1
                else:
                    errors.append((message_count, message_errors))
                
                eccodes.codes_release(msg_id)
            
            print(f"\nValidation Summary:")
            print(f"  Total messages: {message_count}")
            print(f"  Valid messages: {valid_messages}")
            print(f"  Invalid messages: {len(errors)}")
            
            if errors:
                print(f"\nErrors found:")
                for msg_num, msg_errors in errors:
                    print(f"  Message {msg_num}:")
                    for error in msg_errors:
                        print(f"    - {error}")
            
            return len(errors) == 0
    
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate GRIB file integrity')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed validation information')
    
    args = parser.parse_args()
    
    success = validate_grib_file(args.input_file, args.detailed)
    
    if success:
        print("\n✅ File is valid")
    else:
        print("\n❌ File has errors")
        sys.exit(1)


if __name__ == "__main__":
    main()