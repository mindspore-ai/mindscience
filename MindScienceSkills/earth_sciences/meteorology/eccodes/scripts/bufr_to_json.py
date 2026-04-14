#!/usr/bin/env python3
"""
Convert BUFR observational data to JSON format.

Usage:
    python bufr_to_json.py <input.bufr> <output.json> [--compact]
    
Examples:
    python bufr_to_json.py observations.bufr output.json
    python bufr_to_json.py observations.bufr output.json --compact
"""

import sys
import argparse
import json
import eccodes


def bufr_to_json(input_file, output_file, compact=False):
    """Convert BUFR file to JSON format."""
    try:
        messages = []
        
        with open(input_file, 'rb') as f:
            message_count = 0
            
            while True:
                msg_id = eccodes.codes_bufr_new_from_file(f)
                if msg_id is None:
                    break
                
                message_count += 1
                
                try:
                    message_data = {
                        'message_number': message_count,
                        'keys': {}
                    }
                    
                    keys = eccodes.codes_get_keys(msg_id)
                    
                    for key in keys:
                        try:
                            value = eccodes.codes_get(msg_id, key)
                            
                            if isinstance(value, (list, tuple)):
                                if len(value) <= 10:
                                    message_data['keys'][key] = list(value)
                                else:
                                    message_data['keys'][key] = {
                                        'type': 'array',
                                        'length': len(value),
                                        'sample': list(value[:5])
                                    }
                            else:
                                message_data['keys'][key] = value
                        
                        except Exception:
                            message_data['keys'][key] = None
                    
                    messages.append(message_data)
                    print(f"Processed message {message_count}")
                
                except Exception as e:
                    print(f"Error processing message {message_count}: {e}")
                
                finally:
                    eccodes.codes_release(msg_id)
        
        output_data = {
            'file': input_file,
            'total_messages': message_count,
            'messages': messages
        }
        
        with open(output_file, 'w') as f:
            if compact:
                json.dump(output_data, f, separators=(',', ':'))
            else:
                json.dump(output_data, f, indent=2)
        
        print(f"\nSuccessfully converted {message_count} messages to JSON")
        print(f"Output file: {output_file}")
        return True
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert BUFR files to JSON format')
    parser.add_argument('input_file', help='Input BUFR file')
    parser.add_argument('output_file', help='Output JSON file')
    parser.add_argument('--compact', action='store_true', help='Output compact JSON')
    
    args = parser.parse_args()
    
    success = bufr_to_json(args.input_file, args.output_file, args.compact)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()