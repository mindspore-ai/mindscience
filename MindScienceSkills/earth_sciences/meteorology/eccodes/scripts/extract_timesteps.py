#!/usr/bin/env python3
"""
Extract specific time steps from GRIB files.

Usage:
    python extract_timesteps.py <input.grib> <output.grib> --date <YYYYMMDD> --time <HHMM> --step <step>
    
Examples:
    python extract_timesteps.py input.grib output.grib --date 20240330 --time 1200 --step 0
    python extract_timesteps.py input.grib output.grib --date 20240330
"""

import sys
import argparse
import eccodes


def extract_timesteps(input_file, output_file, date=None, time=None, step=None):
    """Extract specific time steps from GRIB file."""
    try:
        with open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                message_count = 0
                extracted_count = 0
                
                while True:
                    msg_id = eccodes.codes_grib_new_from_file(f_in)
                    if msg_id is None:
                        break
                    
                    message_count += 1
                    
                    try:
                        msg_date = eccodes.codes_get(msg_id, 'dataDate', default=None)
                        msg_time = eccodes.codes_get(msg_id, 'dataTime', default=None)
                        msg_step = eccodes.codes_get(msg_id, 'step', default=None)
                        
                        match = True
                        
                        if date is not None and msg_date != date:
                            match = False
                        
                        if time is not None and msg_time != time:
                            match = False
                        
                        if step is not None and msg_step != step:
                            match = False
                        
                        if match:
                            shortName = eccodes.codes_get(msg_id, 'shortName', default='unknown')
                            eccodes.codes_write(msg_id, f_out)
                            extracted_count += 1
                            print(f"Extracted: {shortName} (date={msg_date}, time={msg_time}, step={msg_step})")
                    
                    except Exception as e:
                        print(f"Error processing message {message_count}: {e}")
                    
                    finally:
                        eccodes.codes_release(msg_id)
                
                print(f"Total messages processed: {message_count}")
                print(f"Messages extracted: {extracted_count}")
                return extracted_count > 0
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract specific time steps from GRIB files')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('output_file', help='Output GRIB file')
    parser.add_argument('--date', type=int, help='Date in YYYYMMDD format')
    parser.add_argument('--time', type=int, help='Time in HHMM format')
    parser.add_argument('--step', type=int, help='Forecast step')
    
    args = parser.parse_args()
    
    if not (args.date or args.time or args.step is not None):
        print("Error: Must specify at least one of --date, --time, or --step")
        sys.exit(1)
    
    success = extract_timesteps(args.input_file, args.output_file, args.date, args.time, args.step)
    
    if not success:
        print("Time step extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()