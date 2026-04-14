#!/usr/bin/env python3
"""
Extract spatial subsets from GRIB files.

Usage:
    python extract_subset.py <input.grib> <output.grib> --lat-north <lat> --lat-south <lat> --lon-west <lon> --lon-east <lon>
    
Examples:
    python extract_subset.py input.grib output.grib --lat-north 60 --lat-south 30 --lon-west -120 --lon-east -90
"""

import sys
import argparse
import numpy as np
import eccodes


def extract_spatial_subset(input_file, output_file, lat_north, lat_south, lon_west, lon_east):
    """Extract spatial subset from GRIB file."""
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
                        grid_type = eccodes.codes_get(msg_id, 'gridType', default='unknown')
                        
                        if grid_type == 'regular_ll':
                            Ni = eccodes.codes_get(msg_id, 'Ni')
                            Nj = eccodes.codes_get(msg_id, 'Nj')
                            lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint')
                            lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint')
                            lat_last = eccodes.codes_get(msg_id, 'latitudeOfLastGridPoint')
                            lon_last = eccodes.codes_get(msg_id, 'longitudeOfLastGridPoint')
                            
                            lat_inc = eccodes.codes_get(msg_id, 'jDirectionIncrement')
                            lon_inc = eccodes.codes_get(msg_id, 'iDirectionIncrement')
                            
                            values = eccodes.codes_get_values(msg_id)
                            values = values.reshape((Nj, Ni))
                            
                            lat_indices = []
                            lon_indices = []
                            
                            for j in range(Nj):
                                lat = lat_first - j * lat_inc
                                if lat_south <= lat <= lat_north:
                                    lat_indices.append(j)
                            
                            for i in range(Ni):
                                lon = lon_first + i * lon_inc
                                if lon_west <= lon <= lon_east:
                                    lon_indices.append(i)
                            
                            if lat_indices and lon_indices:
                                subset_values = values[np.ix_(lat_indices, lon_indices)]
                                subset_values = subset_values.flatten()
                                
                                eccodes.codes_set(msg_id, 'Ni', len(lon_indices))
                                eccodes.codes_set(msg_id, 'Nj', len(lat_indices))
                                eccodes.codes_set_values(msg_id, subset_values)
                                
                                if lat_indices:
                                    eccodes.codes_set(msg_id, 'latitudeOfFirstGridPoint', 
                                                    lat_first - lat_indices[0] * lat_inc)
                                    eccodes.codes_set(msg_id, 'latitudeOfLastGridPoint',
                                                    lat_first - lat_indices[-1] * lat_inc)
                                
                                if lon_indices:
                                    eccodes.codes_set(msg_id, 'longitudeOfFirstGridPoint',
                                                    lon_first + lon_indices[0] * lon_inc)
                                    eccodes.codes_set(msg_id, 'longitudeOfLastGridPoint',
                                                    lon_first + lon_indices[-1] * lon_inc)
                                
                                eccodes.codes_write(msg_id, f_out)
                                extracted_count += 1
                                print(f"Extracted message {message_count}")
                            else:
                                print(f"No overlap for message {message_count}")
                        
                        else:
                            print(f"Unsupported grid type for message {message_count}: {grid_type}")
                    
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
    parser = argparse.ArgumentParser(description='Extract spatial subset from GRIB files')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('output_file', help='Output GRIB file')
    parser.add_argument('--lat-north', type=float, required=True, help='Northern latitude boundary')
    parser.add_argument('--lat-south', type=float, required=True, help='Southern latitude boundary')
    parser.add_argument('--lon-west', type=float, required=True, help='Western longitude boundary')
    parser.add_argument('--lon-east', type=float, required=True, help='Eastern longitude boundary')
    
    args = parser.parse_args()
    
    success = extract_spatial_subset(args.input_file, args.output_file, 
                                    args.lat_north, args.lat_south, 
                                    args.lon_west, args.lon_east)
    
    if not success:
        print("Subset extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()