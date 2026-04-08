#!/usr/bin/env python3
"""
NCSS query builder and executor.

This script builds and executes NCSS queries for gridded meteorological data.

Usage:
    python query_ncss.py <ncss_url> [options]

Examples:
    python query_ncss.py http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml --time "2024-01-01 00:00" --bbox -125 -65 25 50 --vars Temperature Relative_humidity
    python query_ncss.py http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml --time-range "2024-01-01 00:00" "2024-01-02 00:00" --point -105.0 40.0 --vars Temperature --output temp.nc
"""

import sys
import argparse
import json
from datetime import datetime


def build_query(ncss, args):
    """
    Build NCSS query from command line arguments.

    Args:
        ncss: NCSS client instance
        args: Parsed command line arguments

    Returns:
        NCSSQuery object
    """
    query = ncss.query()

    if args.time:
        try:
            time = datetime.strptime(args.time, '%Y-%m-%d %H:%M')
            query = query.time(time)
        except ValueError:
            print(f"Invalid time format: {args.time}")
            print("Use format: YYYY-MM-DD HH:MM")
            sys.exit(1)

    if args.time_range:
        if len(args.time_range) != 2:
            print("Time range requires two times: start end")
            sys.exit(1)
        try:
            start = datetime.strptime(args.time_range[0], '%Y-%m-%d %H:%M')
            end = datetime.strptime(args.time_range[1], '%Y-%m-%d %H:%M')
            query = query.time_range(start, end)
        except ValueError:
            print(f"Invalid time range format")
            print("Use format: YYYY-MM-DD HH:MM")
            sys.exit(1)

    if args.all_times:
        query = query.all_times()

    if args.bbox:
        if len(args.bbox) != 4:
            print("Bounding box requires four values: west east south north")
            sys.exit(1)
        west, east, south, north = map(float, args.bbox)
        query = query.lonlat_box(west, east, south, north)

    if args.point:
        if len(args.point) != 2:
            print("Point requires two values: lon lat")
            sys.exit(1)
        lon, lat = map(float, args.point)
        query = query.lonlat_point(lon, lat)

    if args.vars:
        if args.vars == ['all']:
            query = query.variables('all')
        else:
            query = query.variables(*args.vars)

    if args.vertical_level:
        level = float(args.vertical_level)
        query = query.vertical_level(level)

    if args.strides:
        if len(args.strides) == 1:
            query = query.strides(spatial=int(args.strides[0]))
        elif len(args.strides) == 2:
            query = query.strides(
                time=int(args.strides[0]),
                spatial=int(args.strides[1])
            )
        else:
            print("Strides requires 1 or 2 values: [spatial] or [time spatial]")
            sys.exit(1)

    if args.format:
        query = query.accept(args.format)

    if args.add_lonlat:
        query = query.add_lonlat(True)

    return query


def execute_query(ncss_url, args):
    """
    Execute NCSS query.

   .

    Args:
        ncss_url: NCSS endpoint URL
        args: Parsed command line arguments
    """
    try:
        from siphon.ncss import NCSS
    except ImportError:
        print("Error: siphon package not installed")
        print("Install with: pip install siphon")
        sys.exit(1)

    print(f"Connecting to NCSS: {ncss_url}")

    try:
        ncss = NCSS(ncss_url)
    except Exception as e:
        print(f"Error connecting to NCSS: {e}")
        sys.exit(1)

    if args.list_vars:
        print("\nAvailable variables:")
        for var in sorted(ncss.variables):
            print(f"  {var}")
        return

    if args.metadata:
        print("\nMetadata:")
        print(f"  Time coverage: {ncss.metadata.time_coverage}")
        print(f"  Spatial coverage: {ncss.metadata.lat_lon_box}")
        print(f"  Variables: {len(ncss.variables)}")
        return

    query = build_query(ncss, args)

    print("\nExecuting query...")
    try:
        if args.raw:
            data = ncss.get_data_raw(query)
            if args.output:
                with open(args.output, 'wb') as f:
                    f.write(data)
                print(f"Raw data saved to: {args.output}")
            else:
                print(f"Raw data size: {len(data)} bytes")
        else:
            data = ncss.get_data(query)
            print(f"Query successful")

            if args.output:
                if args.format == 'netcdf' or not args.format:
                    if hasattr(data, 'to_netcdf'):
                        data.to_netcdf(args.output)
                        print(f"Data saved to: {args.output}")
                    else:
                        print("Cannot save: data is not xarray Dataset")
                else:
                    print(f"Cannot save {args.format} format to file")
            else:
                print(f"\nData type: {type(data)}")
                if hasattr(data, 'variables'):
                    print(f"Variables: {list(data.variables.keys())}")
                elif hasattr(data, 'columns'):
                    print(f"Columns: {list(data.columns)}")
                elif isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")

    except Exception as e:
        print(f"Error executing query: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Query NetCDF Subset Service (NCSS)'
    )
    parser.add_argument(
        'ncss_url',
        help='NCSS endpoint URL'
    )

    parser.add_argument(
        '--time',
        help='Specific time (YYYY-MM-DD HH:MM)'
    )
    parser.add_argument(
        '--time-range',
        nargs=2,
        metavar=('START', 'END'),
        help='Time range (YYYY-MM-DD HH:MM YYYY-MM-DD HH:MM)'
    )
    parser.add_argument(
        '--all-times',
        action='store_true',
        help='Request all times'
    )
    parser.add_argument(
        '--bbox',
        nargs=4,
        metavar=('WEST', 'EAST', 'SOUTH', 'NORTH'),
        help='Bounding box (west east south north)'
    )
    parser.add_argument(
        '--point',
        nargs=2,
        metavar=('LON', 'LAT'),
        help='Point location (lon lat)'
    )
    parser.add_argument(
        '--vars',
        nargs='+',
        help='Variables to request (use "all" for all variables)'
    )
    parser.add_argument(
        '--vertical-level',
        help='Vertical level (pressure or height)'
    )
    parser.add_argument(
        '--strides',
        nargs='+',
        help='Strides (spatial) or (time spatial)'
    )
    parser.add_argument(
        '--format',
        choices=['netcdf', 'xml', 'csv'],
        help='Output format'
    )
    parser.add_argument(
        '--add-lonlat',
        action='store_true',
        help='Add latitude/longitude to output'
    )
    parser.add_argument(
        '--output',
        help='Output file path'
    )
    parser.add_argument(
        '--raw',
        action='store_true',
        help='Get raw data instead of parsed'
    )
    parser.add_argument(
        '--list-vars',
        action='store_true',
        help='List available variables'
    )
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Show metadata'
    )

    args = parser.parse_args()

    execute_query(args.ncss_url, args)


if __name__ == '__main__':
    main()
