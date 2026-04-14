#!/usr/bin/env python3
"""
Upper air data retrieval utility.

This script retrieves upper air sounding data from various sources
(Wyoming, Iowa State, IGRA2).

Usage:
    python get_upper_air.py <source> <station_id> <time> [options]

Examples:
    python get_upper_air.py wyoming DEN 2024-01-01_00:00
    python get_upper_air.py iastate OUN 2024-01-01_00:00 --interp
    python get_upper_air.py igra2 724060-99999 2024-01-01_00:00 --output sounding.csv
"""

import sys
import argparse
from datetime import datetime


def retrieve_wyoming(station_id, time_str, args):
    """
    Retrieve data from Wyoming upper air archive.

    Args:
        station_id: 3-letter ICAO station ID
        time_str: Time string (YYYY-MM-DD_HH:MM)
        args: Command line arguments

    Returns:
        pandas DataFrame with sounding data
    """
    try:
        from siphon.simplewebservice.wyoming import WyomingUpperAir
    except ImportError:
        print("Error: siphon package not installed")
        print("Install with: pip install siphon")
        sys.exit(1)

    try:
        time = datetime.strptime(time_str, '%Y-%m-%d_%H:%M')
    except ValueError:
        print(f"Invalid time format: {time_str}")
        print("Use format: YYYY-MM-DD_HH:MM")
        sys.exit(1)

    print(f"Retrieving data from Wyoming archive...")
    print(f"  Station: {station_id}")
    print(f"  Time: {time}")

    try:
        df = WyomingUpperAir.request_data(
            time,
            station_id,
            recalc=args.recalc
        )
        print(f"Retrieved {len(df)} levels")
        return df
    except Exception as e:
        print(f"Error retrieving data: {e}")
        sys.exit(1)


def retrieve_iastate(station_id, time_str, args):
    """
    Retrieve data from Iowa State upper air archive.

    Args:
        station_id: 3-letter ICAO station ID
        time_str: Time string (YYYY-MM-DD_HH:MM)
        args: Command line arguments

    Returns:
        pandas DataFrame with sounding data
    """
    try:
        from siphon.simplewebservice.iastate import IAStateUpperAir
    except ImportError:
        print("Error: siphon package not installed")
        print("Install with: pip install siphon")
        sys.exit(1)

    try:
        time = datetime.strptime(time_str, '%Y-%m-%d_%H:%M')
    except ValueError:
        print(f"Invalid time format: {time_str}")
        print("Use format: YYYY-MM-DD_HH:MM")
        sys.exit(1)

    print(f"Retrieving data from Iowa State archive...")
    print(f"  Station: {station_id}")
    print(f"  Time: {time}")

    try:
        df = IAStateUpperAir.request_data(
            time,
            station_id,
            interp_nans=args.interp
        )
        print(f"Retrieved {len(df)} levels")
        return df
    except Exception as e:
        print(f"Error retrieving data: {e}")
        sys.exit(1)


def retrieve_igra2(station_id, time_str, args):
    """
    Retrieve data from IGRA2 upper air archive.

    Args:
        station_id: 11-character IGRA2 station ID
        time_str: Time string (YYYY-MM-DD_HH:MM)
        args: Command line arguments

    Returns:
        pandas DataFrame with sounding data
    """
    try:
        from siphon.simplewebservice.igra2 import IGRAUpperAir
    except ImportError:
        print("Error: siphon package not installed")
        print("Install with: pip install siphon")
        sys.exit(1)

    try:
        time = datetime.strptime(time_str, '%Y-%m-%d_%H:%M')
    except ValueError:
        print(f"Invalid time format: {time_str}")
        print("Use format: YYYY-MM-DD_HH:MM")
        sys.exit(1)

    print(f"Retrieving data from IGRA2 archive...")
    print(f"  Station: {station_id}")
    print(f"  Time: {time}")

    try:
        df = IGRAUpperAir.request_data(
            station_id,
            time,
            derived=args.derived
        )
        print(f"Retrieved {len(df)} levels")
        return df
    except Exception as e:
        print(f"Error retrieving data: {e}")
        sys.exit(1)


def display_data(df, args):
    """
    Display retrieved sounding data.

    Args:
        df: pandas DataFrame with sounding data
        args: Command line arguments
    """
    if args.summary:
        print("\nData Summary:")
        print(f"  Levels: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        if hasattr(df, 'units'):
            print("\nUnits:")
            for col, unit in df.units.items():
                if unit:
                    print(f"  {col}: {unit}")

        print("\nFirst few levels:")
        print(df.head(args.lines))

        if args.show_units:
            print("\nFull data with units:")
            print(df)
    else:
        print("\nData:")
        print(df.head(args.lines))


def save_data(df, output_file):
    """
    Save sounding data to file.

    Args:
        df: pandas DataFrame with sounding data
        output_file: Output file path
    """
    try:
        df.to_csv(output_file, index=False)
        print(f"\nData saved to: {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Retrieve upper air sounding data'
    )
    parser.add_argument(
        'source',
        choices=['wyoming', 'iastate', 'igra2'],
        help='Data source (wyoming, iastate, igra2)'
    )
    parser.add_argument(
        'station_id',
        help='Station ID (ICAO for wyoming/iastate, IGRA2 ID for igra2)'
    )
    parser.add_argument(
        'time',
        help='Time (YYYY-MM-DD_HH:MM)'
    )
    parser.add_argument(
        '--output',
        help='Output file path'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show data summary'
    )
    parser.add_argument(
        '--show-units',
        action='store_true',
        help='Show units information'
    )
    parser.add_argument(
        '--lines',
        type=int,
        default=10,
        help='Number of lines to display (default: 10)'
    )

    parser.add_argument(
        '--recalc',
        action='store_true',
        help='Force server recalculation (Wyoming only)'
    )
    parser.add_argument(
        '--interp',
        action='store_true',
        help='Interpolate NaN values (Iowa State only)'
    )
    parser.add_argument(
        '--derived',
        action='store_true',
        help='Request derived data (IGRA2 only)'
    )

    args = parser.parse_args()

    if args.source == 'wyoming':
        df = retrieve_wyoming(args.station_id, args.time, args)
    elif args.source == 'iastate':
        df = retrieve_iastate(args.station_id, args.time, args)
    elif args.source == 'igra2':
        df = retrieve_igra2(args.station_id, args.time, args)

    display_data(df, args)

    if args.output:
        save_data(df, args.output)


if __name__ == '__main__':
    main()
