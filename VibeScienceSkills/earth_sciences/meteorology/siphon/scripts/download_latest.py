#!/usr/bin/env python3
"""
Download latest dataset from THREDDS catalog.

This script downloads the latest dataset from a THREDDS catalog
using the specified access method.

Usage:
    python download_latest.py <catalog_url> <access_method> [output_file]

Examples:
    python download_latest.py http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.xml OPENDAP
    python download_latest.py http://thredds.ucar.edu/thredds/catalog/nexrad/nexrad.xml HTTPServer latest_radar.nc
"""

import sys
import argparse
from pathlib import Path


def download_latest(catalog_url, access_method, output_file=None):
    """
    Download latest dataset from THREDDS catalog.

    Args:
        catalog_url: URL of THREDDS catalog
        access_method: Access method (OPENDAP, HTTPServer, NetcdfSubset, etc.)
        output_file: Optional output file path

    Returns:
        Path to downloaded file
    """
    try:
        from siphon.catalog import TDSCatalog, get_latest_access_url
    except ImportError:
        print("Error: siphon package not installed")
        print("Install with: pip install siphon")
        sys.exit(1)

    print(f"Accessing catalog: {catalog_url}")

    try:
        cat = TDSCatalog(catalog_url)
    except Exception as e:
        print(f"Error accessing catalog: {e}")
        sys.exit(1)

    print(f"Found {len(cat.datasets)} datasets in catalog")

    try:
        latest_url = get_latest_access_url(catalog_url, access_method)
        print(f"Latest dataset URL: {latest_url}")
    except Exception as e:
        print(f"Error getting latest access URL: {e}")
        sys.exit(1)

    if access_method == 'HTTPServer':
        try:
            dataset = cat.datasets[list(cat.datasets.keys())[0]]
            if output_file:
                dataset.download(output_file)
                print(f"Downloaded to: {output_file}")
                return Path(output_file)
            else:
                dataset.download()
                print(f"Downloaded to current directory")
                return Path(dataset.name)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)
    else:
        print(f"Access method '{access_method}' does not support direct download")
        print(f"Latest URL: {latest_url}")
        print(f"Use this URL with appropriate client (e.g., netCDF4 for OPENDAP)")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Download latest dataset from THREDDS catalog'
    )
    parser.add_argument(
        'catalog_url',
        help='URL of THREDDS catalog'
    )
    parser.add_argument(
        'access_method',
        help='Access method (OPENDAP, HTTPServer, NetcdfSubset, etc.)'
    )
    parser.add_argument(
        'output_file',
        nargs='?',
        help='Optional output file path'
    )

    args = parser.parse_args()

    download_latest(args.catalog_url, args.access_method, args.output_file)


if __name__ == '__main__':
    main()
