#!/usr/bin/env python3
"""
Mesh loading and visualization example
"""

import pyvista as pv

def main():
    # Load mesh from file
    mesh = pv.read('example.vtk')
    
    # Display mesh
    mesh.plot()

if __name__ == '__main__':
    main()
