#!/usr/bin/env python3
"""
Basic PyVista example: Create and display a sphere
"""

import pyvista as pv

def main():
    # Create a sphere
    sphere = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
    
    # Create plotter
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.show()

if __name__ == '__main__':
    main()
