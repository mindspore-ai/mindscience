#!/usr/bin/env python3
"""
Scalar field visualization example
"""

import pyvista as pv
import numpy as np

def main():
    # Create sphere
    sphere = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
    
    # Add scalar field
    scalars = np.random.rand(sphere.n_points)
    sphere['scalars'] = scalars
    
    # Create plotter with scalar bar
    plotter = pv.Plotter()
    plotter.add_mesh(sphere, scalars='scalars', cmap='viridis')
    plotter.add_scalar_bar('title='Random Values')
    plotter.show()

if __name__ == '__main__':
    main()
