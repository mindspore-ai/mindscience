#!/usr/bin/env python3
"""
Field visualization utilities for Meep simulations.

This script provides functions for visualizing electromagnetic fields
from Meep simulations using NumPy and Matplotlib.
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def plot_field_2d(sim, component=mp.Ez, center=None, size=None, 
                    cmap='RdBu', show_geometry=True, alpha=0.9):
    """
    Plot a 2D field component with optional geometry overlay.
    
    Args:
        sim: Meep simulation object
        component: Field component to plot (e.g., mp.Ez, mp.Ex)
        center: Center of plotting region (default: cell center)
        size: Size of plotting region (default: cell size)
        cmap: Colormap for field values
        show_geometry: Whether to overlay dielectric function
        alpha: Transparency of field overlay
    
    Returns:
        matplotlib figure object
    """
    if center is None:
        center = mp.Vector3()
    if size is None:
        size = sim.cell_size
    
    # Get field data
    field_data = sim.get_array(center=center, size=size, component=component)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot geometry if requested
    if show_geometry:
        eps_data = sim.get_array(center=center, size=size, component=mp.Dielectric)
        ax.imshow(eps_data.transpose(), cmap='binary', interpolation='spline36')
        ax.imshow(field_data.transpose(), cmap=cmap, alpha=alpha, 
                 interpolation='spline36')
    else:
        ax.imshow(field_data.transpose(), cmap=cmap, interpolation='spline36')
    
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(ax.images[0] if show_geometry else ax.images[0], ax=ax)
    cbar.set_label(f'Field {mp.component_name(component)}')
    
    return fig


def plot_multiple_fields(sim, components, center=None, size=None, 
                       cmap='RdBu', show_geometry=True, alpha=0.9):
    """
    Plot multiple field components in subplots.
    
    Args:
        sim: Meep simulation object
        components: List of field components to plot
        center: Center of plotting region
        size: Size of plotting region
        cmap: Colormap for field values
        show_geometry: Whether to overlay dielectric function
        alpha: Transparency of field overlay
    
    Returns:
        matplotlib figure object
    """
    if center is None:
        center = mp.Vector3()
    if size is None:
        size = sim.cell_size
    
    n_components = len(components)
    n_cols = min(3, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_components == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, component in enumerate(components):
        if i >= len(axes):
            break
        
        # Get field data
        field_data = sim.get_array(center=center, size=size, component=component)
        
        # Plot geometry if requested
        if show_geometry:
            eps_data = sim.get_array(center=center, size=size, component=mp.Dielectric)
            axes[i].imshow(eps_data.transpose(), cmap='binary', interpolation='spline36')
            axes[i].imshow(field_data.transpose(), cmap=cmap, alpha=alpha, 
                        interpolation='spline36')
        else:
            axes[i].imshow(field_data.transpose(), cmap=cmap, interpolation='spline36')
        
        axes[i].set_title(f'{mp.component_name(component)}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_components, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_field_slice(sim, component=mp.Ez, slice_axis=mp.X, 
                     slice_position=0.0, cmap='RdBa'):
    """
    Plot a field slice along a specific axis.
    
    Args:
        sim: Meep simulation object
        component: Field component to plot
        slice_axis: Axis along which to slice (mp.X, mp.Y, or mp.Z)
        slice_position: Position along slice axis
        cmap: Colormap for field values
    
    Returns:
        matplotlib figure object
    """
    # Determine slice size
    cell_size = sim.cell_size
    if slice_axis == mp.X:
        center = mp.Vector3(slice_position, 0, 0)
        size = mp.Vector3(0, cell_size.y, cell_size.z)
        xlabel = 'y'
        ylabel = 'z'
    elif slice_axis == mp.Y:
        center = mp.Vector3(0, slice_position, 0)
        size = mp.Vector3(cell_size.x, 0, cell_size.z)
        xlabel = 'x'
        ylabel = 'z'
    else:  # mp.Z
        center = mp.Vector3(0, 0, slice_position)
        size = mp.Vector3(cell_size.x, cell_size.y, 0)
        xlabel = 'x'
        ylabel = 'y'
    
    # Get field data
    field_data = sim.get_array(center=center, size=size, component=component)
    
    field_data = sim.get_array(center=center, size=size, component=component)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(field_data.transpose(), cmap=cmap, interpolation='spline36')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{mp.component_name(component)} at {mp.direction_name(slice_axis)}={slice_position}')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'Field {mp.component_name(component)}')
    
    return fig


def plot_energy_density(sim, center=None, size=None, cmap='hot'):
    """
    Plot electromagnetic energy density.
    
    Args:
        sim: Meep simulation object
        center: Center of plotting region
        size: Size of plotting region
        cmap: Colormap for energy density
    
    Returns:
        matplotlib figure object
    """
    if center is None:
        center = mp.Vector3()
    if size is None:
        size = sim.cell_size
    
    # Get electric and magnetic fields
    Ex = sim.get_array(center=center, size=size, component=mp.Ex)
    Ey = sim.get_array(center=center, size=size, component=mp.Ey)
    Ez = sim.get_array(center=center, size=size, component=mp.Ez)
    Hx = sim.get_array(center=center, size=size, component=mp.Hx)
    Hy = sim.get_array(center=center, size=size, component=mp.Hy)
    Hz = sim.get_array(center=center, size=size, component=mp.Hz)
    
    # Calculate energy density
    electric_energy = 0.5 * (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    magnetic_energy = 0.5 * (np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2)
    total_energy = electric_energy + magnetic_energy
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(electric_energy.transpose(), cmap=cmap, interpolation='spline36')
    axes[0].set_title('Electric Energy Density')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(magnetic_energy.transpose(), cmap=cmap, interpolation='spline36')
    axes[1].set_title('Magnetic Energy Density')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(total_energy.transpose(), cmap=cmap, interpolation='spline36')
    axes[2].set_title('Total Energy Density')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    return fig


def plot_poynting_vector(sim, center=None, size=None, cmap='seismic'):
    """
    Plot Poynting vector field.
    
    Args:
        sim: Meep simulation object
        center: Center of plotting region
        size: Size of plotting region
        cmap: Colormap for Poynting vector magnitude
    
    Returns:
        matplotlib figure object
    """
    if center is None:
        center = mp.Vector3()
    if size is None:
        size = sim.cell_size
    
    # Get field components
    Ex = sim.get_array(center=center, size=size, component=mp.Ex)
    Ey = sim.get_array(center=center, size=size, component=mp.Ey)
    Ez = sim.get_array(center=center, size=size, component=mp.Ez)
    Hx = sim.get_array(center=center, size=size, component=mp.Hx)
    Hy = sim.get_array(center=center, size=size, component=mp.Hy)
    Hz = sim.get_array(center=center, size=size, component=mp.Hz)
    
    # Calculate Poynting vector
    Sx = Ey * Hz - Ez * Hy
    Sy = Ez * Hx - Ex * Hz
    Sz = Ex * Hy - Ey * Hx
    
    # Calculate magnitude
    S_magnitude = np.sqrt(np.abs(Sx)**2 + np.abs(Sy)**2 + np.abs(Sz)**2)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    im1 = axes[0, 0].imshow(Sx.transpose(), cmap='RdBu', interpolation='spline36')
    axes[0, 0].set_title('Sx')
    axes[0, 0].axis('off')
    
    im2 = axes[0, 1].imshow(Sy.transpose(), cmap='RdBu', interpolation='spline36')
    axes[0, 1].set_title('Sy')
    axes[0, 1].axis('off')
    
    im3 = axes[1, 0].imshow(Sz.transpose(), cmap='RdBu', interpolation='spline36')
    axes[1, 0].set_title('Sz')
    axes[1, 0].axis('off')
    
    im4 = axes[1, 1].imshow(S_magnitude.transpose(), cmap=cmap, interpolation='spline36')
    axes[1, 1].set_title('|S|')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    return fig


def create_custom_colormap(colors, name='custom'):
    """
    Create a custom colormap from a list of colors.
    
    Args:
        colors: List of (value, color) tuples
        name: Name for the colormap
    
    Returns:
        matplotlib colormap object
    """
    cmap = LinearSegmentedColormap.from_list(name, colors)
    return cmap


def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """
    Save figure to file.
    
    Args:
        fig: matplotlib figure object
        filename: Output filename
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box setting
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save figure
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure to {filename}")


def main():
    """Example usage of visualization functions."""
    print("Meep Field Visualization Utilities")
    print("====================================")
    print()
    print("Available functions:")
    print("- plot_field_2d: Plot 2D field with geometry overlay")
    print("- plot_multiple_fields: Plot multiple field components")
    print("- plot_field_slice: Plot field slice along axis")
    print("- plot_energy_density: Plot electromagnetic energy density")
    print("- plot_poynting_vector: Plot Poynting vector field")
    print("- create_custom_colormap: Create custom colormap")
    print("- save_figure: Save figure to file")
    print()
    print("Example usage:")
    print("  import meep as mp")
    print("  from visualize_fields import plot_field_2d")
    print("  ")
    print("  sim = mp.Simulation(...)")
    print("  sim.run(until=200)")
    print("  ")
    print("  fig = plot_field_2d(sim, component=mp.Ez)")
    print("  plt.show()")


if __name__ == "__main__":
    main()