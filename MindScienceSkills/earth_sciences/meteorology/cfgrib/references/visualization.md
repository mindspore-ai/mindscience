# Visualization with cfgrib

This guide covers visualization techniques and plotting examples using cfgrib, xarray, and matplotlib.

## Basic Plotting

### Time Series Plots

Plot time series data:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select location
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot time series
    ts.plot(ax=ax, marker='o', linestyle='-', markersize=4, alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('2m Temperature at (40°N, 100°W)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timeseries.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Spatial Field Plots

Plot spatial fields:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select time
    temp = ds['t2m'].isel(time=0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot spatial field
    temp.plot(
        ax=ax,
        cmap='coolwarm',
        robust=True,
        cbar_kwargs={'label': 'Temperature (K)', 'orientation': 'horizontal'}
    )
    
    ax.set_title('2m Temperature - Initial Time')
    plt.tight_layout()
    plt.savefig('spatial.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Vertical Profile Plots

Plot vertical profiles:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select location and time
    profile = ds['t'].sel(
        latitude=40.0, 
        longitude=-100.0, 
        time='2024-01-15T12:00:00',
        method='nearest'
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Plot profile
    profile.plot(ax=ax, yincrease=False)
    
    # Add labels
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title('Temperature Profile at (40°N, 100°W)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('profile.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## Map Plots

### Simple Map Plot

Create a simple map plot:

```python
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select time
    temp = ds['t2m'].isel(time=0)
    
    # Create figure with map projection
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    # Add coastlines and borders
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot data
    temp.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        robust=True,
        cbar_kwargs={'label': 'Temperature (K)', 'shrink': 0.8}
    )
    
    ax.set_title('2m Temperature', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('map.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Regional Map Plot

Create a regional map plot:

```python
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select region and time
    temp = ds['t2m'].sel(
        latitude=slice(50, 30),
        longitude=slice(-120, -90)
    ).isel(time=0)
    
    # Calculate center for projection
    center_lat = (temp.latitude[0] + temp.latitude[-1]) / 2
    center_lon = (temp.longitude[0] + temp.longitude[-1]) / 2
    
    # Create figure with regional projection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=ccrs.LambertConformal(
        central_longitude=center_lon,
        central_latitude=center_lat
    ))
    
    # Add features
    ax.coastlines(resolution='10m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    
    # Set extent
    ax.set_extent([
        temp.longitude[0] - 2,
        temp.longitude[-1] + 2,
        temp.latitude[-1] - 2,
        temp.latitude[0] + 2
    ], crs=ccrs.PlateCarree())
    
    # Plot data
    temp.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        robust=True,
        cbar_kwargs={'label': 'Temperature (K)'}
    )
    
    ax.set_title('Regional 2m Temperature', fontsize=14)
    plt.tight_layout()
    plt.savefig('regional_map.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Polar Map Plot

Create a polar map plot:

```python
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select northern hemisphere
    temp = ds['t2m'].sel(latitude=slice(90, 0)).isel(time=0)
    
    # Create figure with polar projection
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=ccrs.NorthPolarStereoographic())
    
    # Add features
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Add circle at 30°N
    ax.plot(
        np.linspace(-180, 180, 100),
        [30] * 100,
        transform=ccrs.PlateCarree(),
        color='black',
        linewidth=0.5,
        linestyle='--'
    )
    
    # Plot data
    temp.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        robust=True,
        cbar_kwargs={'label': 'Temperature (K)'}
    )
    
    ax.set_title('Northern Hemisphere 2m Temperature', fontsize=14)
    plt.tight_layout()
    plt.savefig('polar_map.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## Ensemble Plots

### Ensemble Spaghetti Plot

Plot ensemble members:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Select location
    ensemble = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each member
    for member in ensemble.number.values:
        member_ts = ensemble.sel(number=member)
        member_ts.plot(ax=ax, alpha=0.3, color='blue', linewidth=1)
    
    # Plot ensemble mean
    ensemble_mean = ensemble.mean(dim='number')
    ensemble_mean.plot(ax=ax, color='red', linewidth=2.5, label='Ensemble Mean')
    
    # Plot ensemble spread
    ensemble_std = ensemble.std(dim='number')
    ax.fill_between(
        ensemble.time.values,
        (ensemble_mean - ensemble_std).values,
        (ensemble_mean + ensemble_std).values,
        alpha=0.2,
        color='red',
        label='±1σ'
    )
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Ensemble Spaghetti Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spaghetti.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Ensemble Spread Plot

Plot ensemble spread:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Calculate ensemble spread
    spread = ds['t2m'].std(dim='number')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot spread
    spread.isel(time=0).plot(
        ax=ax,
        cmap='YlOrRd',
        robust=True,
        cbar_kwargs={'label': 'Spread (K)'}
    )
    
    ax.set_title('Ensemble Spread - Initial Time')
    plt.tight_layout()
    plt.savefig('spread.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Probability Plots

Plot probability of exceeding threshold:

```python
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Calculate probability
    prob = (ds['t2m'] > 300).mean(dim='number')
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    # Add features
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Plot probability
    prob.isel(time=0).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='YlOrRd',
        levels=[i/10 for i in range(11)],
        extend='both',
        cbar_kwargs={'label': 'Probability', 'shrink': 0.8}
    )
    
    ax.set_title('Probability of T > 300 K', fontsize=14)
    plt.tight_layout()
    plt.savefig('probability.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## Statistical Plots

### Time Series with Statistics

Plot time series with statistical bounds:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select location
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Calculate statistics
    mean = ts.mean()
    std = ts.std()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot time series
    ts.plot(ax=ax, alpha=0.7, label='Temperature')
    
    # Plot mean
    ax.axhline(mean, color='red', linestyle='--', linewidth=2, label='Mean')
    
    # Plot ±1σ bounds
    ax.axhline(mean + std, color='orange', linestyle=':', linewidth=1.5, label='Mean ± 1σ')
    ax.axhline(mean - std, color='orange', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('2m Temperature with Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timeseries_stats.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Zonal Mean Plot

Plot zonal mean:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Calculate zonal mean
    zonal_mean = ds['t2m'].mean(dim='longitude')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot zonal mean for each time
    for i, time in enumerate(zonal_mean.time.values[:5]):  # First 5 times
        zonal_mean.isel(time=i).plot(ax=ax, label=str(time))
    
    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Zonal Mean Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('zonal_mean.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Hovmöller Diagram

Create Hovmöller diagram:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select location
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Create daily means
    daily = ts.resample(time='1D').mean()
    
    # Create Hovmöller
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot
    c = ax.plot(
        daily.time.dt.dayofyear,
        daily.time.dt.year,
        daily.values,
        marker='o',
        linestyle='none',
        alpha=0.7
    )
    
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Year')
    ax.set_title('Hovmöller Diagram')
    plt.colorbar(c, label='Temperature (K)')
    
    plt.tight_layout()
    plt.savefig('hovmoller.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## Custom Plots

### Multiple Subplots

Create multiple subplots:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select data
    temp = ds['t2m'].isel(time=0)
    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2).isel(time=0)
    pressure = ds['msl'].isel(time=0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot temperature
    temp.plot(ax=axes[0, 0], cmap='coolwarm', robust=True)
    axes[0, 0].set_title('2m Temperature')
    
    # Plot wind speed
    wind_speed.plot(ax=axes[0, 1], cmap='viridis', robust=True)
    axes[0, 1].set_title('10m Wind Speed')
    
    # Plot pressure
    pressure.plot(ax=axes[1, 0], cmap='RdYlBu_r', robust=True)
    axes[1, 0].set_title('Mean Sea Level Pressure')
    
    # Plot time series
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    ts.plot(ax=axes[1, 1])
    axes[1, 1].set_title('Time Series at (40°N, 100°W)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('subplots.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Custom Colormap

Use custom colormap:

```python
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormap
colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select data
    temp = ds['t2m'].isel(time=0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot with custom colormap
    temp.plot(ax=ax, cmap=custom_cmap, robust=True)
    
    ax.set_title('2m Temperature - Custom Colormap')
    plt.tight_layout()
    plt.savefig('custom_cmap.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## Best Practices

### 1. Use Appropriate Figure Sizes

```python
# For time series
fig, ax = plt.subplots(figsize=(14, 6))

# For spatial plots
fig, ax = plt.subplots(figsize=(12, 8))

# For maps
fig, ax = plt.subplots(figsize=(14, 10))
```

### 2. Add Labels and Titles

```python
# Always add labels
ax.set_xlabel('Time')
ax.set_ylabel('Temperature (K)')
ax.set_title('Descriptive Title')

# Add grid
ax.grid(True, alpha=0.3)
```

### 3. Use Robust Color Scaling

```python
# Use robust to handle outliers
data.plot(cmap='coolwarm', robust=True)

# Or specify vmin/vmax
data.plot(cmap='coolwarm', vmin=250, vmax=310)
```

### 4. Save High-Quality Figures

```python
# Save with high DPI
plt.savefig('figure.png', dpi=300, bbox_inches='tight')

# For publications
plt.savefig('figure.pdf', bbox_inches='tight')
```

### 5. Use Context Managers

```python
# Always close datasets
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Create plots
    ds['t2m'].plot()
```

## References

- xarray Plotting: https://xarray.pydata.org/stable/plotting.html
- matplotlib Documentation: https://matplotlib.org/stable/contents.html
- cartopy Documentation: https://scitools.org.uk/cartopy/docs/latest/