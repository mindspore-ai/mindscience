# Time Series Analysis with cfgrib

This guide covers time series analysis methods and patterns using cfgrib and xarray.

## Extracting Time Series

### Point Time Series

Extract time series for a specific location:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select nearest grid point
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Convert to pandas Series
    ts_series = ts.to_series()
    
    # Plot time series
    ts.plot()
```

### Regional Time Series

Extract time series for a region:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select region
    region = ds['t2m'].sel(
        latitude=slice(50, 30),
        longitude=slice(-120, -90)
    )
    
    # Regional average time series
    regional_ts = region.mean(dim=['latitude', 'longitude'])
    
    # Plot
    regional_ts.plot()
```

### Vertical Profile Time Series

Extract time series for vertical profiles:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select location
    profile = ds['t'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Time series for each level
    for level in ds.level.values:
        level_ts = profile.sel(level=level)
        level_ts.plot(label=f'{level} hPa')
    
    plt.legend()
```

## Time Series Statistics

### Basic Statistics

Calculate time series statistics:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Basic statistics
    mean = ts.mean()
    std = ts.std()
    min_val = ts.min()
    max_val = ts.max()
    median = ts.median()
    
    print(f"Mean: {mean:.2f}")
    print(f"Std: {std:.2f}")
    print(f"Range: {min_val:.2f} - {max_val:.2f}")
```

### Trend Analysis

Calculate linear trend:

```python
import xarray as xr
import numpy as np

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Convert time to numeric
    time_numeric = (ts.time - ts.time[0]).dt.days
    
    # Linear regression
    coeffs = np.polyfit(time_numeric, ts.values, 1)
    trend = coeffs[0]  # Slope (change per day)
    intercept = coeffs[1]
    
    # Calculate trend line
    trend_line = coeffs[0] * time_numeric + coeffs[1]
    
    print(f"Trend: {trend:.6f} K/day")
```

### Seasonal Decomposition

Decompose time series into components:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Calculate climatology
    climatology = ts.groupby('time.dayofyear').mean()
    
    # Calculate anomalies
    anomalies = ts.groupby('time.dayofyear') - climatology
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    ts.plot(ax=axes[0], title='Original')
    climatology.plot(ax=axes[1], title='Climatology')
    anomalies.plot(ax=axes[2], title='Anomalies')
    
    plt.tight_layout()
```

## Periodic Analysis

### Diurnal Cycle

Analyze daily cycle:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Group by hour
    hourly = ts.groupby('time.hour').mean()
    
    # Plot diurnal cycle
    hourly.plot(marker='o')
    plt.xlabel('Hour')
    plt.ylabel('Temperature (K)')
    plt.title('Diurnal Cycle')
```

### Annual Cycle

Analyze annual cycle:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Group by day of year
    daily = ts.groupby('time.dayofyear').mean()
    
    # Plot annual cycle
    daily.plot()
    plt.xlabel('Day of Year')
    plt.ylabel('Temperature (K)')
    plt.title('Annual Cycle')
```

### Seasonal Analysis

Analyze seasonal patterns:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Group by season
    seasonal = ts.groupby('time.season').mean()
    
    # Plot seasonal
    seasonal.plot(kind='bar')
    plt.xlabel('Season')
    plt.ylabel('Temperature (K)')
    plt.title('Seasonal Average')
```

## Frequency Analysis

### Fourier Analysis

Perform spectral analysis:

```python
import xarray as xr
from scipy.fft import fft, fftfreq

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Compute FFT
    fft_values = fft(ts.values)
    
    # Compute frequencies
    # Assuming 6-hourly data
    freqs = fftfreq(len(ts), d=6)  # cycles per hour
    
    # Compute power spectrum
    power = np.abs(fft_values)**2
    
    # Plot power spectrum
    plt.plot(freqs, power)
    plt.xlabel('Frequency (cycles/hour)')
    plt.ylabel('Power')
    plt.title('Power Spectrum')
```

### Wavelet Analysis

Perform wavelet analysis (requires pywavelets):

```python
import xarray as xr
import pywt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Continuous wavelet transform
    coeffs, scales, freqs, cone, coi = pywt.cwt(
        ts.values, 
        pywt.ContinuousWavelet('morl'),
        scales=np.arange(1, 100)
    )
    
    # Plot scalogram
    plt.imshow(np.abs(coeffs), aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('Wavelet Scalogram')
```

## Time Series Filtering

### Moving Average

Apply moving average filter:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # 7-point moving average
    ma7 = ts.rolling(time=7, center=True).mean()
    
    # 30-point moving average
    ma30 = ts.rolling(time=30, center=True).mean()
    
    # Plot
    ts.plot(label='Original')
    ma7.plot(label='7-point MA')
    ma30.plot(label='30-point MA')
    plt.legend()
```

### Low-Pass Filter

Apply` low-pass filter:

```python
import xarray as xr
from scipy.signal import butter, filtfilt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Design low-pass filter
    # Assuming 6-hourly data, cutoff at 10-day period
    nyquist = 1 / (6 * 2)  # Nyquist frequency
    cutoff = 1 / (10 * 24)  # 10-day period
    b, a = butter(2, cutoff / nyquist, btype='low')
    
    # Apply filter
    filtered = filtfilt(b, a, ts.values)
    
    # Plot
    plt.plot(ts.time, ts.values, label='Original')
    plt.plot(ts.time, filtered, label='Filtered')
    plt.legend()
```

### High-Pass Filter

Apply high-pass filter:

```python
import xarray as xr
from scipy.signal import butter, filtfilt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Design high-pass filter
    nyquist = 1 / (6 * 2)
    cutoff = 1 / (30 * 24)  # 30-day period
    b, a = butter(2, cutoff / nyquist, btype='high')
    
    # Apply filter
    filtered = filtfilt(b, a, ts.values)
    
    # Plot
    plt.plot(ts.time, ts.values, label='Original')
    plt.plot(ts.time, filtered, label='Filtered')
    plt.legend()
```

## Time Series Correlation

### Autocorrelation

Calculate autocorrelation function:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Normalize
    ts_normalized = (ts - ts.mean()) / ts.std()
    
    # Calculate autocorrelation for different lags
    max_lag = 40
    autocorr = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr.append(1.0)
        else:
            correlation = np.corrcoef(
                ts_normalized[:-lag].values,
                ts_normalized[lag:].values
            )[0, 1]
            autocorr.append(correlation)
    
    # Plot autocorrelation
    plt.plot(range(max_lag + 1), autocorr, marker='o')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function')
```

### Cross-Correlation

Calculate cross-correlation between time series:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts1 = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    ts2 = ds['msl'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Normalize
    ts1_normalized = (ts1 - ts1.mean()) / ts1.std()
    ts2_normalized = (ts2 - ts2.mean()) / ts2.std()
    
    # Calculate cross-correlation
    max_lag = 40
    crosscorr = []
    
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            correlation = np.corrcoef(
                ts1_normalized[:-lag].values,
                ts2_normalized[lag:].values
            )[0, 1]
        else:
            correlation = np.corrcoef(
                ts1_normalized[-lag:].values,
                ts2_normalized[:lag].values
            )[0, 1]
        crosscorr.append(correlation)
    
    # Plot cross-correlation
    plt.plot(range(-max_lag, max_lag + 1), crosscorr, marker='o')
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.title('Cross-correlation Function')
```

## Time Series Forecasting

### Persistence Forecast

Simple persistence forecast:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Split into training and testing
    n_train = int(len(ts) * 0.8)
    train = ts[:n_train]
    test = ts[n_train:]
    
    # Persistence forecast (last value)
    last_value = train[-1].values
    persistence_forecast = np.full(len(test), last_value)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((test.values - persistence_forecast)**2))
    
    print(f"Persistence RMSE: {rmse:.2f} K")
```

### Climatology Forecast

Climatology-based forecast:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Calculate climatology
    climatology = ts.groupby('time.dayofyear').mean()
    
    # Forecast using climatology
    forecast = []
    for time in test.time:
        doy = time.dt.dayofyear
        forecast.append(climatology.sel(dayofyear=doy).values)
    
    forecast = np.array(forecast)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((test.values - forecast)**2))
    
    print(f"Climatology RMSE: {rmse:.2f} K")
```

## Best Practices

### 1. Handle Missing Data

```python
# Handle missing values
ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
ts_clean = ts.dropna('time')

# Or interpolate
ts_filled = ts.interpolate_na('time', method='linear')
```

### 2. Normalize Data

```python
# Normalize before analysis
ts_normalized = (ts - ts.mean()) / ts.std()
```

### 3. Check Stationarity

```python
# Check for stationarity (simple test)
mean_first_half = ts[:len(ts)//2].mean()
mean_second_half = ts[len(ts)//2:].mean()

if abs(mean_first_half - mean_second_half) > ts.std():
    print("Warning: Time series may not be stationary")
```

### 4. Use Appropriate Time Scales

```python
# For high-frequency data
hourly = ts.resample(time='1H').mean()

# For daily data
daily = ts.resample(time='1D').mean()

# For monthly data
monthly = ts.resample(time='1M').mean()
```

## References

- xarray Documentation: https://xarray.pydata.org/
- scipy.signal: https://docs.scipy.org/doc/scipy/reference/signal.html
- scipy.fft: https://docs.scipy.org/doc/scipy/reference/fft.html