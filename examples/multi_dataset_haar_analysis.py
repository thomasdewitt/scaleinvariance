#!/usr/bin/env python3
"""
Multi-Dataset Haar Fluctuation Analysis

This example demonstrates Haar fluctuation analysis across multiple temperature datasets:
1. CET (Central England Temperature): Regional monthly data (1659-2025) 
2. Berkeley Earth: Global land+ocean temperature anomalies (1850+)
3. LGMR: Paleoclimate global temperature reconstruction (last 24,000 years)

Each dataset receives appropriate preprocessing before Haar analysis to estimate
the Hurst exponent and compare scaling behavior across different temporal scales.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from pathlib import Path
from scaleinvariance import haar_fluctuation_hurst

def load_cet_data(filepath):
    """Load CET monthly temperature data.
    
    Args:
        filepath: Path to meantemp_monthly_totals.txt file
        
    Returns:
        DataFrame with Year and monthly temperature columns
    """
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Skip header lines and process data
    for line in lines[5:]:  # Data starts at line 6
        line = line.strip()
        if line and not line.startswith('Year'):
            parts = line.split()
            if len(parts) >= 13:  # Year + 12 months
                try:
                    year = int(parts[0])
                    temps = []
                    for i in range(1, 13):
                        try:
                            temp = float(parts[i])
                            temps.append(np.nan if temp == -99.9 else temp)
                        except (ValueError, IndexError):
                            temps.append(np.nan)
                    data.append([year] + temps)
                except ValueError:
                    continue
    
    columns = ['Year'] + [f'Month_{i}' for i in range(1, 13)]
    return pd.DataFrame(data, columns=columns)

def reshape_to_timeseries(df):
    """Convert CET monthly data to continuous timeseries.
    
    Args:
        df: DataFrame with Year and monthly temperature columns
        
    Returns:
        tuple: (pandas Series with monthly data, start_year)
    """
    months_since_start = []
    temps = []
    start_year = int(df['Year'].min())
    
    for _, row in df.iterrows():
        year = int(row['Year'])
        for month in range(1, 13):
            month_index = (year - start_year) * 12 + (month - 1)
            temp = row[f'Month_{month}']
            if not np.isnan(temp):
                months_since_start.append(month_index)
                temps.append(temp)
    
    return pd.Series(temps, index=months_since_start), start_year

def seasonal_detrend(timeseries):
    """Remove seasonal cycle from CET data.
    
    Args:
        timeseries: pandas Series with numeric index representing months
        
    Returns:
        tuple: (detrended anomalies, monthly climatology dict)
    """
    months = (timeseries.index % 12) + 1
    
    # Calculate monthly climatology
    monthly_means = {}
    for month in range(1, 13):
        month_data = timeseries[months == month]
        monthly_means[month] = month_data.mean() if len(month_data) > 0 else 0.0
    
    # Create anomalies by subtracting monthly means
    anomalies = timeseries.copy()
    for i, month in enumerate(months):
        anomalies.iloc[i] = timeseries.iloc[i] - monthly_means[month]
    
    return anomalies, monthly_means

def load_berkeley_data(filepath):
    """Load Berkeley Earth Land+Ocean temperature anomalies.
    
    Args:
        filepath: Path to Land_and_Ocean_complete.txt file
        
    Returns:
        DataFrame with Year, Month, Anomaly columns
    """
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find data start after header
    for i, line in enumerate(lines):
        if 'Year, Month,  Anomaly' in line:
            data_start = i + 2
            break
    
    # Parse data lines
    for line in lines[data_start:]:
        line = line.strip()
        if line and not line.startswith('%'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    year, month, anomaly = int(parts[0]), int(parts[1]), float(parts[2])
                    if not np.isnan(anomaly):
                        data.append([year, month, anomaly])
                except (ValueError, IndexError):
                    continue
    
    return pd.DataFrame(data, columns=['Year', 'Month', 'Anomaly'])

def load_lgmr_data(filepath):
    """Load LGMR paleoclimate data from NetCDF file.
    
    Args:
        filepath: Path to LGMR_GMST_climo.nc file
        
    Returns:
        DataFrame with Age_BP and Temperature columns (sorted oldest first)
    """
    ds = xr.open_dataset(filepath)
    df = pd.DataFrame({
        'Age_BP': ds.age.values,  # Years before present
        'Temperature': ds.gmst.values  # Global mean surface temperature
    })
    return df.sort_values('Age_BP', ascending=False).reset_index(drop=True)

def berkeley_get_anomalies(df):
    """Extract Berkeley Earth anomalies (already detrended relative to 1951-1980)."""
    return df['Anomaly'].values

def lgmr_get_temperatures(df):
    """Extract LGMR temperatures.
        
    Args:
        df: DataFrame with Temperature column
        
    Returns:
        numpy array of raw temperatures
    """
    return df['Temperature'].values

def main():
    """Load multiple temperature datasets and perform Haar fluctuation analysis."""
    datasets = {}
    base_path = Path(__file__).parent
    
    # Process CET data: seasonal detrending required
    print("Processing CET data...")
    cet_df = load_cet_data(base_path / 'meantemp_monthly_totals.txt')
    cet_timeseries, cet_start_year = reshape_to_timeseries(cet_df)
    cet_anomalies, _ = seasonal_detrend(cet_timeseries)
    cet_H, cet_H_err = haar_fluctuation_hurst(cet_anomalies.values, min_sep=1, )
    datasets['CET'] = {
        'anomalies': cet_anomalies.values,
        'H': cet_H, 'H_err': cet_H_err,
        'years': cet_start_year + cet_timeseries.index / 12.0,
        'time_unit': 'months'
    }
    
    # Process Berkeley Earth data
    print("Processing Berkeley Earth data...")
    berkeley_df = load_berkeley_data(base_path / 'Land_and_Ocean_complete.txt')
    berkeley_anomalies = berkeley_get_anomalies(berkeley_df)
    berkeley_H, berkeley_H_err = haar_fluctuation_hurst(berkeley_anomalies, min_sep=1, max_sep=12*10)
    berkeley_years = berkeley_df['Year'] + (berkeley_df['Month'] - 1) / 12.0
    datasets['Berkeley'] = {
        'anomalies': berkeley_anomalies,
        'H': berkeley_H, 'H_err': berkeley_H_err,
        'years': berkeley_years.values,
        'time_unit': 'months'
    }
    
    # Process LGMR paleoclimate data
    print("Processing LGMR paleoclimate data...")
    lgmr_df = load_lgmr_data(base_path / 'LGMR_GMST_climo.nc')
    lgmr_temps = lgmr_get_temperatures(lgmr_df)
    lgmr_H, lgmr_H_err = haar_fluctuation_hurst(lgmr_temps, min_sep=1, )
    datasets['LGMR'] = {
        'anomalies': lgmr_temps,
        'H': lgmr_H, 'H_err': lgmr_H_err,
        'years': lgmr_df['Age_BP'].values,
        'time_unit': 'years'
    }
    
    # Print results
    print(f"\nHurst Exponent Results:")
    print(f"CET (1659-2025):     H = {cet_H:.3f} ± {cet_H_err:.3f}")
    print(f"Berkeley (1850+):    H = {berkeley_H:.3f} ± {berkeley_H_err:.3f}")
    print(f"LGMR (24k years):    H = {lgmr_H:.3f} ± {lgmr_H_err:.3f}")
    
    # Create visualization
    create_comparison_plot(datasets)
    return datasets

def create_comparison_plot(datasets):
    """Create comparison plots for temperature data and Haar fluctuation analysis."""
    from scaleinvariance.analysis.haar_fluctuation import haar_fluctuation_analysis
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    
    # Top panel: Modern temperature anomalies
    for i, (name, data) in enumerate(datasets.items()):
        if name != 'LGMR':
            axes[0].plot(data['years'], data['anomalies'], color=colors[i], 
                        alpha=0.7, linewidth=0.6, label=f"{name} (H={data['H']:.3f})")
    
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Temperature Anomaly (°C)')
    axes[0].set_title('Modern Temperature Records')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Middle panel: LGMR paleoclimate timeseries
    lgmr_data = datasets['LGMR']
    axes[1].plot(lgmr_data['years'], lgmr_data['anomalies'], color=colors[2], 
                alpha=0.8, linewidth=0.8, label=f"LGMR (H={lgmr_data['H']:.3f})")
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('LGMR Paleoclimate Record (24,000 years)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlabel('Years Before Present')
    
    # Bottom panel: Haar fluctuation analysis for all datasets
    for i, (name, data) in enumerate(datasets.items()):
        lags, fluctuations = haar_fluctuation_analysis(data['anomalies'])
        
        # Convert lags to years
        if data['time_unit'] == 'months':
            lag_years = lags / 12.0
        else:
            lag_years = lags * 200
        
        axes[2].loglog(lag_years, fluctuations, color=colors[i], marker=markers[i],
                      markersize=3, linewidth=1.2, alpha=0.8,
                      label=f'{name} (H={data["H"]:.3f}±{data["H_err"]:.3f})')
    
    axes[2].set_xlabel('Lag (years)')
    axes[2].set_ylabel('Haar Fluctuation')
    axes[2].set_title('Multi-Dataset Haar Fluctuation Analysis')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    datasets = main()