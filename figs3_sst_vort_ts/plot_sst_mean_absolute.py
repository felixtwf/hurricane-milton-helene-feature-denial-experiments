import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os

#%%----------------- read CNAPS2    
filepath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/both'
filepath_neither = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/neither'
filepath_nohelene = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nohelene'
filepath_nomilton = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nomilton'

# Output directory for results
outdir = './sst_mean_analysis'
os.makedirs(outdir, exist_ok=True)

# Load reference dataset to get coordinates
ds_ref = xr.open_dataset(f'{filepath}/sst.nc')
lon_rho = ds_ref.lon_rho.values
lat_rho = ds_ref.lat_rho.values
ocean_time = ds_ref.ocean_time.values

# Load SST data for control and all three cases
print("Loading SST data...")
sst_control = np.load(f'{filepath}/sst.npz')['sst']  # Control case (both hurricanes)
sst_neither = np.load(f'{filepath_neither}/sst.npz')['sst']  # No-hurricane case
sst_nohelene = np.load(f'{filepath_nohelene}/sst.npz')['sst']  # No-Helene case
sst_nomilton = np.load(f'{filepath_nomilton}/sst.npz')['sst']  # No-Milton case

print(f"SST data shape: {sst_control.shape}")
print(f"Number of timesteps: {len(ocean_time)}")

# Define two areas
# Area 1: LC (Loop Current) - small area
lc_lon_min, lc_lon_max = -88, -85
lc_lat_min, lc_lat_max = 24, 26.5

# Area 2: Eastern Shelf - larger area
es_lon_min, es_lon_max = -88, -82
es_lat_min, es_lat_max = 24, 30

# Create masks for the two areas
print("Creating spatial masks...")
# LC mask
lc_mask = ((lon_rho >= lc_lon_min) & (lon_rho <= lc_lon_max) & 
           (lat_rho >= lc_lat_min) & (lat_rho <= lc_lat_max))

# Eastern Shelf mask
es_mask = ((lon_rho >= es_lon_min) & (lon_rho <= es_lon_max) & 
           (lat_rho >= es_lat_min) & (lat_rho <= es_lat_max))

print(f"LC area: {np.sum(lc_mask)} grid points")
print(f"Eastern Shelf area: {np.sum(es_mask)} grid points")

# Calculate mean SST (absolute values) for each area and each case
print("Calculating mean SST (absolute values)...")
ntime = len(ocean_time)

# Initialize arrays to store mean SST (absolute values)
lc_mean_control = np.zeros(ntime)    # Control case (both hurricanes)
lc_mean_neither = np.zeros(ntime)    # No-hurricane case
lc_mean_nohelene = np.zeros(ntime)  # No-Helene case
lc_mean_nomilton = np.zeros(ntime)  # No-Milton case

es_mean_control = np.zeros(ntime)    # Control case (both hurricanes)
es_mean_neither = np.zeros(ntime)   # No-hurricane case
es_mean_nohelene = np.zeros(ntime)  # No-Helene case
es_mean_nomilton = np.zeros(ntime)  # No-Milton case

for i in range(ntime):
    # LC area means (absolute values)
    lc_mean_control[i] = np.nanmean(sst_control[i][lc_mask])
    lc_mean_neither[i] = np.nanmean(sst_neither[i][lc_mask])
    lc_mean_nohelene[i] = np.nanmean(sst_nohelene[i][lc_mask])
    lc_mean_nomilton[i] = np.nanmean(sst_nomilton[i][lc_mask])
    
    # Eastern Shelf area means (absolute values)
    es_mean_control[i] = np.nanmean(sst_control[i][es_mask])
    es_mean_neither[i] = np.nanmean(sst_neither[i][es_mask])
    es_mean_nohelene[i] = np.nanmean(sst_nohelene[i][es_mask])
    es_mean_nomilton[i] = np.nanmean(sst_nomilton[i][es_mask])

# Convert time to pandas datetime
time_pd = pd.to_datetime(ocean_time)

# Create DataFrame for output
results_df = pd.DataFrame({
    'Time': time_pd,
    'LC_Control': lc_mean_control,
    'LC_NoHurricane': lc_mean_neither,
    'LC_NoHelene': lc_mean_nohelene,
    'LC_NoMilton': lc_mean_nomilton,
    'ES_Control': es_mean_control,
    'ES_NoHurricane': es_mean_neither,
    'ES_NoHelene': es_mean_nohelene,
    'ES_NoMilton': es_mean_nomilton
})

# Save results to CSV
output_file = f'{outdir}/sst_mean_absolute.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS - ABSOLUTE SST VALUES")
print("="*80)
print("\nLoop Current Area (LC): -88° to -85°E, 24° to 26.5°N")
print("-" * 80)
print(f"{'Case':<20} {'Mean (°C)':<15} {'Std (°C)':<15} {'Min (°C)':<15} {'Max (°C)':<15}")
print("-" * 80)
print(f"{'Control (Both)':<20} {np.nanmean(lc_mean_control):<15.3f} {np.nanstd(lc_mean_control):<15.3f} {np.nanmin(lc_mean_control):<15.3f} {np.nanmax(lc_mean_control):<15.3f}")
print(f"{'No-Hurricane':<20} {np.nanmean(lc_mean_neither):<15.3f} {np.nanstd(lc_mean_neither):<15.3f} {np.nanmin(lc_mean_neither):<15.3f} {np.nanmax(lc_mean_neither):<15.3f}")
print(f"{'No-Helene':<20} {np.nanmean(lc_mean_nohelene):<15.3f} {np.nanstd(lc_mean_nohelene):<15.3f} {np.nanmin(lc_mean_nohelene):<15.3f} {np.nanmax(lc_mean_nohelene):<15.3f}")
print(f"{'No-Milton':<20} {np.nanmean(lc_mean_nomilton):<15.3f} {np.nanstd(lc_mean_nomilton):<15.3f} {np.nanmin(lc_mean_nomilton):<15.3f} {np.nanmax(lc_mean_nomilton):<15.3f}")

print("\nEastern Shelf Area (ES): -88° to -82°E, 24° to 30°N")
print("-" * 80)
print(f"{'Case':<20} {'Mean (°C)':<15} {'Std (°C)':<15} {'Min (°C)':<15} {'Max (°C)':<15}")
print("-" * 80)
print(f"{'Control (Both)':<20} {np.nanmean(es_mean_control):<15.3f} {np.nanstd(es_mean_control):<15.3f} {np.nanmin(es_mean_control):<15.3f} {np.nanmax(es_mean_control):<15.3f}")
print(f"{'No-Hurricane':<20} {np.nanmean(es_mean_neither):<15.3f} {np.nanstd(es_mean_neither):<15.3f} {np.nanmin(es_mean_neither):<15.3f} {np.nanmax(es_mean_neither):<15.3f}")
print(f"{'No-Helene':<20} {np.nanmean(es_mean_nohelene):<15.3f} {np.nanstd(es_mean_nohelene):<15.3f} {np.nanmin(es_mean_nohelene):<15.3f} {np.nanmax(es_mean_nohelene):<15.3f}")
print(f"{'No-Milton':<20} {np.nanmean(es_mean_nomilton):<15.3f} {np.nanstd(es_mean_nomilton):<15.3f} {np.nanmin(es_mean_nomilton):<15.3f} {np.nanmax(es_mean_nomilton):<15.3f}")
print("="*80)

#%%----------------- Plot time series
print("\nCreating time series plots...")

# Define Milton time period for highlighting
milton_start = pd.to_datetime('2024-10-08')
milton_end = pd.to_datetime('2024-10-10')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Loop Current area
# Add filled area for Milton period
ax1.axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
ax1.plot(time_pd, lc_mean_control, 'k-', linewidth=2.5, label='Control (Both Hurricanes)', alpha=0.9, linestyle='-', zorder=3)
ax1.plot(time_pd, lc_mean_neither, 'b-', linewidth=2, label='No-Hurricane', alpha=0.8, zorder=3)
ax1.plot(time_pd, lc_mean_nohelene, 'r-', linewidth=2, label='No-Helene', alpha=0.8, zorder=3)
ax1.plot(time_pd, lc_mean_nomilton, 'g-', linewidth=2, label='No-Milton', alpha=0.8, zorder=3)
ax1.set_xlabel('Date', fontsize=14)
ax1.set_ylabel('Mean SST (°C)', fontsize=14)
ax1.set_title('Loop Current Area: Mean SST Time Series\n(-88° to -85°E, 24° to 26.5°N)', fontsize=16, fontweight='bold')
ax1.legend(loc='best', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=12)

# Plot 2: Eastern Shelf area
# Add filled area for Milton period
ax2.axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
ax2.plot(time_pd, es_mean_control, 'k-', linewidth=2.5, label='Control (Both Hurricanes)', alpha=0.9, linestyle='-', zorder=3)
ax2.plot(time_pd, es_mean_neither, 'b-', linewidth=2, label='No-Hurricane', alpha=0.8, zorder=3)
ax2.plot(time_pd, es_mean_nohelene, 'r-', linewidth=2, label='No-Helene', alpha=0.8, zorder=3)
ax2.plot(time_pd, es_mean_nomilton, 'g-', linewidth=2, label='No-Milton', alpha=0.8, zorder=3)
ax2.set_xlabel('Date', fontsize=14)
ax2.set_ylabel('Mean SST (°C)', fontsize=14)
ax2.set_title('Eastern Shelf Area: Mean SST Time Series\n(-88° to -82°E, 24° to 30°N)', fontsize=16, fontweight='bold')
ax2.legend(loc='best', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=12)

# Format x-axis dates
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(f'{outdir}/sst_mean_absolute_timeseries_comparison.png', dpi=300, bbox_inches='tight')
print(f"Time series plot saved to: {outdir}/sst_mean_absolute_timeseries_comparison.png")
plt.close()

# Create separate comparison plots for each area
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# LC area comparison
axes[0].axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
axes[0].plot(time_pd, lc_mean_control, 'k-', linewidth=2.5, label='Control (Both)', marker='o', markersize=3, alpha=0.8, linestyle='-', zorder=3)
axes[0].plot(time_pd, lc_mean_neither, 'b-', linewidth=2.5, label='No-Hurricane', marker='s', markersize=3, alpha=0.7, zorder=3)
axes[0].plot(time_pd, lc_mean_nohelene, 'r-', linewidth=2.5, label='No-Helene', marker='^', markersize=3, alpha=0.7, zorder=3)
axes[0].plot(time_pd, lc_mean_nomilton, 'g-', linewidth=2.5, label='No-Milton', marker='d', markersize=3, alpha=0.7, zorder=3)
axes[0].set_xlabel('Date', fontsize=13)
axes[0].set_ylabel('Mean SST (°C)', fontsize=13)
axes[0].set_title('Loop Current Area\n(-88° to -85°E, 24° to 26.5°N)', fontsize=14, fontweight='bold')
axes[0].legend(loc='best', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=11)

# ES area comparison
axes[1].axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
axes[1].plot(time_pd, es_mean_control, 'k-', linewidth=2.5, label='Control (Both)', marker='o', markersize=3, alpha=0.8, linestyle='-', zorder=3)
axes[1].plot(time_pd, es_mean_neither, 'b-', linewidth=2.5, label='No-Hurricane', marker='s', markersize=3, alpha=0.7, zorder=3)
axes[1].plot(time_pd, es_mean_nohelene, 'r-', linewidth=2.5, label='No-Helene', marker='^', markersize=3, alpha=0.7, zorder=3)
axes[1].plot(time_pd, es_mean_nomilton, 'g-', linewidth=2.5, label='No-Milton', marker='d', markersize=3, alpha=0.7, zorder=3)
axes[1].set_xlabel('Date', fontsize=13)
axes[1].set_ylabel('Mean SST (°C)', fontsize=13)
axes[1].set_title('Eastern Shelf Area\n(-88° to -82°E, 24° to 30°N)', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=11)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f'{outdir}/sst_mean_absolute_timeseries_side_by_side.png', dpi=300, bbox_inches='tight')
print(f"Side-by-side comparison plot saved to: {outdir}/sst_mean_absolute_timeseries_side_by_side.png")
plt.close()

# Create difference plots (relative to control)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# LC area differences from control
axes[0].axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
axes[0].plot(time_pd, lc_mean_neither - lc_mean_control, 'b-', linewidth=2.5, 
             label='No-Hurricane - Control', marker='s', markersize=3, alpha=0.7, zorder=3)
axes[0].plot(time_pd, lc_mean_nohelene - lc_mean_control, 'r-', linewidth=2.5, 
             label='No-Helene - Control', marker='^', markersize=3, alpha=0.7, zorder=3)
axes[0].plot(time_pd, lc_mean_nomilton - lc_mean_control, 'g-', linewidth=2.5, 
             label='No-Milton - Control', marker='d', markersize=3, alpha=0.7, zorder=3)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
axes[0].set_xlabel('Date', fontsize=13)
axes[0].set_ylabel('SST Difference from Control (°C)', fontsize=13)
axes[0].set_title('Loop Current Area: Difference from Control', fontsize=14, fontweight='bold')
axes[0].legend(loc='best', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=11)

# ES area differences from control
axes[1].axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
axes[1].plot(time_pd, es_mean_neither - es_mean_control, 'b-', linewidth=2.5, 
             label='No-Hurricane - Control', marker='s', markersize=3, alpha=0.7, zorder=3)
axes[1].plot(time_pd, es_mean_nohelene - es_mean_control, 'r-', linewidth=2.5, 
             label='No-Helene - Control', marker='^', markersize=3, alpha=0.7, zorder=3)
axes[1].plot(time_pd, es_mean_nomilton - es_mean_control, 'g-', linewidth=2.5, 
             label='No-Milton - Control', marker='d', markersize=3, alpha=0.7, zorder=3)
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
axes[1].set_xlabel('Date', fontsize=13)
axes[1].set_ylabel('SST Difference from Control (°C)', fontsize=13)
axes[1].set_title('Eastern Shelf Area: Difference from Control', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=11)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f'{outdir}/sst_mean_absolute_differences_from_control.png', dpi=300, bbox_inches='tight')
print(f"Difference from control plot saved to: {outdir}/sst_mean_absolute_differences_from_control.png")
plt.close()

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
print(f"\nAll outputs saved to: {outdir}/")
print("  - sst_mean_absolute.csv: Detailed time series data (absolute SST values)")
print("  - sst_mean_absolute_timeseries_comparison.png: Stacked time series plots")
print("  - sst_mean_absolute_timeseries_side_by_side.png: Side-by-side comparison")
print("  - sst_mean_absolute_differences_from_control.png: Difference plots from control")
