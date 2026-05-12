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
outdir = './sst_diff_analysis'
os.makedirs(outdir, exist_ok=True)

# Load reference dataset to get coordinates
ds_ref = xr.open_dataset(f'{filepath}/sst.nc')
lon_rho = ds_ref.lon_rho.values
lat_rho = ds_ref.lat_rho.values
ocean_time = ds_ref.ocean_time.values

# Load SST data for control and all three cases
print("Loading SST data...")
sst_ref = np.load(f'{filepath}/sst.npz')['sst']  # Control case (both hurricanes)
sst_neither = np.load(f'{filepath_neither}/sst.npz')['sst']  # No-hurricane case
sst_nohelene = np.load(f'{filepath_nohelene}/sst.npz')['sst']  # No-Helene case
sst_nomilton = np.load(f'{filepath_nomilton}/sst.npz')['sst']  # No-Milton case

# Calculate SST differences (Control - Case)
# These differences highlight the impacts of hurricanes
print("Calculating SST differences...")
sst_diff_neither = sst_ref - sst_neither   # Control - No-Hurricane (both hurricanes' impacts)
sst_diff_nohelene = sst_ref - sst_nohelene # Control - No-Helene (Helene's impacts)
sst_diff_nomilton = sst_ref - sst_nomilton # Control - No-Milton (Milton's impacts)

print(f"SST data shape: {sst_ref.shape}")
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

# Calculate mean SST differences for each area and each case
print("Calculating mean SST differences...")
ntime = len(ocean_time)

# Initialize arrays to store mean SST differences
lc_mean_both = np.zeros(ntime)      # Control - No-Hurricane (both hurricanes' impacts)
lc_mean_helene = np.zeros(ntime)   # Control - No-Helene (Helene's impacts)
lc_mean_milton = np.zeros(ntime)   # Control - No-Milton (Milton's impacts)

es_mean_both = np.zeros(ntime)     # Control - No-Hurricane (both hurricanes' impacts)
es_mean_helene = np.zeros(ntime)   # Control - No-Helene (Helene's impacts)
es_mean_milton = np.zeros(ntime)  # Control - No-Milton (Milton's impacts)

for i in range(ntime):
    # LC area means from differences
    lc_mean_both[i] = np.nanmean(sst_diff_neither[i][lc_mask])
    lc_mean_helene[i] = np.nanmean(sst_diff_nohelene[i][lc_mask])
    lc_mean_milton[i] = np.nanmean(sst_diff_nomilton[i][lc_mask])
    
    # Eastern Shelf area means from differences
    es_mean_both[i] = np.nanmean(sst_diff_neither[i][es_mask])
    es_mean_helene[i] = np.nanmean(sst_diff_nohelene[i][es_mask])
    es_mean_milton[i] = np.nanmean(sst_diff_nomilton[i][es_mask])

# Convert time to pandas datetime
time_pd = pd.to_datetime(ocean_time)

# Create DataFrame for output
results_df = pd.DataFrame({
    'Time': time_pd,
    'LC_Control_NoHurricane': lc_mean_both,
    'LC_Control_NoHelene': lc_mean_helene,
    'LC_Control_NoMilton': lc_mean_milton,
    'ES_Control_NoHurricane': es_mean_both,
    'ES_Control_NoHelene': es_mean_helene,
    'ES_Control_NoMilton': es_mean_milton
})

# Save results to CSV
output_file = f'{outdir}/sst_mean_differences.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS - SST DIFFERENCES (Control - Case)")
print("="*80)
print("\nLoop Current Area (LC): -88° to -85°E, 24° to 26.5°N")
print("(Positive values indicate cooling, negative values indicate warming)")
print("-" * 80)
print(f"{'Case':<30} {'Mean (°C)':<15} {'Std (°C)':<15} {'Min (°C)':<15} {'Max (°C)':<15}")
print("-" * 80)
print(f"{'Control - No-Hurricane (Both)':<30} {np.nanmean(lc_mean_both):<15.3f} {np.nanstd(lc_mean_both):<15.3f} {np.nanmin(lc_mean_both):<15.3f} {np.nanmax(lc_mean_both):<15.3f}")
print(f"{'Control - No-Helene':<30} {np.nanmean(lc_mean_helene):<15.3f} {np.nanstd(lc_mean_helene):<15.3f} {np.nanmin(lc_mean_helene):<15.3f} {np.nanmax(lc_mean_helene):<15.3f}")
print(f"{'Control - No-Milton':<30} {np.nanmean(lc_mean_milton):<15.3f} {np.nanstd(lc_mean_milton):<15.3f} {np.nanmin(lc_mean_milton):<15.3f} {np.nanmax(lc_mean_milton):<15.3f}")

print("\nEastern Shelf Area (ES): -88° to -82°E, 24° to 30°N")
print("(Positive values indicate cooling, negative values indicate warming)")
print("-" * 80)
print(f"{'Case':<30} {'Mean (°C)':<15} {'Std (°C)':<15} {'Min (°C)':<15} {'Max (°C)':<15}")
print("-" * 80)
print(f"{'Control - No-Hurricane (Both)':<30} {np.nanmean(es_mean_both):<15.3f} {np.nanstd(es_mean_both):<15.3f} {np.nanmin(es_mean_both):<15.3f} {np.nanmax(es_mean_both):<15.3f}")
print(f"{'Control - No-Helene':<30} {np.nanmean(es_mean_helene):<15.3f} {np.nanstd(es_mean_helene):<15.3f} {np.nanmin(es_mean_helene):<15.3f} {np.nanmax(es_mean_helene):<15.3f}")
print(f"{'Control - No-Milton':<30} {np.nanmean(es_mean_milton):<15.3f} {np.nanstd(es_mean_milton):<15.3f} {np.nanmin(es_mean_milton):<15.3f} {np.nanmax(es_mean_milton):<15.3f}")
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
ax1.plot(time_pd, lc_mean_both, 'b-', linewidth=2, label='Control - No-Hurricane (Both)', alpha=0.8, zorder=3)
ax1.plot(time_pd, lc_mean_helene, 'r-', linewidth=2, label='Control - No-Helene', alpha=0.8, zorder=3)
ax1.plot(time_pd, lc_mean_milton, 'g-', linewidth=2, label='Control - No-Milton', alpha=0.8, zorder=3)
ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
ax1.set_xlabel('Date', fontsize=14)
ax1.set_ylabel('Mean SST Difference (°C)', fontsize=14)
ax1.set_title('Loop Current Area: SST Difference Time Series\n(Control - Case, -88° to -85°E, 24° to 26.5°N)', fontsize=16, fontweight='bold')
ax1.legend(loc='best', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=12)

# Plot 2: Eastern Shelf area
# Add filled area for Milton period
ax2.axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
ax2.plot(time_pd, es_mean_both, 'b-', linewidth=2, label='Control - No-Hurricane (Both)', alpha=0.8, zorder=3)
ax2.plot(time_pd, es_mean_helene, 'r-', linewidth=2, label='Control - No-Helene', alpha=0.8, zorder=3)
ax2.plot(time_pd, es_mean_milton, 'g-', linewidth=2, label='Control - No-Milton', alpha=0.8, zorder=3)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
ax2.set_xlabel('Date', fontsize=14)
ax2.set_ylabel('Mean SST Difference (°C)', fontsize=14)
ax2.set_title('Eastern Shelf Area: SST Difference Time Series\n(Control - Case, -88° to -82°E, 24° to 30°N)', fontsize=16, fontweight='bold')
ax2.legend(loc='best', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=12)

# Format x-axis dates
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(f'{outdir}/sst_diff_timeseries_comparison.png', dpi=300, bbox_inches='tight')
print(f"Time series plot saved to: {outdir}/sst_diff_timeseries_comparison.png")
plt.close()

# Create separate comparison plots for each area
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# LC area comparison
axes[0].axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
axes[0].plot(time_pd, lc_mean_both, 'b-', linewidth=2.5, label='Control - No-Hurricane (Both)', marker='o', markersize=3, alpha=0.7, zorder=3)
axes[0].plot(time_pd, lc_mean_helene, 'r-', linewidth=2.5, label='Control - No-Helene', marker='s', markersize=3, alpha=0.7, zorder=3)
axes[0].plot(time_pd, lc_mean_milton, 'g-', linewidth=2.5, label='Control - No-Milton', marker='^', markersize=3, alpha=0.7, zorder=3)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
axes[0].set_xlabel('Date', fontsize=13)
axes[0].set_ylabel('Mean SST Difference (°C)', fontsize=13)
axes[0].set_title('Loop Current Area\n(-88° to -85°E, 24° to 26.5°N)', fontsize=14, fontweight='bold')
axes[0].legend(loc='best', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=11)

# ES area comparison
axes[1].axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
axes[1].plot(time_pd, es_mean_both, 'b-', linewidth=2.5, label='Control - No-Hurricane (Both)', marker='o', markersize=3, alpha=0.7, zorder=3)
axes[1].plot(time_pd, es_mean_helene, 'r-', linewidth=2.5, label='Control - No-Helene', marker='s', markersize=3, alpha=0.7, zorder=3)
axes[1].plot(time_pd, es_mean_milton, 'g-', linewidth=2.5, label='Control - No-Milton', marker='^', markersize=3, alpha=0.7, zorder=3)
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
axes[1].set_xlabel('Date', fontsize=13)
axes[1].set_ylabel('Mean SST Difference (°C)', fontsize=13)
axes[1].set_title('Eastern Shelf Area\n(-88° to -82°E, 24° to 30°N)', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=11)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f'{outdir}/sst_diff_timeseries_side_by_side.png', dpi=300, bbox_inches='tight')
print(f"Side-by-side comparison plot saved to: {outdir}/sst_diff_timeseries_side_by_side.png")
plt.close()

# Create plots showing individual hurricane impacts
# (Both impacts - Individual impact = Other hurricane's impact)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# LC area: Calculate individual hurricane impacts
# Milton's impact = Both impacts - Helene's impact
# Helene's impact = Both impacts - Milton's impact
lc_milton_only = lc_mean_both - lc_mean_helene  # Milton's isolated impact
lc_helene_only = lc_mean_both - lc_mean_milton  # Helene's isolated impact

axes[0].axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
axes[0].plot(time_pd, lc_helene_only, 'r-', linewidth=2.5, 
             label="Helene's Isolated Impact", marker='s', markersize=3, alpha=0.7, zorder=3)
axes[0].plot(time_pd, lc_milton_only, 'g-', linewidth=2.5, 
             label="Milton's Isolated Impact", marker='^', markersize=3, alpha=0.7, zorder=3)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
axes[0].set_xlabel('Date', fontsize=13)
axes[0].set_ylabel('SST Difference (°C)', fontsize=13)
axes[0].set_title('Loop Current Area: Individual Hurricane Impacts', fontsize=14, fontweight='bold')
axes[0].legend(loc='best', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=11)

# ES area: Calculate individual hurricane impacts
es_milton_only = es_mean_both - es_mean_helene  # Milton's isolated impact
es_helene_only = es_mean_both - es_mean_milton  # Helene's isolated impact

axes[1].axvspan(milton_start, milton_end, alpha=0.2, color='orange', zorder=0, label='Milton Period (10/8-10/10)')
axes[1].plot(time_pd, es_helene_only, 'r-', linewidth=2.5, 
             label="Helene's Isolated Impact", marker='s', markersize=3, alpha=0.7, zorder=3)
axes[1].plot(time_pd, es_milton_only, 'g-', linewidth=2.5, 
             label="Milton's Isolated Impact", marker='^', markersize=3, alpha=0.7, zorder=3)
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
axes[1].set_xlabel('Date', fontsize=13)
axes[1].set_ylabel('SST Difference (°C)', fontsize=13)
axes[1].set_title('Eastern Shelf Area: Individual Hurricane Impacts', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=11)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f'{outdir}/sst_individual_impacts.png', dpi=300, bbox_inches='tight')
print(f"Individual impacts plot saved to: {outdir}/sst_individual_impacts.png")
plt.close()

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
print(f"\nAll outputs saved to: {outdir}/")
print("  - sst_mean_differences.csv: Detailed time series data (Control - Case)")
print("  - sst_diff_timeseries_comparison.png: Stacked time series plots")
print("  - sst_diff_timeseries_side_by_side.png: Side-by-side comparison")
print("  - sst_individual_impacts.png: Individual hurricane impact plots")
print("\nNote: Positive values indicate cooling, negative values indicate warming")
print("      (Control SST - Case SST)")
