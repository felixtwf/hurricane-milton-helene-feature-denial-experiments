import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os

def calculate_vorticity(u, v, lon, lat):
    """
    Calculate dimensionless relative vorticity: ζ/f = (∂v/∂x - ∂u/∂y) / f
    where f is the Coriolis parameter

    Parameters:
    -----------
    u : ndarray
        Zonal velocity component (m/s)
    v : ndarray
        Meridional velocity component (m/s)
    lon : ndarray
        Longitude array (degrees)
    lat : ndarray
        Latitude array (degrees)

    Returns:
    --------
    vorticity : ndarray
        Dimensionless relative vorticity (ζ/f)
    """
    # Convert degrees to radians for proper distance calculation
    R = 6371000  # Earth radius in meters
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Calculate grid spacing in meters
    # For latitude (y-direction): dy = R * d(lat in radians)
    dlat = np.gradient(lat_rad, axis=0)
    dy = R * dlat

    # For longitude (x-direction): dx = R * cos(lat) * d(lon in radians)
    dlon = np.gradient(lon_rad, axis=1)
    dx = R * np.cos(lat_rad) * dlon

    # Calculate gradients
    # ∂v/∂x
    dv_dx = np.gradient(v, axis=1) / dx

    # ∂u/∂y
    du_dy = np.gradient(u, axis=0) / dy

    # Relative vorticity: ζ = ∂v/∂x - ∂u/∂y
    zeta = dv_dx - du_dy

    # Calculate Coriolis parameter: f = 2 * Ω * sin(lat)
    # where Ω = 7.2921e-5 rad/s (Earth's angular velocity)
    omega = 7.2921e-5  # rad/s
    f = 2 * omega * np.sin(lat_rad)

    # Calculate dimensionless vorticity ζ/f
    # Add small epsilon to avoid division by zero near equator
    epsilon = 1e-10
    vorticity = zeta / (f + epsilon)

    return vorticity

#%%----------------- read CNAPS2
filepath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/both'
filepath_neither = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/neither'
filepath_nohelene = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nohelene'
filepath_nomilton = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nomilton'

# Output directory for results
outdir = './vorticity_diff_analysis'
os.makedirs(outdir, exist_ok=True)

# Load reference dataset to get coordinates
ds_ref = xr.open_dataset(f'{filepath}/sst.nc')
lon_rho = ds_ref.lon_rho.values
lat_rho = ds_ref.lat_rho.values
ocean_time = ds_ref.ocean_time.values

# Load current data for control and all three cases
print("Loading current data...")
with np.load(f'{filepath}/current_0.npz') as ref_surf:
    surfu_ref = ref_surf['surfu']
    surfv_ref = ref_surf['surfv']

with np.load(f'{filepath_neither}/current_0.npz') as neither_surf:
    surfu_neither = neither_surf['surfu']
    surfv_neither = neither_surf['surfv']

with np.load(f'{filepath_nohelene}/current_0.npz') as nohelene_surf:
    surfu_nohelene = nohelene_surf['surfu']
    surfv_nohelene = nohelene_surf['surfv']

with np.load(f'{filepath_nomilton}/current_0.npz') as nomilton_surf:
    surfu_nomilton = nomilton_surf['surfu']
    surfv_nomilton = nomilton_surf['surfv']

print(f"Current data shape: {surfu_ref.shape}")
print(f"Number of timesteps: {len(ocean_time)}")

# Calculate vorticity for all cases
print("Calculating vorticity for all cases...")
ntime = len(ocean_time)
vort_ref = np.zeros_like(surfu_ref)
vort_neither = np.zeros_like(surfu_neither)
vort_nohelene = np.zeros_like(surfu_nohelene)
vort_nomilton = np.zeros_like(surfu_nomilton)

for i in range(ntime):
    print(f"Processing timestep {i+1}/{ntime}")
    vort_ref[i] = calculate_vorticity(surfu_ref[i], surfv_ref[i], lon_rho, lat_rho)
    vort_neither[i] = calculate_vorticity(surfu_neither[i], surfv_neither[i], lon_rho, lat_rho)
    vort_nohelene[i] = calculate_vorticity(surfu_nohelene[i], surfv_nohelene[i], lon_rho, lat_rho)
    vort_nomilton[i] = calculate_vorticity(surfu_nomilton[i], surfv_nomilton[i], lon_rho, lat_rho)

# Calculate vorticity differences (Control - Case)
print("Calculating vorticity differences...")
vort_diff_neither = vort_ref - vort_neither   # Control - No-Hurricane (both hurricanes' impacts)
vort_diff_nohelene = vort_ref - vort_nohelene # Control - No-Helene (Helene's impacts)
vort_diff_nomilton = vort_ref - vort_nomilton # Control - No-Milton (Milton's impacts)

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

# Calculate mean vorticity differences for each area and each case
print("Calculating mean vorticity differences...")

# Initialize arrays to store mean vorticity differences and absolute values
lc_mean_both = np.zeros(ntime)      # Control - No-Hurricane (both hurricanes' impacts)
lc_mean_helene = np.zeros(ntime)   # Control - No-Helene (Helene's impacts)
lc_mean_milton = np.zeros(ntime)   # Control - No-Milton (Milton's impacts)

es_mean_both = np.zeros(ntime)     # Control - No-Hurricane (both hurricanes' impacts)
es_mean_helene = np.zeros(ntime)   # Control - No-Helene (Helene's impacts)
es_mean_milton = np.zeros(ntime)  # Control - No-Milton (Milton's impacts)

# Also store absolute vorticity values for control case
lc_mean_ref = np.zeros(ntime)
es_mean_ref = np.zeros(ntime)

for i in range(ntime):
    # LC area means from differences
    lc_mean_both[i] = np.nanmean(vort_diff_neither[i][lc_mask])
    lc_mean_helene[i] = np.nanmean(vort_diff_nohelene[i][lc_mask])
    lc_mean_milton[i] = np.nanmean(vort_diff_nomilton[i][lc_mask])
    lc_mean_ref[i] = np.nanmean(vort_ref[i][lc_mask])

    # Eastern Shelf area means from differences
    es_mean_both[i] = np.nanmean(vort_diff_neither[i][es_mask])
    es_mean_helene[i] = np.nanmean(vort_diff_nohelene[i][es_mask])
    es_mean_milton[i] = np.nanmean(vort_diff_nomilton[i][es_mask])
    es_mean_ref[i] = np.nanmean(vort_ref[i][es_mask])

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
output_file = f'{outdir}/vorticity_mean_differences.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Calculate percentage changes relative to control case absolute values
# To avoid division by very small numbers, calculate percentage based on mean absolute vorticity
lc_ref_mean = np.nanmean(np.abs(lc_mean_ref))
es_ref_mean = np.nanmean(np.abs(es_mean_ref))

# Calculate percentage changes
lc_pct_both = (np.nanmean(lc_mean_both) / lc_ref_mean) * 100
lc_pct_helene = (np.nanmean(lc_mean_helene) / lc_ref_mean) * 100
lc_pct_milton = (np.nanmean(lc_mean_milton) / lc_ref_mean) * 100

es_pct_both = (np.nanmean(es_mean_both) / es_ref_mean) * 100
es_pct_helene = (np.nanmean(es_mean_helene) / es_ref_mean) * 100
es_pct_milton = (np.nanmean(es_mean_milton) / es_ref_mean) * 100

# Print summary statistics
print("\n" + "="*90)
print("SUMMARY STATISTICS - VORTICITY DIFFERENCES (ζ/f) (Control - Case)")
print("="*90)
print("\nLoop Current Area (LC): -88° to -85°E, 24° to 26.5°N")
print("(Positive values indicate increased cyclonic vorticity)")
print(f"Control case mean |ζ/f|: {lc_ref_mean:.4f}")
print("-" * 90)
print(f"{'Case':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'% Change':<12}")
print("-" * 90)
print(f"{'Control - No-Hurricane (Both)':<30} {np.nanmean(lc_mean_both):<12.4f} {np.nanstd(lc_mean_both):<12.4f} {np.nanmin(lc_mean_both):<12.4f} {np.nanmax(lc_mean_both):<12.4f} {lc_pct_both:<12.2f}")
print(f"{'Control - No-Helene':<30} {np.nanmean(lc_mean_helene):<12.4f} {np.nanstd(lc_mean_helene):<12.4f} {np.nanmin(lc_mean_helene):<12.4f} {np.nanmax(lc_mean_helene):<12.4f} {lc_pct_helene:<12.2f}")
print(f"{'Control - No-Milton':<30} {np.nanmean(lc_mean_milton):<12.4f} {np.nanstd(lc_mean_milton):<12.4f} {np.nanmin(lc_mean_milton):<12.4f} {np.nanmax(lc_mean_milton):<12.4f} {lc_pct_milton:<12.2f}")

print("\nEastern Shelf Area (ES): -88° to -82°E, 24° to 30°N")
print("(Positive values indicate increased cyclonic vorticity)")
print(f"Control case mean |ζ/f|: {es_ref_mean:.4f}")
print("-" * 90)
print(f"{'Case':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'% Change':<12}")
print("-" * 90)
print(f"{'Control - No-Hurricane (Both)':<30} {np.nanmean(es_mean_both):<12.4f} {np.nanstd(es_mean_both):<12.4f} {np.nanmin(es_mean_both):<12.4f} {np.nanmax(es_mean_both):<12.4f} {es_pct_both:<12.2f}")
print(f"{'Control - No-Helene':<30} {np.nanmean(es_mean_helene):<12.4f} {np.nanstd(es_mean_helene):<12.4f} {np.nanmin(es_mean_helene):<12.4f} {np.nanmax(es_mean_helene):<12.4f} {es_pct_helene:<12.2f}")
print(f"{'Control - No-Milton':<30} {np.nanmean(es_mean_milton):<12.4f} {np.nanstd(es_mean_milton):<12.4f} {np.nanmin(es_mean_milton):<12.4f} {np.nanmax(es_mean_milton):<12.4f} {es_pct_milton:<12.2f}")
print("="*90)
print("\nNote: % Change = (Mean Difference / Control Mean |ζ/f|) × 100")

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
ax1.set_ylabel('Mean Vorticity Difference (ζ/f)', fontsize=14)
ax1.set_title('Loop Current Area: Vorticity Difference Time Series\n(Control - Case, -88° to -85°E, 24° to 26.5°N)', fontsize=16, fontweight='bold')
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
ax2.set_ylabel('Mean Vorticity Difference (ζ/f)', fontsize=14)
ax2.set_title('Eastern Shelf Area: Vorticity Difference Time Series\n(Control - Case, -88° to -82°E, 24° to 30°N)', fontsize=16, fontweight='bold')
ax2.legend(loc='best', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=12)

# Format x-axis dates
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig(f'{outdir}/vorticity_diff_timeseries_comparison.png', dpi=300, bbox_inches='tight')
print(f"Time series plot saved to: {outdir}/vorticity_diff_timeseries_comparison.png")
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
axes[0].set_ylabel('Mean Vorticity Difference (ζ/f)', fontsize=13)
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
axes[1].set_ylabel('Mean Vorticity Difference (ζ/f)', fontsize=13)
axes[1].set_title('Eastern Shelf Area\n(-88° to -82°E, 24° to 30°N)', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=11)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f'{outdir}/vorticity_diff_timeseries_side_by_side.png', dpi=300, bbox_inches='tight')
print(f"Side-by-side comparison plot saved to: {outdir}/vorticity_diff_timeseries_side_by_side.png")
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
axes[0].set_ylabel('Vorticity Difference (ζ/f)', fontsize=13)
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
axes[1].set_ylabel('Vorticity Difference (ζ/f)', fontsize=13)
axes[1].set_title('Eastern Shelf Area: Individual Hurricane Impacts', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=11)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f'{outdir}/vorticity_individual_impacts.png', dpi=300, bbox_inches='tight')
print(f"Individual impacts plot saved to: {outdir}/vorticity_individual_impacts.png")
plt.close()

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
print(f"\nAll outputs saved to: {outdir}/")
print("  - vorticity_mean_differences.csv: Detailed time series data (Control - Case)")
print("  - vorticity_diff_timeseries_comparison.png: Stacked time series plots")
print("  - vorticity_diff_timeseries_side_by_side.png: Side-by-side comparison")
print("  - vorticity_individual_impacts.png: Individual hurricane impact plots")
print("\nNote: Positive values indicate increased cyclonic vorticity")
print("      (Control vorticity - Case vorticity)")
