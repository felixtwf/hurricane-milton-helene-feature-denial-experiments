import numpy as np
import xarray as xr
import cmocean
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os

# Hurricane category definitions based on wind speed (mph)
def get_hurricane_category(vmax):
    """
    Determine hurricane category based on maximum wind speed (mph)
    
    Parameters:
    -----------
    vmax : float
        Maximum wind speed in mph
        
    Returns:
    --------
    category : int
        Hurricane category (0-5, where 0 = not a hurricane)
    """
    if vmax < 64:
        return 0  # Not a hurricane
    elif vmax < 83:
        return 1
    elif vmax < 96:
        return 2
    elif vmax <= 113:
        return 3
    elif vmax <= 137:
        return 4
    else:
        return 5

# 0 = Tropical storm [34<W<64]
# 1 = Category 1 [64<=W<83]
# 2 = Category 2 [83<=W<96]
# 3 = Category 3 [96<=W<113]
# 4 = Category 4 [113<=W<137]
# 5 = Category 5 [W >= 137]


def get_category_color(category):
    """
    Get color for hurricane category using different depths of blue
    
    Parameters:
    -----------
    category : int
        Hurricane category (0-5)
        
    Returns:
    --------
    color : str
        Color string for plotting
    """
    colors = {
        0: 'blue',     # Not a hurricane
        1: 'yellow',     # Category 1
        2: 'orange',       # Category 2
        3: 'red',          # Category 3
        4: 'magenta',      # Category 4
        5: 'purple'           # Category 5
    }
    return colors.get(category, 'lightgray')

def get_category_label(category):
    """
    Get label for hurricane category
    
    Parameters:
    -----------
    category : int
        Hurricane category (0-5)
        
    Returns:
    --------
    label : str
        Category label
    """
    labels = {
        0: 'Non-Hurricane',
        1: 'Cat 1',
        2: 'Cat 2', 
        3: 'Cat 3',
        4: 'Cat 4',
        5: 'Cat 5'
    }
    return labels.get(category, 'Unknown')

def plot_spatial_map(ax):
    """Set up spatial map with consistent formatting"""
    gulf_extent = [-94, -81, 20, 31]
    ax.set_extent(gulf_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='gray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='grey', alpha=0.6, zorder=2)
    gl.top_labels = False  
    gl.right_labels = False  
    gl.xlabel_style = {'fontsize': 12}
    gl.ylabel_style = {'fontsize': 12}
    gl.xlocator = plt.FixedLocator([-94, -92, -90, -88, -86, -84, -82, -80])
    gl.ylocator = plt.FixedLocator([18, 20, 22, 24, 26, 28, 30])
    return ax

def load_ohc_data(filepath, isotherm='12C'):
    """
    Load OHC data from netCDF file for a specific isotherm depth.
    
    Parameters:
    -----------
    filepath : str
        Path to directory containing OHC data
    isotherm : str
        Isotherm depth identifier ('12C', '20C', or '26C')
        
    Returns:
    --------
    ohc : numpy array
        OHC data array
    lon_rho : numpy array
        Longitude coordinates
    lat_rho : numpy array
        Latitude coordinates
    ocean_time : array-like
        Time coordinates
    """
    # Look for netCDF file with OHC data for specific isotherm
    ohc_file = f'{filepath}/OHC_{isotherm}.nc'
    if not os.path.exists(ohc_file):
        raise FileNotFoundError(f"Could not find {ohc_file}")
    
    # Load netCDF dataset
    ds = xr.open_dataset(ohc_file)
    
    # Extract OHC data (variable name is uppercase OHC according to data structure)
    if 'OHC' in ds:
        ohc = ds.OHC.values
    elif 'ohc' in ds:
        ohc = ds.ohc.values
    else:
        raise ValueError(f"OHC variable not found in {ohc_file}. Available variables: {list(ds.data_vars)}")
    
    # Extract and apply masks
    # 1. Apply land mask (mask_rho: 0 = land, 1 = water)
    if 'mask_rho' in ds:
        mask_rho = ds.mask_rho.values
        # Apply mask: set land areas (mask == 0) to NaN
        # mask_rho is 2D (eta_rho, xi_rho), OHC is 3D (ocean_time, eta_rho, xi_rho)
        # Need to broadcast mask to match OHC dimensions
        mask_3d = np.broadcast_to(mask_rho[np.newaxis, :, :], ohc.shape)
        ohc = np.where(mask_3d == 0, np.nan, ohc)
    
    # 2. Mask where isotherm_depth equals h (isotherm at or below seafloor)
    if 'isotherm_depth' in ds and 'h' in ds:
        isotherm_depth = ds.isotherm_depth.values
        h = ds.h.values
        # h is 2D (eta_rho, xi_rho), isotherm_depth is 3D (ocean_time, eta_rho, xi_rho)
        # Broadcast h to match isotherm_depth dimensions
        h_3d = np.broadcast_to(h[np.newaxis, :, :], isotherm_depth.shape)
        # Mask where isotherm_depth equals or exceeds h (isotherm at/below seafloor)
        isotherm_mask = np.isclose(isotherm_depth, h_3d, rtol=1e-5, atol=1e-5) | (isotherm_depth >= h_3d)
        ohc = np.where(isotherm_mask, np.nan, ohc)
    
    # Convert from J/m² to kJ/cm²
    # OHC is calculated as: sum(temp*wt)*Rho0*Cp
    # where temp is in °C, wt (layer thickness) is in m
    # Rho0 = 1035 kg/m³, Cp = 3985 J/(kg×°C)
    # Units: (°C × m) × (kg/m³) × (J/(kg×°C)) = J/m²
    # To convert J/m² to kJ/cm²: divide by 10,000,000 (1e7)
    # 1 m² = 10,000 cm², so 1 J/m² = 1e-4 J/cm² = 1e-7 kJ/cm²
    ohc = ohc / 1e7
    
    # Extract coordinates (lon_rho and lat_rho are variables in the netCDF file)
    # xarray can access them directly whether they're in coords or data_vars
    if 'lon_rho' in ds:
        lon_rho = ds.lon_rho.values
    elif 'lon' in ds:
        lon_rho = ds.lon.values
    else:
        raise ValueError(f"No longitude coordinate found in {ohc_file}")
    
    if 'lat_rho' in ds:
        lat_rho = ds.lat_rho.values
    elif 'lat' in ds:
        lat_rho = ds.lat.values
    else:
        raise ValueError(f"No latitude coordinate found in {ohc_file}")
    
    # Extract time coordinate
    if 'ocean_time' in ds:
        ocean_time = ds.ocean_time.values
    elif 'time' in ds:
        ocean_time = ds.time.values
    else:
        raise ValueError(f"No time coordinate found in {ohc_file}")
    
    return ohc, lon_rho, lat_rho, ocean_time

def plot_hurricane_track(ax, model_time, ds_h_all, ds_m_all):
    """
    Plot hurricane track on the map based on model time
    
    Parameters:
    -----------
    ax : matplotlib axes
        Axes to plot on
    model_time : pandas Timestamp
        Current model time
    ds_h_all : xarray Dataset
        Helene hurricane data
    ds_m_all : xarray Dataset
        Milton hurricane data
    """
    # Get the time ranges for each hurricane
    helene_start = pd.to_datetime(ds_h_all.time.min().values)
    helene_end = pd.to_datetime(ds_h_all.time.max().values)
    milton_start = pd.to_datetime(ds_m_all.time.min().values)
    milton_end = pd.to_datetime(ds_m_all.time.max().values)
    
    # Plot appropriate hurricane track based on model time
    if helene_start <= model_time <= helene_end:
        # Plot Helene track
        hurricane_name = "Helene"
        ds_hurricane = ds_h_all
        marker_shape = 'o'
        
        # Plot full track line (all timestamps)
        ax.plot(ds_hurricane.lon, ds_hurricane.lat, 'k--', linewidth=1, 
               transform=ccrs.PlateCarree(), label=f'{hurricane_name} Track', zorder=2)
        
        # Filter points up to and including current timestamp for markers only
        valid_indices = []
        for j, time_point in enumerate(ds_hurricane.time):
            point_time = pd.to_datetime(time_point.values)
            if point_time <= model_time:
                valid_indices.append(j)
        
        # Plot points with category colors (only up to current time)
        for j in valid_indices:
            point_time = pd.to_datetime(ds_hurricane.time.isel(time=j).values)
            vmax = ds_hurricane.vmax.isel(time=j).values
            category = get_hurricane_category(vmax)
            color = get_category_color(category)
            
            # Check if this timestamp matches the current model time (12:00)
            is_current_time = (point_time.date() == model_time.date() and point_time.hour == 12)
            edge_color = 'w' if is_current_time else 'none'
            line_width = 2 if is_current_time else 0
            
            # Plot point
            ax.scatter(ds_hurricane.lon.isel(time=j), ds_hurricane.lat.isel(time=j),
                      c=color, s=50, marker=marker_shape, edgecolors=edge_color, linewidth=line_width,
                      transform=ccrs.PlateCarree(), zorder=2)
    
    elif milton_start <= model_time <= milton_end:
        # Plot Milton track
        hurricane_name = "Milton"
        ds_hurricane = ds_m_all
        marker_shape = 'o'
        
        # Plot full track line (all timestamps)
        ax.plot(ds_hurricane.lon, ds_hurricane.lat, 'k--', linewidth=1, 
               transform=ccrs.PlateCarree(), label=f'{hurricane_name} Track', zorder=2)
        
        # Filter points up to and including current timestamp for markers only
        valid_indices = []
        for j, time_point in enumerate(ds_hurricane.time):
            point_time = pd.to_datetime(time_point.values)
            if point_time <= model_time:
                valid_indices.append(j)
        
        # Plot points with category colors (only up to current time)
        for j in valid_indices:
            point_time = pd.to_datetime(ds_hurricane.time.isel(time=j).values)
            vmax = ds_hurricane.vmax.isel(time=j).values
            category = get_hurricane_category(vmax)
            color = get_category_color(category)
            
            # Check if this timestamp matches the current model time (12:00)
            is_current_time = (point_time.date() == model_time.date() and point_time.hour == 12)
            edge_color = 'w' if is_current_time else 'none'
            line_width = 2 if is_current_time else 0
            
            # Plot point
            ax.scatter(ds_hurricane.lon.isel(time=j), ds_hurricane.lat.isel(time=j),
                      c=color, s=50, marker=marker_shape, edgecolors=edge_color, linewidth=line_width,
                      transform=ccrs.PlateCarree(), zorder=2)

#%%----------------- read OHC data    

# Base path for OHC data
base_path = '/projects/water/oomg/twu/UGOS/hurricanes/analysis/OHC'

# Define all cases to process
cases = [
    {
        'name': 'control',
        'filepath': f'{base_path}/ref'
    },
    {
        'name': 'No Helene',
        'filepath': f'{base_path}/rm_helene'
    },
    {
        'name': 'No Milton',
        'filepath': f'{base_path}/rm_milton'
    },
    {
        'name': 'No Hurricanes',
        'filepath': f'{base_path}/rm_both'
    }
]

# Output directory
output_base = './figs'
os.makedirs(f'{output_base}/original', exist_ok=True)
os.makedirs(f'{output_base}/difference', exist_ok=True)

#%%----------------- read hurricane track
hurricane_path = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/hurricanes'
ds_helene = xr.open_dataset(f'{hurricane_path}/helene.nc')
ds_milton = xr.open_dataset(f'{hurricane_path}/milton.nc')

# Select the period during which hurricane is stronger than cat1
storm_types = ['SD', 'SS', 'TD', 'TS', 'HU']
ds_h_sub = ds_helene.where(ds_helene.type.isin(storm_types), drop=True)
ds_m_sub = ds_milton.where(ds_milton.type.isin(storm_types), drop=True)

# Use all hurricane data (every 6 hours)
ds_h_all = ds_h_sub
ds_m_all = ds_m_sub

print(f"Helene total timestamps: {len(ds_h_all.time)}")
print(f"Milton total timestamps: {len(ds_m_all.time)}")

# Define isotherm depths to process
isotherms = ['12C', '20C', '26C']

# Load all OHC data first (for all isotherms)
ohc_data = {}
for case in cases:
    print(f"\nLoading OHC data for {case['name']}...")
    ohc_data[case['name']] = {}
    for isotherm in isotherms:
        try:
            ohc, lon_rho, lat_rho, ocean_time = load_ohc_data(case['filepath'], isotherm)
            ohc_data[case['name']][isotherm] = {
                'ohc': ohc,
                'lon_rho': lon_rho,
                'lat_rho': lat_rho,
                'ocean_time': ocean_time
            }
            print(f"  Successfully loaded OHC_{isotherm}: shape {ohc.shape}")
        except Exception as e:
            print(f"  Error loading OHC_{isotherm} for {case['name']}: {e}")
            raise

# Use coordinates from ref case, 12C (assuming all cases and isotherms have same grid)
ref_case_12C = ohc_data['control']['12C']
lon_rho = ref_case_12C['lon_rho']
lat_rho = ref_case_12C['lat_rho']
ocean_time = ref_case_12C['ocean_time']

# Determine OHC levels for each isotherm based on data range
levels_ohc = {}
levels_diff = {}

for isotherm in isotherms:
    # Get the range from all cases for this isotherm
    all_ohc_values = []
    for case_name, case_data in ohc_data.items():
        all_ohc_values.append(case_data[isotherm]['ohc'])
    all_ohc_array = np.concatenate([arr.flatten() for arr in all_ohc_values])
    ohc_min, ohc_max = np.nanmin(all_ohc_array), np.nanmax(all_ohc_array)

    # Calculate 95th percentile for better colorscale range
    ohc_p95 = np.nanpercentile(all_ohc_array, 95)

    # Create levels for original plots based on isotherm type
    if isotherm == '12C':
        # 12C has very high OHC values (max ~2626, p95 ~2147)
        # Use range 0-2200 with 100 interval
        levels_ohc[isotherm] = np.arange(0, 2201, 100)
    elif isotherm == '20C':
        # 20C has moderate OHC values (max ~839, p95 ~542)
        # Use range 0-600 with 50 interval
        levels_ohc[isotherm] = np.arange(0, 601, 50)
    else:  # 26C
        # 26C - keep original range 0-160 with 10 interval
        levels_ohc[isotherm] = np.arange(0, 161, 10)

    # For difference plots, use symmetric levels around zero
    ref_ohc = ohc_data['control'][isotherm]['ohc']
    diff_max = max([np.nanmax(np.abs(data[isotherm]['ohc'] - ref_ohc))
                    for case_name, data in ohc_data.items() if case_name != 'control'])

    # Set difference levels based on isotherm type
    if isotherm == '12C':
        levels_diff[isotherm] = np.arange(-300, 300.1, 10)
    elif isotherm == '20C':
        levels_diff[isotherm] = np.arange(-200, 200.1, 10)
    else:  # 26C
        levels_diff[isotherm] = np.arange(-50, 50.1, 5)

    print(f"\nOHC_{isotherm} range: {ohc_min:.2f} to {ohc_max:.2f}")
    print(f"OHC_{isotherm} 95th percentile: {ohc_p95:.2f}")
    print(f"OHC_{isotherm} difference range: -{diff_max:.2f} to {diff_max:.2f}")

#%%----------------- Plot original OHC for all cases and isotherms
print(f"\n{'='*60}")
print("Plotting original OHC for all cases and isotherms")
print(f"{'='*60}\n")

for case in cases:
    case_name = case['name']
    
    for isotherm in isotherms:
        print(f"Processing original OHC_{isotherm} for case: {case_name}")
        
        # Create output directory
        outpath = f'{output_base}/original/{case_name}/OHC_{isotherm}'
        os.makedirs(outpath, exist_ok=True)
        
        ohc = ohc_data[case_name][isotherm]['ohc']
        ocean_time = ohc_data[case_name][isotherm]['ocean_time']
        levels = levels_ohc[isotherm]
        
        for i in range(len(ocean_time)):
            print(f"  Processing timestep {i+1}/{len(ocean_time)}")
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), 
                                  subplot_kw={'projection': ccrs.PlateCarree()})
        
            # Create map
            plot_spatial_map(ax)
            
            # Plot OHC
            cf = ax.contourf(lon_rho, lat_rho, ohc[i], 
                            levels=levels, extend='max', 
                            transform=ccrs.PlateCarree(), 
                            cmap=cmocean.cm.thermal, zorder=0)
            
            # Add contour lines
            # cs2 = ax.contour(lon_rho, lat_rho, ohc[i], 
                            # levels=levels[::5], colors='k', linewidths=0.5, 
                            # transform=ccrs.PlateCarree(), zorder=3, alpha=0.3)
            
            # Get current model time
            model_time = pd.to_datetime(ocean_time[i])
            
            # Plot hurricane track
            plot_hurricane_track(ax, model_time, ds_h_all, ds_m_all)
            
            # Add legend for hurricane categories
            legend_elements = []
            for cat in range(1, 6):  # Categories 1-5
                color = get_category_color(cat)
                label = get_category_label(cat)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=6, 
                                                markeredgecolor='black', label=label, linestyle='None'))
            
            # Add legend
            legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                              bbox_to_anchor=(0.99, 0.99))
            legend.set_zorder(10)
            
            # Labels and title
            ax.set_xlabel('Longitude', fontsize=16)
            ax.set_ylabel('Latitude', fontsize=16)
            ax.set_title(f'OHC_{isotherm} - {case_name} - {model_time.strftime("%Y-%m-%d")}', 
                        fontsize=20)
            
            # Colorbar
            axpos = ax.get_position()
            cax = fig.add_axes([axpos.x1+.008, axpos.y0, 0.015, axpos.height])
            # Show ticks every 20 (0, 20, 40, 60, 80, 100, 120, 140, 160)
            cbar = fig.colorbar(cf, cax=cax, ticks=levels[::2], label=f'OHC_{isotherm} (kJ/cm²)')    
            cbar.ax.tick_params(labelsize=12)

            # Save figure
            plt.savefig(f'{outpath}/ohc_{isotherm}_{model_time.strftime("%Y_%m_%d")}.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Completed original plots for {case_name} - OHC_{isotherm}")

#%%----------------- Plot difference (other cases - ref) for all isotherms
print(f"\n{'='*60}")
print("Plotting OHC differences (other cases - ref) for all isotherms")
print(f"{'='*60}\n")

# Process difference cases (all except ref)
diff_cases = [case for case in cases if case['name'] != 'control']

for case in diff_cases:
    case_name = case['name']
    
    for isotherm in isotherms:
        print(f"Processing difference plots for case: {case_name} - OHC_{isotherm}")
        
        # Create output directory
        outpath = f'{output_base}/difference/{case_name}/OHC_{isotherm}'
        os.makedirs(outpath, exist_ok=True)
        
        ohc = ohc_data[case_name][isotherm]['ohc']
        ocean_time = ohc_data[case_name][isotherm]['ocean_time']
        ref_ohc = ohc_data['control'][isotherm]['ohc']
        ref_ocean_time = ohc_data['control'][isotherm]['ocean_time']
        levels = levels_diff[isotherm]
        
        # Calculate difference (ensure same time dimension)
        min_time = min(len(ocean_time), len(ref_ocean_time))
        ohc_diff = ref_ohc[:min_time] - ohc[:min_time] 
        
        for i in range(min_time):
            print(f"  Processing timestep {i+1}/{min_time}")
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), 
                                  subplot_kw={'projection': ccrs.PlateCarree()})
        
            # Create map
            plot_spatial_map(ax)
            
            # Plot OHC difference
            cf = ax.contourf(lon_rho, lat_rho, ohc_diff[i],
                            levels=levels, extend='both',
                            transform=ccrs.PlateCarree(),
                            cmap=cmocean.cm.balance, zorder=0)

            # # Add zero contour line
            # cs2 = ax.contour(lon_rho, lat_rho, ohc_diff[i],
            #                 [0], colors='k', linewidths=2,
            #                 transform=ccrs.PlateCarree(), zorder=3)

            # # Add other contour lines
            # cs3 = ax.contour(lon_rho, lat_rho, ohc_diff[i],
            #                 levels=levels[::5], colors='k', linewidths=0.5,
            #                 transform=ccrs.PlateCarree(), zorder=3, alpha=0.3)
            
            # Get current model time (use ref time for consistency)
            model_time = pd.to_datetime(ref_ocean_time[i])
            
            # Plot hurricane track
            plot_hurricane_track(ax, model_time, ds_h_all, ds_m_all)
            
            # Add legend for hurricane categories
            legend_elements = []
            for cat in range(1, 6):  # Categories 1-5
                color = get_category_color(cat)
                label = get_category_label(cat)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=6, 
                                                markeredgecolor='black', label=label, linestyle='None'))
            
            # Add legend
            legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                              bbox_to_anchor=(0.99, 0.99))
            legend.set_zorder(10)
            
            # Labels and title
            ax.set_xlabel('Longitude', fontsize=16)
            ax.set_ylabel('Latitude', fontsize=16)
            ax.set_title(f'OHC_{isotherm} Diff. (control - {case_name}) - {model_time.strftime("%Y-%m-%d")}', 
                        fontsize=20)
            
            # Colorbar
            axpos = ax.get_position()
            cax = fig.add_axes([axpos.x1+.008, axpos.y0, 0.015, axpos.height])
            cbar = fig.colorbar(cf, cax=cax, ticks=levels[::5], label=f'Δ OHC_{isotherm} (kJ/cm²)')    
            cbar.ax.tick_params(labelsize=12)

            # Save figure
            plt.savefig(f'{outpath}/ohc_{isotherm}_diff_{model_time.strftime("%Y_%m_%d")}.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Completed difference plots for {case_name} - OHC_{isotherm}")

print("\nAll plots completed for all cases!")

