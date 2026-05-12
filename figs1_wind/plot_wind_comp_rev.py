import numpy as np
import xarray as xr
import cmocean
import matplotlib.pyplot as plt
import cool_maps.plot as cplt
import cartopy.crs as ccrs
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

#%%----------------- read wind data    

# Wind data files
wind_file_hurricane = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/forcing_ref/useast_wind_era_REF.nc'
wind_file_climatology = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/forcing_noHC/useast_wind_era_no_hurricanes.nc'

# Read wind datasets
ds_wind_hurricane = xr.open_dataset(wind_file_hurricane)
ds_wind_climatology = xr.open_dataset(wind_file_climatology)

# Get coordinates and wind components
lon = ds_wind_hurricane.lon.values
lat = ds_wind_hurricane.lat.values
u_var = 'Uwind'
v_var = 'Vwind'

#%%----------------- read hurricane track
ds_helene = xr.open_dataset('/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/hurricanes/helene.nc')
ds_milton = xr.open_dataset('/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/hurricanes/milton.nc')

# Select the period during which hurricane is stronger than cat1
storm_types = ['SD', 'SS', 'TD', 'TS', 'HU']
ds_h_all = ds_helene.where(ds_helene.type.isin(storm_types), drop=True)
ds_m_all = ds_milton.where(ds_milton.type.isin(storm_types), drop=True)

#%%----------------- Plot wind fields with hurricane tracks
extent = [-94, -81, 20, 31]
levels_wind = np.arange(0, 16.01, 0.5)  # Wind speed levels in m/s
dd = 4; mm = 4  # Downsampling for vectors/streamlines

# Target dates
helene_date = pd.to_datetime('2024-09-26')
milton_date = pd.to_datetime('2024-10-09')

# Get hurricane track times
helene_track_times = pd.to_datetime(ds_h_all.time.values)
milton_track_times = pd.to_datetime(ds_m_all.time.values)

# Find the 3rd timestamp on 9/26 for Helene and 4th timestamp on 10/9 for Milton
helene_date_indices = np.where([t.date() == helene_date.date() for t in helene_track_times])[0]
milton_date_indices = np.where([t.date() == milton_date.date() for t in milton_track_times])[0]

helene_target_idx = helene_date_indices[2] if len(helene_date_indices) > 2 else None
milton_target_idx = milton_date_indices[3] if len(milton_date_indices) > 3 else None

# Get the actual hurricane timestamps
helene_target_time = helene_track_times[helene_target_idx] if helene_target_idx is not None else None
milton_target_time = milton_track_times[milton_target_idx] if milton_target_idx is not None else None

# Find closest wind time to each hurricane timestamp
time_dim = 'wind_time'
wind_time_hurricane = pd.to_datetime(ds_wind_hurricane[time_dim].values)
wind_time_climatology = pd.to_datetime(ds_wind_climatology[time_dim].values)

# Use the same wind time for both hurricane and climatology data (closest to hurricane timestamp)
helene_idx_hurricane = np.argmin(np.abs(wind_time_hurricane - helene_target_time)) if helene_target_time is not None else None
helene_idx_climatology = np.argmin(np.abs(wind_time_climatology - helene_target_time)) if helene_target_time is not None else None
milton_idx_hurricane = np.argmin(np.abs(wind_time_hurricane - milton_target_time)) if milton_target_time is not None else None
milton_idx_climatology = np.argmin(np.abs(wind_time_climatology - milton_target_time)) if milton_target_time is not None else None

# Create 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), 
                        subplot_kw={'projection': ccrs.Mercator()})
axes = axes.flatten()

# Helper function to plot wind field
def plot_wind_field(ax, lon, lat, u, v, wind_speed, title, ds_hurricane=None, target_time_idx=None):
    """Plot wind field on a map"""
    # Create map
    cplt.create(extent, ax=ax, proj=ccrs.Mercator(), gridlines=False, bathymetry=True)
    
    # Plot wind speed as filled contours
    cf = ax.contourf(lon, lat, wind_speed, 
                    levels=levels_wind, extend='max', 
                    cmap=cmocean.cm.speed, 
                    transform=ccrs.PlateCarree())
    
    # Plot wind vectors (downsampled)
    lon_sub = lon[::dd, ::mm]
    lat_sub = lat[::dd, ::mm]
    u_sub = u[::dd, ::mm]
    v_sub = v[::dd, ::mm]
    
    q = ax.quiver(lon_sub, lat_sub, u_sub, v_sub,
             transform=ccrs.PlateCarree(), color='k',
             scale=200, width=0.005, headwidth=5)
    
    # Add quiverkey to each subplot
    ax.quiverkey(q, 1, 1.05, 10, '10 m/s', labelpos='E', 
                 coordinates='axes', fontproperties={'size': 9})
    
    # Plot hurricane track if provided
    if ds_hurricane is not None and target_time_idx is not None:
        # Plot track line
        ax.plot(ds_hurricane.lon, ds_hurricane.lat, 'k--', linewidth=1.5, 
               transform=ccrs.PlateCarree(), zorder=49)
        
        # Plot points with category colors
        for j, time_point in enumerate(ds_hurricane.time):
            vmax = ds_hurricane.vmax.isel(time=j).values
            category = get_hurricane_category(vmax)
            color = get_category_color(category)
            
            # Highlight point at target time index
            is_target_time = (j == target_time_idx)
            edge_color = 'k' if is_target_time else 'none'
            line_width = 2.5 if is_target_time else 0
            
            # Plot point
            scatter = ax.scatter(ds_hurricane.lon.isel(time=j), ds_hurricane.lat.isel(time=j),
                      c=color, s=60, marker='o', edgecolors=edge_color, linewidth=line_width,
                      transform=ccrs.PlateCarree(), zorder=50)
            
            # Add time label next to the target marker
            if is_target_time:
                point_time = pd.to_datetime(time_point.values)
                time_str = point_time.strftime('%m-%d %H:%M')
                ax.text(ds_hurricane.lon.isel(time=j).values + 0.3, 
                       ds_hurricane.lat.isel(time=j).values + 0.3,
                       time_str, fontsize=8, fontweight='bold',
                       transform=ccrs.PlateCarree(), zorder=60,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Labels and title
    #ax.set_xlabel('Longitude', fontsize=12)
    #ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar next to each subplot
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, 
                       label='Wind Speed (m/s)', ticks=levels_wind[::4])
    cbar.ax.tick_params(labelsize=9)
    
    return cf, q

# Extract wind data for each case
def extract_wind_data(ds, u_var, v_var, time_idx, time_dim='wind_time'):
    """Extract u and v wind components at a specific time index"""
    u = ds[u_var].isel({time_dim: time_idx}).squeeze().values
    v = ds[v_var].isel({time_dim: time_idx}).squeeze().values
    return u, v

# Helene with hurricane
u_helene_hurricane, v_helene_hurricane = extract_wind_data(
    ds_wind_hurricane, u_var, v_var, helene_idx_hurricane, time_dim)
wind_speed_helene_hurricane = np.sqrt(u_helene_hurricane**2 + v_helene_hurricane**2)

# Helene climatology
u_helene_clim, v_helene_clim = extract_wind_data(
    ds_wind_climatology, u_var, v_var, helene_idx_climatology, time_dim)
wind_speed_helene_clim = np.sqrt(u_helene_clim**2 + v_helene_clim**2)

# Milton with hurricane
u_milton_hurricane, v_milton_hurricane = extract_wind_data(
    ds_wind_hurricane, u_var, v_var, milton_idx_hurricane, time_dim)
wind_speed_milton_hurricane = np.sqrt(u_milton_hurricane**2 + v_milton_hurricane**2)

# Milton climatology
u_milton_clim, v_milton_clim = extract_wind_data(
    ds_wind_climatology, u_var, v_var, milton_idx_climatology, time_dim)
wind_speed_milton_clim = np.sqrt(u_milton_clim**2 + v_milton_clim**2)

# Plot top row: Helene
cf1, q1 = plot_wind_field(axes[0], lon, lat, u_helene_hurricane, v_helene_hurricane,
                     wind_speed_helene_hurricane,
                     f'(a) Helene-Hurricane ({helene_date.strftime("%m-%d")})',
                     ds_h_all, helene_target_idx)

# cf2, q2 = plot_wind_field(axes[1], lon, lat, u_helene_clim, v_helene_clim,
#                      wind_speed_helene_clim,
#                      f'Helene-Climatology ({helene_date.strftime("%m-%d")})',
#                      ds_h_all, helene_target_idx)

cf2, q2 = plot_wind_field(axes[1], lon, lat, u_helene_clim, v_helene_clim,
                     wind_speed_helene_clim,
                     f'(b) Helene-Climatology ({helene_date.strftime("%m-%d")})')  #no hurricane tracks


# Plot bottom row: Milton
cf3, q3 = plot_wind_field(axes[2], lon, lat, u_milton_hurricane, v_milton_hurricane,
                     wind_speed_milton_hurricane,
                     f'(c) Milton-Hurricane ({milton_date.strftime("%m-%d")})',
                     ds_m_all, milton_target_idx)

cf4, q4 = plot_wind_field(axes[3], lon, lat, u_milton_clim, v_milton_clim,
                     wind_speed_milton_clim,
                     f'(d) Milton-Climatology ({milton_date.strftime("%m-%d")})',
                     ds_m_all, milton_target_idx)

outpath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/wind'
# plt.savefig(f'{outpath}/wind_field_comparison.png', 
#             dpi=300, bbox_inches='tight')
# plt.close() 