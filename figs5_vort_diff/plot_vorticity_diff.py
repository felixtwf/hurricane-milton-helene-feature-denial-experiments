import numpy as np
import xarray as xr
import cmocean
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

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

def plot_hurricane_track(ax, model_time, ds_h_all, ds_m_all):
    """
    Plot hurricane track and add legend based on current model time.
    
    Parameters:
    -----------
    ax : matplotlib.axes
        Axes object to plot on
    model_time : pandas.Timestamp
        Current model time
    ds_h_all : xarray.Dataset
        Helene hurricane dataset
    ds_m_all : xarray.Dataset
        Milton hurricane dataset
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
    
    # Add legend for hurricane categories
    legend_elements = []
    for cat in range(1, 6):  # Categories 1-5
        color = get_category_color(cat)
        label = get_category_label(cat)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=6, 
                                        markeredgecolor='black', label=label, linestyle='None'))
    
    # Add legend positioned to avoid being beneath land
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                      bbox_to_anchor=(0.99, 0.99))
    # Ensure legend appears above other elements
    legend.set_zorder(10)

#%%----------------- read CNAPS2    

filepath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/both'
filepath_neither = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/neither'
filepath_nohelene = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nohelene'
filepath_nomilton = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nomilton'

outpath_neither = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/diff/vorticity/neither'
outpath_nohelene = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/diff/vorticity/nohelene'
outpath_nomilton = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/diff/vorticity/nomilton'

ds_ref = xr.open_dataset(f'{filepath}/sst.nc')
lon_rho = ds_ref.lon_rho.values
lat_rho = ds_ref.lat_rho.values

xi_rho_ind = ds_ref.xi_rho.values[(lon_rho[0,:]>-87) & (lon_rho[0,:]<-85)]
eta_rho_ind = ds_ref.eta_rho.values[np.argmin(np.abs(lat_rho[:,0]-25))]

with np.load(f'{filepath}/current_0.npz') as ref_surf:
    surfu_ref = ref_surf['surfu']
    surfv_ref = ref_surf['surfv']
    speed0_ref = ref_surf['speed']

with np.load(f'{filepath_neither}/current_0.npz') as neither_surf:
    surfu_neither = neither_surf['surfu']
    surfv_neither = neither_surf['surfv']
    speed0_neither = neither_surf['speed']
    
with np.load(f'{filepath_nohelene}/current_0.npz') as nohelene_surf:
    surfu_nohelene = nohelene_surf['surfu']
    surfv_nohelene = nohelene_surf['surfv']
    speed0_nohelene = nohelene_surf['speed']
    
with np.load(f'{filepath_nomilton}/current_0.npz') as nomilton_surf:
    surfu_nomilton = nomilton_surf['surfu']
    surfv_nomilton = nomilton_surf['surfv']
    speed0_nomilton = nomilton_surf['speed']

# Calculate velocity differences
surfu_diff_neither = surfu_ref - surfu_neither
surfv_diff_neither = surfv_ref - surfv_neither

surfu_diff_nohelene = surfu_ref - surfu_nohelene
surfv_diff_nohelene = surfv_ref - surfv_nohelene

surfu_diff_nomilton = surfu_ref - surfu_nomilton
surfv_diff_nomilton = surfv_ref - surfv_nomilton

#%%----------------- read hurricane track
ds_helene = xr.open_dataset('../hurricanes/helene.nc')
ds_milton = xr.open_dataset('../hurricanes/milton.nc')

# Select the period during which hurricane is stronger than cat1
storm_types = ['SD', 'SS', 'TD', 'TS', 'HU']
ds_h_sub = ds_helene.where(ds_helene.type.isin(storm_types), drop=True)
ds_m_sub = ds_milton.where(ds_milton.type.isin(storm_types), drop=True)

# Use all hurricane data (every 6 hours)
ds_h_all = ds_h_sub
ds_m_all = ds_m_sub

print(f"Helene total timestamps: {len(ds_h_all.time)}")
print(f"Milton total timestamps: {len(ds_m_all.time)}")

#%%----------------- Plot diff (Ref-neither)
extent = [-94, -81, 20, 31]
# Dimensionless vorticity difference levels (ζ/f) - adjust based on your data range
levels_vort = np.arange(-0.2, 0.21, 0.02)  # Dimensionless
dd = 8; mm = 12

for i in range(len(ds_ref.ocean_time)):
    print(f"Processing timestep {i+1}/{len(ds_ref.ocean_time)}")
    
    # Calculate vorticity differences
    vort_ref = calculate_vorticity(surfu_ref[i], surfv_ref[i], lon_rho, lat_rho)
    vort_neither = calculate_vorticity(surfu_neither[i], surfv_neither[i], lon_rho, lat_rho)
    vort_diff_neither = vort_ref - vort_neither
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Create map using the new function
    plot_spatial_map(ax)
    
    # Plot vorticity difference
    cf = ax.contourf(lon_rho, lat_rho, vort_diff_neither,
                    levels=levels_vort, extend='both',
                    transform=ccrs.PlateCarree(),
                    cmap=cmocean.cm.balance, zorder=0)
    
    # Overlay velocity difference arrows
    lon0 = lon_rho[::dd,::mm]
    lat0 = lat_rho[::dd,::mm]
    u0 = surfu_diff_neither[i, ::dd,::mm]
    v0 = surfv_diff_neither[i, ::dd,::mm]
    # q = ax.quiver(lon0, lat0, u0, v0,
                  # color='g', scale=6, width=0.005,
                  # transform=ccrs.PlateCarree(), zorder=4)
    # ax.quiverkey(q, X=0.96, Y=-0.08, U=0.5,
             # label='0.5 m/s',
             # labelpos='S',
             # coordinates='axes',
             # color='k',
             # fontproperties={'size': 10, 'weight': 'bold'})
        
    # Get current model time
    model_time = pd.to_datetime(ds_ref.ocean_time[i].values)
    
    # Plot hurricane track and legend
    plot_hurricane_track(ax, model_time, ds_h_all, ds_m_all)
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Latitude', fontsize=16)
    ax.set_title(f'Vorticity Diff. (Control-Neither) - {model_time.strftime("%Y-%m-%d")}',
                fontsize=15)

    # Colorbar
    axpos = ax.get_position()
    cax = fig.add_axes([axpos.x1+.008, axpos.y0, 0.015, axpos.height])
    cbar = fig.colorbar(cf, cax=cax, ticks=levels_vort[::2], label='Δ(ζ/f)')
    cbar.ax.tick_params(labelsize=12)
    
    # Save figure
    plt.savefig(f'{outpath_neither}/vorticity_diff_{model_time.strftime("%Y_%m_%d")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

print("All plots completed for Ref-Neither case!")

#%%----------------- Plot diff (Ref-nohelene)
for i in range(len(ds_ref.ocean_time)):
    print(f"Processing timestep {i+1}/{len(ds_ref.ocean_time)}")
    
    # Calculate vorticity differences
    vort_ref = calculate_vorticity(surfu_ref[i], surfv_ref[i], lon_rho, lat_rho)
    vort_nohelene = calculate_vorticity(surfu_nohelene[i], surfv_nohelene[i], lon_rho, lat_rho)
    vort_diff_nohelene = vort_ref - vort_nohelene
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Create map using the new function
    plot_spatial_map(ax)
    
    # Plot vorticity difference
    cf = ax.contourf(lon_rho, lat_rho, vort_diff_nohelene,
                    levels=levels_vort, extend='both',
                    transform=ccrs.PlateCarree(),
                    cmap=cmocean.cm.balance, zorder=0)
    
    # Overlay velocity difference arrows
    lon0 = lon_rho[::dd,::mm]
    lat0 = lat_rho[::dd,::mm]
    u0 = surfu_diff_nohelene[i, ::dd,::mm]
    v0 = surfv_diff_nohelene[i, ::dd,::mm]
    # q = ax.quiver(lon0, lat0, u0, v0,
                  # color='g', scale=6, width=0.005,
                  # transform=ccrs.PlateCarree(), zorder=4)
    # ax.quiverkey(q, X=0.96, Y=-0.08, U=0.5,
             # label='0.5 m/s',
             # labelpos='S',
             # coordinates='axes',
             # color='k',
             # fontproperties={'size': 10, 'weight': 'bold'})
        
    # Get current model time
    model_time = pd.to_datetime(ds_ref.ocean_time[i].values)
    
    # Plot hurricane track and legend
    plot_hurricane_track(ax, model_time, ds_h_all, ds_m_all)
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Latitude', fontsize=16)
    ax.set_title(f'Vorticity Diff. (Control-No Helene) - {model_time.strftime("%Y-%m-%d")}',
                fontsize=15)

    # Colorbar
    axpos = ax.get_position()
    cax = fig.add_axes([axpos.x1+.008, axpos.y0, 0.015, axpos.height])
    cbar = fig.colorbar(cf, cax=cax, ticks=levels_vort[::2], label='Δ(ζ/f)')
    cbar.ax.tick_params(labelsize=12)
    
    # Save figure
    plt.savefig(f'{outpath_nohelene}/vorticity_diff_{model_time.strftime("%Y_%m_%d")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

print("All plots completed for Ref-nohelene case!")

#%%----------------- Plot diff (Ref-nomilton)
for i in range(len(ds_ref.ocean_time)):
    print(f"Processing timestep {i+1}/{len(ds_ref.ocean_time)}")
    
    # Calculate vorticity differences
    vort_ref = calculate_vorticity(surfu_ref[i], surfv_ref[i], lon_rho, lat_rho)
    vort_nomilton = calculate_vorticity(surfu_nomilton[i], surfv_nomilton[i], lon_rho, lat_rho)
    vort_diff_nomilton = vort_ref - vort_nomilton
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Create map using the new function
    plot_spatial_map(ax)
    
    # Plot vorticity difference
    cf = ax.contourf(lon_rho, lat_rho, vort_diff_nomilton,
                    levels=levels_vort, extend='both',
                    transform=ccrs.PlateCarree(),
                    cmap=cmocean.cm.balance, zorder=0)
    
    # Overlay velocity difference arrows
    lon0 = lon_rho[::dd,::mm]
    lat0 = lat_rho[::dd,::mm]
    u0 = surfu_diff_nomilton[i, ::dd,::mm]
    v0 = surfv_diff_nomilton[i, ::dd,::mm]
    # q = ax.quiver(lon0, lat0, u0, v0,
                  # color='g', scale=6, width=0.005,
                  # transform=ccrs.PlateCarree(), zorder=4)
    # ax.quiverkey(q, X=0.96, Y=-0.08, U=0.5,
             # label='0.5 m/s',
             # labelpos='S',
             # coordinates='axes',
             # color='k',
             # fontproperties={'size': 10, 'weight': 'bold'})
             
             
    # Get current model time
    model_time = pd.to_datetime(ds_ref.ocean_time[i].values)
    
    # Plot hurricane track and legend
    plot_hurricane_track(ax, model_time, ds_h_all, ds_m_all)
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Latitude', fontsize=16)
    ax.set_title(f'Vorticity Diff. (Control-No Milton) - {model_time.strftime("%Y-%m-%d")}',
                fontsize=15)

    # Colorbar
    axpos = ax.get_position()
    cax = fig.add_axes([axpos.x1+.008, axpos.y0, 0.015, axpos.height])
    cbar = fig.colorbar(cf, cax=cax, ticks=levels_vort[::2], label='Δ(ζ/f)')
    cbar.ax.tick_params(labelsize=12)
    
    # Save figure
    plt.savefig(f'{outpath_nomilton}/vorticity_diff_{model_time.strftime("%Y_%m_%d")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

print("All plots completed for Ref-nomilton case!")
print("All plots completed for all cases!")
