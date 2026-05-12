import numpy as np
import xarray as xr
import cmocean
import matplotlib.pyplot as plt
import cool_maps.plot as cplt
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

#%%----------------- read CNAPS2    

filepath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/both'
filepath_neither = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/neither'
filepath_nohelene = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nohelene'
filepath_nomilton = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nomilton'




outpath_neither = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/diff/sst/res'
outpath_nohelene = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/diff/sst/sum'



ds_ref = xr.open_dataset(f'{filepath}/sst.nc')
lon_rho = ds_ref.lon_rho.values
lat_rho = ds_ref.lat_rho.values

xi_rho_ind = ds_ref.xi_rho.values[(lon_rho[0,:]>-86) & (lon_rho[0,:]<-84)]
eta_rho_ind = ds_ref.eta_rho.values[np.argmin(np.abs(lat_rho[:,0]-25))]


sst_ref = np.load(f'{filepath}/sst.npz')['sst']
ssh_ref = np.load(f'{filepath}/ssh.npz')['ssh']

sst_neither = np.load(f'{filepath_neither}/sst.npz')['sst']
ssh_neither = np.load(f'{filepath_neither}/ssh.npz')['ssh']

sst_nohelene = np.load(f'{filepath_nohelene}/sst.npz')['sst']
ssh_nohelene = np.load(f'{filepath_nohelene}/ssh.npz')['ssh']

sst_nomilton = np.load(f'{filepath_nomilton}/sst.npz')['sst']
ssh_nomilton = np.load(f'{filepath_nomilton}/ssh.npz')['ssh']

sst_diff_neither = sst_ref - sst_neither
ssh_diff_neither = ssh_ref - ssh_neither

sst_diff_nohelene = sst_ref - sst_nohelene
ssh_diff_nohelene = ssh_ref - ssh_nohelene

sst_diff_nomilton = sst_ref - sst_nomilton
ssh_diff_nomilton = ssh_ref - ssh_nomilton



# with np.load(f'{filepath}/current_0.npz') as surface:
    # surfu = surface['surfu']
    # surfv = surface['surfv']
    # speed0 = surface['speed']
    
# with np.load(f'{filepath}/current_1500.npz') as subsurf:
    # u1500 = subsurf['u1500']
    # v1500 = subsurf['v1500']
    # speed1500 = subsurf['speed_1500']

#%%----------------- read hurricane track
ds_helene = xr.open_dataset('./hurricanes/helene.nc')
ds_milton = xr.open_dataset('./hurricanes/milton.nc')

# Select the period during which hurricane is stronger than cat1
storm_types = ['SD', 'SS', 'TD', 'TS', 'HU']
ds_h_sub = ds_helene.where(ds_helene.type.isin(storm_types), drop=True)
ds_m_sub = ds_milton.where(ds_milton.type.isin(storm_types), drop=True)

# Use all hurricane data (every 6 hours)
ds_h_all = ds_h_sub
ds_m_all = ds_m_sub

print(f"Helene total timestamps: {len(ds_h_all.time)}")
print(f"Milton total timestamps: {len(ds_m_all.time)}")

#%%----------------- Plot OHC with hurricane tracks
extent = [-94, -81, 20, 31]
levels_sst = np.arange(-0.2, 0.201, 0.02)
dd = 5; mm = 10


#-------------------residual: sst_diff_neither[i] - sst_diff_nohelene[i] - sst_diff_nomilton[i]
for i in range(len(ds_ref.ocean_time)):
    print(f"Processing timestep {i+1}/{len(ds_ref.ocean_time)}")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), 
                          subplot_kw={'projection': ccrs.Mercator()})
    
    # Create map
    cplt.create(extent, ax=ax, proj=ccrs.Mercator(), gridlines=False, bathymetry=True)
    
    # Plot sst
    cf = ax.contourf(lon_rho, lat_rho, sst_diff_neither[i] - sst_diff_nohelene[i] - sst_diff_nomilton[i], 
                    levels=levels_sst, extend='both', 
                    transform=ccrs.PlateCarree(), 
                    cmap=cmocean.cm.balance)
    
    # plot section
    # if i == 0:
        # ax.plot(lon_rho[eta_rho_ind, xi_rho_ind[0]:xi_rho_ind[-1]+1], 
                # lat_rho[eta_rho_ind, xi_rho_ind[0]:xi_rho_ind[-1]+1],
                # 'r-', transform=ccrs.PlateCarree()) 
        
    # Get current model time
    model_time = pd.to_datetime(ds_ref.ocean_time[i].values)
    
    # Determine which hurricane to show based on model time
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
        line_style = '--'
        
        # Plot track line
        ax.plot(ds_hurricane.lon, ds_hurricane.lat, 'k--', linewidth=1, 
               transform=ccrs.PlateCarree(), label=f'{hurricane_name} Track', zorder=49)
        
        # Plot points with category colors
        for j, time_point in enumerate(ds_hurricane.time):
            point_time = pd.to_datetime(time_point.values)
            vmax = ds_hurricane.vmax.isel(time=j).values
            category = get_hurricane_category(vmax)
            color = get_category_color(category)
            
            # Check if this timestamp matches the current model time (12:00)
            is_current_time = (point_time.date() == model_time.date() and point_time.hour == 12)
            edge_color = 'k' if is_current_time else 'none'
            line_width = 2 if is_current_time else 0
            
            # Plot point
            ax.scatter(ds_hurricane.lon.isel(time=j), ds_hurricane.lat.isel(time=j),
                      c=color, s=50, marker=marker_shape, edgecolors=edge_color, linewidth=line_width,
                      transform=ccrs.PlateCarree(), zorder=50)
            
            # # Add category label if it's a hurricane (category > 0) and current time
            # if category > 0 and is_current_time:
                # ax.text(ds_hurricane.lon.isel(time=j) + 0.2, ds_hurricane.lat.isel(time=j) + 0.2,
                       # get_category_label(category), fontsize=8, fontweight='bold',
                       # transform=ccrs.PlateCarree(), zorder=60)
    
    elif milton_start <= model_time <= milton_end:
        # Plot Milton track
        hurricane_name = "Milton"
        ds_hurricane = ds_m_all
        marker_shape = 'o'
        line_style = '--'
        
        # Plot track line
        ax.plot(ds_hurricane.lon, ds_hurricane.lat, 'k--', linewidth=1, 
               transform=ccrs.PlateCarree(), label=f'{hurricane_name} Track', zorder=49)
        
        # Plot points with category colors
        for j, time_point in enumerate(ds_hurricane.time):
            point_time = pd.to_datetime(time_point.values)
            vmax = ds_hurricane.vmax.isel(time=j).values
            category = get_hurricane_category(vmax)
            color = get_category_color(category)
            
            # Check if this timestamp matches the current model time (12:00)
            is_current_time = (point_time.date() == model_time.date() and point_time.hour == 12)
            edge_color = 'k' if is_current_time else 'none'
            line_width = 2 if is_current_time else 0
            
            # Plot point
            ax.scatter(ds_hurricane.lon.isel(time=j), ds_hurricane.lat.isel(time=j),
                      c=color, s=50, marker=marker_shape, edgecolors=edge_color, linewidth=line_width,
                      transform=ccrs.PlateCarree(), zorder=50)
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Latitude', fontsize=16)
    ax.set_title(f'SST (Res) - {model_time.strftime("%Y-%m-%d")} T12', 
                fontsize=20)
    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, label='Temperature (°C)', ticks=levels_sst[::5])
    cbar.ax.tick_params(labelsize=12)
    
    # Save figure
    plt.savefig(f'{outpath_neither}/sst_res_{model_time.strftime("%Y_%m_%d")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

print("All plots completed!") 





levels_sst = np.arange(-2, 2.01, 0.2)
#------------------ sum: sst_diff_nohelene[i] + sst_diff_nomilton[i]

for i in range(len(ds_ref.ocean_time)):
    print(f"Processing timestep {i+1}/{len(ds_ref.ocean_time)}")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), 
                          subplot_kw={'projection': ccrs.Mercator()})
    
    # Create map
    cplt.create(extent, ax=ax, proj=ccrs.Mercator(), gridlines=False, bathymetry=True)
    
    # Plot sst
    cf = ax.contourf(lon_rho, lat_rho, sst_diff_nohelene[i] + sst_diff_nomilton[i], 
                    levels=levels_sst, extend='both', 
                    transform=ccrs.PlateCarree(), 
                    cmap=cmocean.cm.balance)
    
    # plot section
    # if i == 0:
        # ax.plot(lon_rho[eta_rho_ind, xi_rho_ind[0]:xi_rho_ind[-1]+1], 
                # lat_rho[eta_rho_ind, xi_rho_ind[0]:xi_rho_ind[-1]+1],
                # 'r-', transform=ccrs.PlateCarree()) 
        
    # Get current model time
    model_time = pd.to_datetime(ds_ref.ocean_time[i].values)
    
    # Determine which hurricane to show based on model time
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
        line_style = '--'
        
        # Plot track line
        ax.plot(ds_hurricane.lon, ds_hurricane.lat, 'k--', linewidth=1, 
               transform=ccrs.PlateCarree(), label=f'{hurricane_name} Track', zorder=49)
        
        # Plot points with category colors
        for j, time_point in enumerate(ds_hurricane.time):
            point_time = pd.to_datetime(time_point.values)
            vmax = ds_hurricane.vmax.isel(time=j).values
            category = get_hurricane_category(vmax)
            color = get_category_color(category)
            
            # Check if this timestamp matches the current model time (12:00)
            is_current_time = (point_time.date() == model_time.date() and point_time.hour == 12)
            edge_color = 'k' if is_current_time else 'none'
            line_width = 2 if is_current_time else 0
            
            # Plot point
            ax.scatter(ds_hurricane.lon.isel(time=j), ds_hurricane.lat.isel(time=j),
                      c=color, s=50, marker=marker_shape, edgecolors=edge_color, linewidth=line_width,
                      transform=ccrs.PlateCarree(), zorder=50)
            
            # # Add category label if it's a hurricane (category > 0) and current time
            # if category > 0 and is_current_time:
                # ax.text(ds_hurricane.lon.isel(time=j) + 0.2, ds_hurricane.lat.isel(time=j) + 0.2,
                       # get_category_label(category), fontsize=8, fontweight='bold',
                       # transform=ccrs.PlateCarree(), zorder=60)
    
    elif milton_start <= model_time <= milton_end:
        # Plot Milton track
        hurricane_name = "Milton"
        ds_hurricane = ds_m_all
        marker_shape = 'o'
        line_style = '--'
        
        # Plot track line
        ax.plot(ds_hurricane.lon, ds_hurricane.lat, 'k--', linewidth=1, 
               transform=ccrs.PlateCarree(), label=f'{hurricane_name} Track', zorder=49)
        
        # Plot points with category colors
        for j, time_point in enumerate(ds_hurricane.time):
            point_time = pd.to_datetime(time_point.values)
            vmax = ds_hurricane.vmax.isel(time=j).values
            category = get_hurricane_category(vmax)
            color = get_category_color(category)
            
            # Check if this timestamp matches the current model time (12:00)
            is_current_time = (point_time.date() == model_time.date() and point_time.hour == 12)
            edge_color = 'k' if is_current_time else 'none'
            line_width = 2 if is_current_time else 0
            
            # Plot point
            ax.scatter(ds_hurricane.lon.isel(time=j), ds_hurricane.lat.isel(time=j),
                      c=color, s=50, marker=marker_shape, edgecolors=edge_color, linewidth=line_width,
                      transform=ccrs.PlateCarree(), zorder=50)
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Latitude', fontsize=16)
    ax.set_title(f'SST (Helene+Milton) - {model_time.strftime("%Y-%m-%d")} T12', 
                fontsize=20)
    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, label='Temperature (°C)', ticks=levels_sst[::5])
    cbar.ax.tick_params(labelsize=12)
    
    # Save figure
    plt.savefig(f'{outpath_nohelene}/sst_sum_{model_time.strftime("%Y_%m_%d")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

print("All plots completed!") 




