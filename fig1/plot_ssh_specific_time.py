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
        0: 'lightgray',     # Not a hurricane
        1: 'white',     # Category 1
        2: 'skyblue',       # Category 2
        3: 'blue',          # Category 3
        4: 'navy',      # Category 4
        5: 'black'           # Category 5
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
# filepath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/neither'
# filepath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nohelene'
# filepath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nomilton'

outpath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/original/ssh/both'
# outpath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/original/ssh/neither'
# outpath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/original/ssh/nohelene'
# outpath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/original/ssh/nomilton'

ds = xr.open_dataset(f'{filepath}/sst.nc')
lon_rho = ds.lon_rho.values
lat_rho = ds.lat_rho.values

xi_rho_ind = ds.xi_rho.values[(lon_rho[0,:]>-86) & (lon_rho[0,:]<-84)]
eta_rho_ind = ds.eta_rho.values[np.argmin(np.abs(lat_rho[:,0]-25))]

sst = np.load(f'{filepath}/sst.npz')['sst']
ssh = np.load(f'{filepath}/ssh.npz')['ssh']

with np.load(f'{filepath}/current_0.npz') as surface:
    surfu = surface['surfu']
    surfv = surface['surfv']
    speed0 = surface['speed']
    
with np.load(f'{filepath}/current_1500.npz') as subsurf:
    u1500 = subsurf['u1500']
    v1500 = subsurf['v1500']
    speed1500 = subsurf['speed_1500']

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

#%%----------------- Plot SSH for specific time with both hurricane tracks
extent = [-94, -81, 20, 31]
levels_sst = np.arange(-0.6, 0.60001, 0.02)

# Find the specific time index for 2025-09-24
target_date = pd.to_datetime('2025-09-24')
time_indices = []
for i, time_val in enumerate(ds.ocean_time):
    model_time = pd.to_datetime(time_val.values)
    if model_time.date() == target_date.date():
        time_indices.append(i)

if not time_indices:
    print(f"No data found for {target_date.strftime('%Y-%m-%d')}")
    print("Available dates:")
    for i, time_val in enumerate(ds.ocean_time):
        model_time = pd.to_datetime(time_val.values)
        print(f"  {model_time.strftime('%Y-%m-%d')}")
    exit()

# Use the first matching time index (assuming 12:00)
i = time_indices[0]
model_time = pd.to_datetime(ds.ocean_time[i].values)

print(f"Plotting SSH for {model_time.strftime('%Y-%m-%d %H:%M:%S')}")

fig, ax = plt.subplots(1, 1, figsize=(12, 10), 
                      subplot_kw={'projection': ccrs.Mercator()})

# Create map
cplt.create(extent, ax=ax, proj=ccrs.Mercator(), gridlines=False, bathymetry=True)

# Plot SSH
cf = ax.contourf(lon_rho, lat_rho, ssh[i], 
                levels=levels_sst, extend='both', 
                transform=ccrs.PlateCarree(), 
                cmap=cmocean.cm.balance)
cs2 = ax.contour(lon_rho, lat_rho, ssh[i], [-0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.3, 0.4], colors='k', linewidths=0.5, transform=ccrs.PlateCarree()) 
cs3 = ax.contour(lon_rho, lat_rho, ssh[i], [0.4], colors='k', linewidths=2, transform=ccrs.PlateCarree())         

# Plot section
ax.plot(lon_rho[eta_rho_ind, xi_rho_ind[0]:xi_rho_ind[-1]+1], 
        lat_rho[eta_rho_ind, xi_rho_ind[0]:xi_rho_ind[-1]+1],
        'r-', transform=ccrs.PlateCarree(), linewidth=2, label='Section') 

# Plot Helene track
ax.plot(ds_h_all.lon, ds_h_all.lat, 'b-', linewidth=2, 
       transform=ccrs.PlateCarree(), alpha=0.8, label='Helene Track')

# Plot Milton track  
ax.plot(ds_m_all.lon, ds_m_all.lat, 'r--', linewidth=2, 
       transform=ccrs.PlateCarree(), alpha=0.8, label='Milton Track')

# Add legend for tracks
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

# Labels and title
ax.set_xlabel('Longitude', fontsize=16)
ax.set_ylabel('Latitude', fontsize=16)
ax.set_title(f'CNAPS2 SSH - {model_time.strftime("%Y-%m-%d")} T12:00:00\nwith Hurricane Tracks', 
            fontsize=20)

# Colorbar
cbar = plt.colorbar(cf, ax=ax, label='SSH (m)', ticks=levels_sst[::5])
cbar.ax.tick_params(labelsize=12)

# Save figure
plt.savefig(f'{outpath}/ssh_{model_time.strftime("%Y_%m_%d")}_both_tracks.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("Plot completed!")
