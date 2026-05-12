# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:42:34 2025

@author: ydeng23
"""

import numpy as np
import xarray as xr
import cmocean as cmo
import matplotlib.pyplot as plt
import cool_maps.plot as cplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd



#%%----------------- read CNAPS2 topo    

grid = xr.open_dataset('C:/Ocean/CNAPSV2/useast_grd5_2_cnapsv2.nc')
lon_rho = grid['lon_rho'].values
lat_rho = grid['lat_rho'].values
lon_u = grid['lon_u'].values
lat_u = grid['lat_u'].values
lon_v = grid['lon_v'].values
lat_v = grid['lat_v'].values
depth = grid['h'].values




#%%----------------- Plot SSH for specific time with both hurricane tracks
# extent = [-94, -81, 20, 31]
extent = [-98, -75, 15, 35]
# levels_sst = np.arange(-0.6, 0.60001, 0.02)
ssh_levels = np.arange(-5000,0.1,50)

fig, ax = plt.subplots(1, 1, figsize=(12, 10), 
                      subplot_kw={'projection': ccrs.Mercator()})

# Create map
cplt.create(extent, ax=ax, proj=ccrs.Mercator(), gridlines=False, bathymetry=False)
cf = ax.contourf(lon_rho, lat_rho, -depth, levels=ssh_levels, cmap=cmo.cm.deep_r, extend='both', zorder=-5, transform=ccrs.PlateCarree())

# Labels and title
ax.set_xlabel('Longitude', fontsize=16)
ax.set_ylabel('Latitude', fontsize=16)


# Colorbar
# cbar = plt.colorbar(cf, ax=ax, label='SSH (m)', ticks=levels_sst[::5])
# cbar.ax.tick_params(labelsize=12)

# Save figure
plt.savefig('gom_topo_small.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("Plot completed!")


