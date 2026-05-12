import numpy as np
import xarray as xr
import cmocean
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from geopy.distance import geodesic

#%%----------------- read CNAPS2

filepath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/fig_sect/data/both'
filepath_neither = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/fig_sect/data/neither'
filepath_nohelene = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/fig_sect/data/nohelene'
filepath_nomilton = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/fig_sect/data/nomilton'

outpath = '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/fig_sect/plots_12C'

import os
os.makedirs(outpath, exist_ok=True)


ds_temp_ref = xr.open_dataset(f'{filepath}/temp_sec.nc')
ds_uvel_ref = xr.open_dataset(f'{filepath}/uvel_sec.nc')
ds_vvel_ref = xr.open_dataset(f'{filepath}/vvel_sec.nc')

ds_temp_neither = xr.open_dataset(f'{filepath_neither}/temp_sec.nc')
ds_temp_nohelene = xr.open_dataset(f'{filepath_nohelene}/temp_sec.nc')
ds_temp_nomilton = xr.open_dataset(f'{filepath_nomilton}/temp_sec.nc')

# Calculate distance along section using the new coordinates
from geopy.distance import geodesic

def calculate_section_distance(lons, lats):
    """Calculate cumulative distance along section in km"""
    distances = np.zeros(len(lons))
    for i in range(1, len(lons)):
        dist = geodesic((lats[i-1], lons[i-1]), (lats[i], lons[i])).km
        distances[i] = distances[i-1] + dist
    return distances

# Get distance for reference (all cases should have same section)
section_lon = ds_temp_ref.section_lon.values
section_lat = ds_temp_ref.section_lat.values
distance = calculate_section_distance(section_lon, section_lat)
lon_start, lat_start = abs(section_lon.min()), abs(section_lat.max())
lon_end, lat_end = abs(section_lon.max()), abs(section_lat.min())
    
    
levels_vel = np.arange(-1, 1.01, .02)
levels_temp = np.arange(5, 30.01, 0.2)


for i in range(len(ds_temp_ref.ocean_time)):
    print(f"Processing timestep {i+1}/{len(ds_temp_ref.ocean_time)}")
    model_time = pd.to_datetime(ds_temp_ref.ocean_time[i].values)

    #plot temperature
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))

    # Create 2D meshgrid for plotting: distance vs depth
    # section_z has shape (s_rho, section_point)
    distance_2d = np.tile(distance, (len(ds_temp_ref.s_rho), 1))

    # Optional: plot filled contours as background
    # cf = ax.contourf(distance_2d, ds_temp_ref.section_z.values, ds_temp_ref.temp[i].values,
    #                  levels=levels_temp, extend='both', cmap=cmocean.cm.thermal, alpha=0.3)

    # Plot isotherms for all cases
    # Reference case (Control)
    cs1 = ax.contour(distance_2d, ds_temp_ref.section_z.values, ds_temp_ref.temp.isel(ocean_time=i).values.T,
                     [12], colors='k', linewidths=2, label='Control')
    #ax.clabel(cs1, inline=True, fontsize=10, fmt='%g°C')

    # Neither hurricane case
    cs2 = ax.contour(distance_2d, ds_temp_neither.section_z.values, ds_temp_neither.temp.isel(ocean_time=i).values.T,
                     [12], colors='r', linewidths=2, linestyles='--')

    # No Helene case
    cs3 = ax.contour(distance_2d, ds_temp_nohelene.section_z.values, ds_temp_nohelene.temp.isel(ocean_time=i).values.T,
                     [12], colors='b', linewidths=2, linestyles=':')

    # No Milton case
    cs4 = ax.contour(distance_2d, ds_temp_nomilton.section_z.values, ds_temp_nomilton.temp.isel(ocean_time=i).values.T,
                     [12], colors='m', linewidths=2, linestyles='-.')

    # Labels and title
    ax.set_xlabel('Distance along section (km)', fontsize=14)
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    ax.set_title(f'12 °C Isotherms - {model_time.strftime("%Y-%m-%d")} T12:00:00\n' +
                f'Section: {lat_start}N, {lon_start}W to {lat_end}N, {lon_end}W', fontsize=14)
    ax.set_ylim(-600, -200)
    ax.set_xlim(distance[0], distance[-1])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Create custom legend handles
    legend_elements = [
        Line2D([0], [0], color='k', lw=2, label='Control', linestyle='-'),
        Line2D([0], [0], color='r', lw=2, label='No hurricane', linestyle='--'),
        Line2D([0], [0], color='b', lw=2, label='No Helene', linestyle=':'),
        Line2D([0], [0], color='m', lw=2, label='No Milton', linestyle='-.')
    ]

    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    # Save figure
    plt.savefig(f'{outpath}/temp_section_{model_time.strftime("%Y_%m_%d")}.png',
                dpi=300, bbox_inches='tight')
    plt.close()  

    # #plot uvel
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # cf = ax.contourf(np.tile(ds_uvel_ref.lon_u, (50,1)), ds_uvel_ref.z_rho_u[i], ds_uvel_ref.u[i], levels=levels_vel, extend='both', cmap=cmocean.cm.balance)
    
    # ax.set_xlabel('Longitude', fontsize=16)
    # ax.set_ylabel('Depth', fontsize=16)
    # ax.set_title(f'U 25N - {model_time.strftime("%Y-%m-%d")} T12:00:00', 
                # fontsize=20)
    # ax.set_ylim(-1500,0)
    
    # cbar = plt.colorbar(cf, ax=ax, label='Velocity (m/s)', ticks=levels_vel[::10])
    # cbar.ax.tick_params(labelsize=12)
    
    # plt.savefig(f'{outpath}/uvel_1500_{model_time.strftime("%Y_%m_%d")}.png', 
                # dpi=300, bbox_inches='tight')
                

    # #plot vvel
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # cf = ax.contourf(np.tile(ds_vvel_ref.lon_v, (50,1)), ds_vvel_ref.z_rho_v[i], ds_vvel_ref.v[i], levels=levels_vel, extend='both', cmap=cmocean.cm.balance)
    
    # ax.set_xlabel('Longitude', fontsize=16)
    # ax.set_ylabel('Depth', fontsize=16)
    # ax.set_title(f'V 25N - {model_time.strftime("%Y-%m-%d")} T12:00:00', 
                # fontsize=20)
    # ax.set_ylim(-1500,0)
    
    # cbar = plt.colorbar(cf, ax=ax, label='Velocity (m/s)', ticks=levels_vel[::10])
    # cbar.ax.tick_params(labelsize=12)
    
    # plt.savefig(f'{outpath}/vvel_1500_{model_time.strftime("%Y_%m_%d")}.png', 
                # dpi=300, bbox_inches='tight')
                