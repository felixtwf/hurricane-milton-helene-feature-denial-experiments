# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def calculate_z_rho(h, zeta, s_rho, Cs_r, hc, Vtransform):
    """
    Calculate depth at rho points from ROMS sigma coordinates.

    Parameters:
    h: bathymetry depth (positive, meters)
    zeta: sea surface height (meters)
    s_rho: sigma coordinate at rho points
    Cs_r: S-coordinate stretching curves at rho points
    hc: critical depth (meters)
    Vtransform: vertical transformation type (1 or 2)

    Returns:
    z_rho: depth at rho points (negative below surface, meters)
    """
    if Vtransform == 1:
        # Original ROMS vertical coordinate
        z_rho = hc * (s_rho - Cs_r) + Cs_r * h
        if zeta is not None:
            z_rho = z_rho + zeta * (1 + z_rho / h)
    elif Vtransform == 2:
        # New ROMS vertical coordinate (more common)
        z_rho = (hc * s_rho + Cs_r * h) / (hc + h)
        if zeta is not None:
            z_rho = zeta + (zeta + h) * z_rho
    else:
        raise ValueError(f"Vtransform must be 1 or 2, got {Vtransform}")

    return z_rho


#--------------------------------------Define all cases to process
cases = {
    'both': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/EXP_ref_sep23/output/useast_avg.nc',
    'neither': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/Exp_rm_both/output/useast_avg.nc',
    'nohelene': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/Exp_rm_helene/output/useast_avg.nc',
    'nomilton': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/Exp_rm_milton/output/useast_avg.nc'
}

outpaths = {
    'both': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/fig_sect/data/both',
    'neither': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/fig_sect/data/neither',
    'nohelene': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/fig_sect/data/nohelene',
    'nomilton': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/figs/fig_sect/data/nomilton'
}

# Load reference dataset once for vertical coordinate parameters
print("Loading reference dataset for vertical coordinate parameters...")
ds_ref = xr.open_dataset('/projects/water/oomg/twu/ENKF/case_93to19_mercator/Output/ENKF_AVG/useast_avg_2019.nc',
                         drop_variables=['dstart'], chunks={'ocean_time':1})
ds_ref = ds_ref.isel(ocean_time=0)

# Extract vertical coordinate parameters from reference
Vtransform_ref = int(ds_ref.Vtransform.values)
Cs_r_ref = ds_ref.Cs_r.values
hc_ref = float(ds_ref.hc.values)
s_rho_ref = ds_ref.s_rho.values

# Close reference dataset
ds_ref.close()
del ds_ref
gc.collect()

print(f"Reference vertical coordinate system: Vtransform={Vtransform_ref}, hc={hc_ref}m")

# Loop through all cases
for case_name, filepath_DA in cases.items():
    print(f"\n{'='*60}")
    print(f"Processing case: {case_name}")
    print(f"{'='*60}")

    outpath = outpaths[case_name]

    # Create output directory if it doesn't exist
    os.makedirs(outpath, exist_ok=True)

    # Open dataset with lazy loading
    ds_DA = xr.open_dataset(filepath_DA, drop_variables=['dstart'], chunks={'ocean_time':1})
    # ds_DA = ds_DA.sel(ocean_time=slice('2024-09-20', '2024-10-15'))
    # ds_DA = ds_DA.sel(ocean_time=slice(None, '2024-10-15'))

    # Load grid coordinates (small memory footprint)
    lon_rho = ds_DA.lon_rho.values
    lat_rho = ds_DA.lat_rho.values
    lon_u = ds_DA.lon_u.values
    lat_u = ds_DA.lat_u.values
    lon_v = ds_DA.lon_v.values
    lat_v = ds_DA.lat_v.values

    # Load bathymetry from main dataset
    h = ds_DA.h.values  # Bottom depth at rho points

    # Use vertical coordinate parameters from reference dataset
    s_rho = s_rho_ref
    Cs_r = Cs_r_ref
    hc = hc_ref
    Vtransform = Vtransform_ref

    #--------------extract diagonal CE section (25N, 86W) to (24N, 85W)
    # Define section endpoints
    lon_start, lat_start = -86.0, 26.0
    lon_end, lat_end = -84.0, 24.0

    # Create points along the diagonal section
    n_points = 50  # Number of points along the section
    lon_section = np.linspace(lon_start, lon_end, n_points)
    lat_section = np.linspace(lat_start, lat_end, n_points)

    # Find nearest grid indices for each point along the section
    from scipy.spatial import cKDTree

    # Build KDTree for rho points (only once per case)
    print("Building spatial indices...")
    points_rho = np.column_stack([lon_rho.ravel(), lat_rho.ravel()])
    tree_rho = cKDTree(points_rho)

    # Build KDTree for u points
    points_u = np.column_stack([lon_u.ravel(), lat_u.ravel()])
    tree_u = cKDTree(points_u)

    # Build KDTree for v points
    points_v = np.column_stack([lon_v.ravel(), lat_v.ravel()])
    tree_v = cKDTree(points_v)

    # Query nearest points along section
    section_points = np.column_stack([lon_section, lat_section])
    dist_rho, idx_rho = tree_rho.query(section_points)
    dist_u, idx_u = tree_u.query(section_points)
    dist_v, idx_v = tree_v.query(section_points)

    # Convert flat indices to 2D indices
    eta_rho_idx, xi_rho_idx = np.unravel_index(idx_rho, lon_rho.shape)
    eta_u_idx, xi_u_idx = np.unravel_index(idx_u, lon_u.shape)
    eta_v_idx, xi_v_idx = np.unravel_index(idx_v, lon_v.shape)

    # Clean up trees
    del tree_rho, tree_u, tree_v, points_rho, points_u, points_v

    # Extract data along the section (keeping data lazy)
    print("Extracting temperature section...")
    # For temperature (on rho grid)
    temp_sec_list = []
    h_sec_rho = []
    for i in range(len(eta_rho_idx)):
        temp_sec_list.append(ds_DA.temp.isel(eta_rho=eta_rho_idx[i], xi_rho=xi_rho_idx[i]))
        h_sec_rho.append(h[eta_rho_idx[i], xi_rho_idx[i]])
    temp_sec = xr.concat(temp_sec_list, dim='section_point')
    h_sec_rho = np.array(h_sec_rho)

    print("Extracting u-velocity section...")
    # For u velocity (on u grid with dimensions: eta_u, xi_u)
    uvel_sec_list = []
    h_sec_u = []
    for i in range(len(eta_u_idx)):
        uvel_sec_list.append(ds_DA.u.isel(eta_u=eta_u_idx[i], xi_u=xi_u_idx[i]))
        # Get bathymetry at u points (need to load h on u grid or interpolate)
        # For simplicity, use nearest rho point bathymetry
        h_sec_u.append(h[eta_u_idx[i], xi_u_idx[i]])
    uvel_sec = xr.concat(uvel_sec_list, dim='section_point')
    h_sec_u = np.array(h_sec_u)

    print("Extracting v-velocity section...")
    # For v velocity (on v grid with dimensions: eta_v, xi_v)
    vvel_sec_list = []
    h_sec_v = []
    for i in range(len(eta_v_idx)):
        vvel_sec_list.append(ds_DA.v.isel(eta_v=eta_v_idx[i], xi_v=xi_v_idx[i]))
        # Get bathymetry at v points
        h_sec_v.append(h[eta_v_idx[i], xi_v_idx[i]])
    vvel_sec = xr.concat(vvel_sec_list, dim='section_point')
    h_sec_v = np.array(h_sec_v)

    print("Calculating static depth coordinates (no SSH variation)...")
    # Calculate depth at each section point using bathymetry only (zeta=0)
    # This assumes depth doesn't vary with time

    # Calculate depth for rho points (temperature)
    z_rho_sec = np.zeros((len(s_rho), len(eta_rho_idx)))
    for i in range(len(eta_rho_idx)):
        z_rho_sec[:, i] = calculate_z_rho(h_sec_rho[i], 0.0, s_rho, Cs_r, hc, Vtransform)

    # Calculate depth for u points (u-velocity)
    z_u_sec = np.zeros((len(s_rho), len(eta_u_idx)))
    for i in range(len(eta_u_idx)):
        z_u_sec[:, i] = calculate_z_rho(h_sec_u[i], 0.0, s_rho, Cs_r, hc, Vtransform)

    # Calculate depth for v points (v-velocity)
    z_v_sec = np.zeros((len(s_rho), len(eta_v_idx)))
    for i in range(len(eta_v_idx)):
        z_v_sec[:, i] = calculate_z_rho(h_sec_v[i], 0.0, s_rho, Cs_r, hc, Vtransform)

    # Add section coordinates as new coordinate variables
    # Temperature file gets full depth info
    temp_sec = temp_sec.assign_coords(section_lon=('section_point', lon_section))
    temp_sec = temp_sec.assign_coords(section_lat=('section_point', lat_section))
    temp_sec = temp_sec.assign_coords(section_h=('section_point', h_sec_rho))
    temp_sec = temp_sec.assign_coords(section_z=(['s_rho', 'section_point'], z_rho_sec))

    # U-velocity file gets depth info for u points
    uvel_sec = uvel_sec.assign_coords(section_lon=('section_point', lon_section))
    uvel_sec = uvel_sec.assign_coords(section_lat=('section_point', lat_section))
    uvel_sec = uvel_sec.assign_coords(section_h=('section_point', h_sec_u))
    uvel_sec = uvel_sec.assign_coords(section_z=(['s_rho', 'section_point'], z_u_sec))

    # V-velocity file gets depth info for v points
    vvel_sec = vvel_sec.assign_coords(section_lon=('section_point', lon_section))
    vvel_sec = vvel_sec.assign_coords(section_lat=('section_point', lat_section))
    vvel_sec = vvel_sec.assign_coords(section_h=('section_point', h_sec_v))
    vvel_sec = vvel_sec.assign_coords(section_z=(['s_rho', 'section_point'], z_v_sec))

    # Save sectional data only (this triggers computation with chunked processing)
    print("Saving temperature section...")
    temp_sec.to_netcdf(f'{outpath}/temp_sec.nc')
    print("Saving u-velocity section...")
    uvel_sec.to_netcdf(f'{outpath}/uvel_sec.nc')
    print("Saving v-velocity section...")
    vvel_sec.to_netcdf(f'{outpath}/vvel_sec.nc')

    print(f"Extracted diagonal section from ({lat_start}N, {lon_start}W) to ({lat_end}N, {lon_end}W)")
    print(f"Number of points along section: {n_points}")
    print(f"Saved files to: {outpath}")

    # Clean up to free memory
    ds_DA.close()
    del ds_DA, temp_sec, uvel_sec, vvel_sec
    del temp_sec_list, uvel_sec_list, vvel_sec_list
    del h_sec_rho, h_sec_u, h_sec_v
    del z_rho_sec, z_u_sec, z_v_sec
    gc.collect()

print("\n" + "="*60)
print("All cases processed successfully!")
print("="*60)


#-----------test plot section
# extent = [-94, -81, 20, 31]
# levels_sst = np.arange(26, 32.01, 0.2)

# fig, ax = plt.subplots(1, 1, figsize=(12, 10), 
                      # subplot_kw={'projection': ccrs.Mercator()})

# # Create map
# cplt.create(extent, ax=ax, proj=ccrs.Mercator(), gridlines=False, bathymetry=True)
# cf = ax.contourf(lon_rho, lat_rho, ds_DA_surftemp[0], 
                # levels=levels_sst, extend='both', 
                # transform=ccrs.PlateCarree(), 
                # cmap=cmocean.cm.thermal)

# ax.plot(lon_rho[eta_rho_ind, xi_rho_ind[0]:xi_rho_ind[-1]+1], 
        # lat_rho[eta_rho_ind, xi_rho_ind[0]:xi_rho_ind[-1]+1],
        # 'r-', transform=ccrs.PlateCarree())                

# # Labels and title
# ax.set_xlabel('Longitude', fontsize=16)
# ax.set_ylabel('Latitude', fontsize=16)


# # Colorbar
# cbar = plt.colorbar(cf, ax=ax, label='Temperature (°C)', ticks=levels_sst[::5])
# cbar.ax.tick_params(labelsize=12)

# # Save figure
# plt.savefig(f'./sst_section.png', 
            # dpi=300, bbox_inches='tight')

