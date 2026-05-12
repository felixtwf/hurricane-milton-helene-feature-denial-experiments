import os
import numpy as np
import xarray as xr
import cmocean
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd


# Hurricane category definitions based on wind speed (mph)
def get_hurricane_category(vmax):
    if vmax < 64:
        return 0
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
    colors = {
        0: 'blue',
        1: 'yellow',
        2: 'orange',
        3: 'red',
        4: 'magenta',
        5: 'purple',
    }
    return colors.get(category, 'lightgray')


def get_category_label(category):
    labels = {
        0: 'Non-Hurricane',
        1: 'Cat 1',
        2: 'Cat 2',
        3: 'Cat 3',
        4: 'Cat 4',
        5: 'Cat 5',
    }
    return labels.get(category, 'Unknown')


def setup_spatial_map(ax, show_xlabels=True, show_ylabels=True):
    gulf_extent = [-94, -81, 20, 31]
    ax.set_extent(gulf_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='gray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5,
                      color='grey', alpha=0.6, zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_ylabels
    gl.bottom_labels = show_xlabels
    gl.xlabel_style = {'fontsize': 13}
    gl.ylabel_style = {'fontsize': 13}
    gl.xlocator = plt.FixedLocator([-92, -88, -84])
    gl.ylocator = plt.FixedLocator([22, 26, 30])
    return ax


# ---------------- Experiments (4 columns) ----------------
cases = [
    {
        'name': 'Control',
        'filepath': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/both',
    },
    {
        'name': 'No Hurricane',
        'filepath': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/neither',
    },
    {
        'name': 'No Helene',
        'filepath': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nohelene',
    },
    {
        'name': 'No Milton',
        'filepath': '/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/exp_forcing/processed/nomilton',
    },
]

outpath_combined = '.'
os.makedirs(outpath_combined, exist_ok=True)

# ---------------- Load all experiment data ----------------
print("Loading data for all experiments...")
all_data = {}
for case in cases:
    print(f"  Loading {case['name']} from {case['filepath']}")
    ds = xr.open_dataset(f"{case['filepath']}/sst.nc")
    ssh = np.load(f"{case['filepath']}/ssh.npz")['ssh']
    with np.load(f"{case['filepath']}/current_0.npz") as surface:
        surfu = surface['surfu']
        surfv = surface['surfv']
        speed0 = surface['speed']
    all_data[case['name']] = {
        'lon': ds.lon_rho.values,
        'lat': ds.lat_rho.values,
        'time': pd.to_datetime(ds.ocean_time.values),
        'ssh': ssh,
        'surfu': surfu,
        'surfv': surfv,
        'speed0': speed0,
    }
    ds.close()

# ---------------- Hurricane tracks ----------------
ds_helene = xr.open_dataset('/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/hurricanes/helene.nc')
ds_milton = xr.open_dataset('/projects/water/oomg/tangf/CNAPS2/cnaps2_hurricane/hurricanes/milton.nc')
storm_types = ['SD', 'SS', 'TD', 'TS', 'HU']
ds_h_all = ds_helene.where(ds_helene.type.isin(storm_types), drop=True)
ds_m_all = ds_milton.where(ds_milton.type.isin(storm_types), drop=True)

helene_start = pd.to_datetime(ds_h_all.time.min().values)
helene_end = pd.to_datetime(ds_h_all.time.max().values)
milton_start = pd.to_datetime(ds_m_all.time.min().values)
milton_end = pd.to_datetime(ds_m_all.time.max().values)

# ---------------- Target dates: 10/10 - 10/16 ----------------
sample_year = all_data[cases[0]['name']]['time'][0].year
target_dates = pd.date_range(f'{sample_year}-10-10', f'{sample_year}-10-16', freq='D')

n_rows = len(target_dates)
n_cols = len(cases)

# ---------------- Build combined figure ----------------
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.2 * n_cols, 3.2 * n_rows),
    subplot_kw={'projection': ccrs.PlateCarree()},
)

levels_sst = np.arange(-0.6, 0.60001, 0.02)
SPEED_15KT = 0.77  # 1.5 knots ~ 0.7716 m/s
cf = None

for row_idx, target_date in enumerate(target_dates):
    for col_idx, case in enumerate(cases):
        ax = axes[row_idx, col_idx]
        data = all_data[case['name']]

        # Find matching time index by date
        t_idx = None
        for j, t in enumerate(data['time']):
            if t.date() == target_date.date():
                t_idx = j
                break
        if t_idx is None:
            ax.set_visible(False)
            continue

        show_xlabels = (row_idx == n_rows - 1)
        show_ylabels = (col_idx == 0)
        setup_spatial_map(ax, show_xlabels, show_ylabels)

        lon_rho = data['lon']
        lat_rho = data['lat']
        ssh = data['ssh']
        speed0 = data['speed0']
        # surfu = data['surfu']
        # surfv = data['surfv']

        # SSH shading (dimmed via alpha)
        cf = ax.contourf(
            lon_rho, lat_rho, ssh[t_idx],
            levels=levels_sst, extend='both',
            transform=ccrs.PlateCarree(),
            cmap=cmocean.cm.balance, alpha=0.45, zorder=0,
        )
        # SSH contours (dimmed: gray + thin + low alpha)
        ax.contour(
            lon_rho, lat_rho, ssh[t_idx],
            [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
            colors='gray', linewidths=0.4, alpha=0.5,
            transform=ccrs.PlateCarree(), zorder=3,
        )
        # 0.4 m SSH contour (dimmed)
        ax.contour(
            lon_rho, lat_rho, ssh[t_idx], [0.4],
            colors='black', linewidths=1.2, alpha=0.9,
            transform=ccrs.PlateCarree(), zorder=3,
        )
        # 1.5-knot surface-current contour (highlighted)
        ax.contour(
            lon_rho, lat_rho, speed0[t_idx], [SPEED_15KT],
            colors='red', linewidths=2.8,
            transform=ccrs.PlateCarree(), zorder=5,
        )

        # # Surface-current arrows (commented out for v2)
        # skip = 8
        # q = ax.quiver(
        #     lon_rho[::skip, ::skip], lat_rho[::skip, ::skip],
        #     surfu[t_idx, ::skip, ::skip], surfv[t_idx, ::skip, ::skip],
        #     transform=ccrs.PlateCarree(),
        #     scale=20, width=0.003, color='g', zorder=4,
        # )
        # ax.quiverkey(q, X=0.1, Y=0.96, U=1, label='1 m/s',
        #              labelpos='S', coordinates='axes', color='k',
        #              fontproperties={'size': 12, 'weight': 'bold'})

        model_time = pd.to_datetime(data['time'][t_idx])

        # Hurricane track (same storm shown for all 4 columns at a given row)
        if helene_start <= model_time <= helene_end:
            ds_hurricane = ds_h_all
        elif milton_start <= model_time <= milton_end:
            ds_hurricane = ds_m_all
        else:
            ds_hurricane = None

        if ds_hurricane is not None:
            ax.plot(
                ds_hurricane.lon, ds_hurricane.lat,
                'k--', linewidth=1.0,
                transform=ccrs.PlateCarree(), zorder=2,
            )
            for j in range(len(ds_hurricane.time)):
                point_time = pd.to_datetime(ds_hurricane.time.isel(time=j).values)
                if point_time > model_time:
                    continue
                vmax = ds_hurricane.vmax.isel(time=j).values
                category = get_hurricane_category(vmax)
                color = get_category_color(category)
                is_current_time = (
                    point_time.date() == model_time.date() and point_time.hour == 12
                )
                edge_color = 'k' if is_current_time else 'none'
                line_width = 1.5 if is_current_time else 0
                ax.scatter(
                    ds_hurricane.lon.isel(time=j),
                    ds_hurricane.lat.isel(time=j),
                    c=color, s=45, marker='o',
                    edgecolors=edge_color, linewidth=line_width,
                    transform=ccrs.PlateCarree(), zorder=6,
                )

        # Column title (top row only)
        if row_idx == 0:
            ax.set_title(case['name'], fontsize=20, fontweight='bold', pad=8)

        # Row label = date (left column only)
        if col_idx == 0:
            ax.text(
                -0.22, 0.5, model_time.strftime('%m/%d'),
                transform=ax.transAxes,
                fontsize=20, fontweight='bold',
                rotation=90, va='center', ha='center',
            )

# ---------------- Tight layout, then add shared legend & colorbar ----------------
fig.tight_layout(rect=[0.015, 0.045, 0.94, 0.985], h_pad=0.6, w_pad=0.4)

# Shared colorbar on the right
cbar_ax = fig.add_axes([0.952, 0.10, 0.011, 0.82])
cbar = fig.colorbar(cf, cax=cbar_ax, ticks=levels_sst[::5])
cbar.set_label('SSH (m)', fontsize=18)
cbar.ax.tick_params(labelsize=14)

# Shared legend at the bottom (categories + 1.5-knot line)
legend_elements = []
for cat in range(1, 6):
    legend_elements.append(plt.Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor=get_category_color(cat), markersize=11,
        markeredgecolor='black', label=get_category_label(cat),
        linestyle='None',
    ))
legend_elements.append(plt.Line2D(
    [0], [0], color='red', linewidth=2.8, label='1.5-knot surface current',
))
legend_elements.append(plt.Line2D(
    [0], [0], color='dimgray', linewidth=1.2, label='SSH 0.4 m',
))
fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=len(legend_elements),
    fontsize=15,
    bbox_to_anchor=(0.5, 0.005),
    frameon=True,
)

outfile = f'{outpath_combined}/ssh_uv_v2_{target_dates[0].strftime("%Y%m%d")}_{target_dates[-1].strftime("%Y%m%d")}.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved combined figure: {outfile}")
