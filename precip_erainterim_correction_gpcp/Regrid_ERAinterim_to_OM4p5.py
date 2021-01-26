#!/usr/bin/env python

import xarray as xr
import xesmf
import numpy as np
import warnings
import cftime


erainterim_indir = '/lustre/f2/dev/Raphael.Dussin/forcings/ERAinterim/'
erainterim_outdir = '/lustre/f2/dev/Raphael.Dussin/forcings/ERAinterim_OM4p5'
firstyear=1980
lastyear=2018


## Building the grids

# ERA-interim
erai_grid = xr.open_dataset(f'{erainterim_indir}/precip_ERAinterim_1979_daily_ROMS.nc', 
                            decode_times=False, 
                            drop_variables=['rain', 'rain_time'])

lon = erai_grid['lon'].values
lon_bnds = np.concatenate((np.array([lon[0] -0.5 * 0.7031]),
                           0.5 * (lon[:-1] + lon[1:]),
                           np.array([lon[-1] + 0.5 * 0.7031])), axis=0)

lat = erai_grid['lat'].values
lat_bnds = np.concatenate((np.array([-90]),
                           0.5 * (lat[:-1] + lat[1:]),
                           np.array([90])), axis=0)

erai_grid['lon_b'] = xr.DataArray(data=lon_bnds, dims=('lonp1'))
erai_grid['lat_b'] = xr.DataArray(data=lat_bnds, dims=('latp1'))

# OM4p5
dirgrid='/lustre/f2/pdata/gfdl/gfdl_O/datasets/OM4_05/mosaic_ocean.v20180227.unpacked'
om4_supergrid = xr.open_dataset(f'{dirgrid}/ocean_hgrid.nc')

om4_grid = xr.Dataset()
lon = om4_supergrid['x'].values[1::2,1::2].copy()
lat = om4_supergrid['y'].values[1::2,1::2].copy()
lon_b = om4_supergrid['x'].values[0::2,0::2].copy()
lat_b = om4_supergrid['y'].values[0::2,0::2].copy()
angle = om4_supergrid['angle_dx'].values[1::2,1::2].copy()

om4_grid['lon'] = xr.DataArray(data=lon, dims=('y', 'x'))
om4_grid['lat'] = xr.DataArray(data=lat, dims=('y', 'x'))
om4_grid['angle'] = xr.DataArray(data=np.deg2rad(angle), dims=('y', 'x'))
om4_grid['lon_b'] = xr.DataArray(data=lon_b, dims=('yp1', 'xp1'))
om4_grid['lat_b'] = xr.DataArray(data=lat_b, dims=('yp1', 'xp1'))


regrid_conserve = xesmf.Regridder(erai_grid, om4_grid,
                                  method='conservative',
                                  periodic=True, reuse_weights=True)

regrid_patch = xesmf.Regridder(erai_grid, om4_grid,
                               method='patch',
                               periodic=True, reuse_weights=True)

regrid_bilin = xesmf.Regridder(erai_grid, om4_grid,
                               method='bilinear',
                               periodic=True, reuse_weights=True)


## Functions

def expand_time_serie(ds, timevar):
    """ add first and last bogus time slices """
    first_slice = ds.isel({timevar:0})
    second_slice = ds.isel({timevar:1})
    last_slice = ds.isel({timevar:-1})
    dt = second_slice[timevar] - first_slice[timevar]
    prologue = first_slice.copy(deep=True)
    prologue[timevar] = first_slice[timevar] - dt
    epilogue = last_slice.copy(deep=True)
    epilogue[timevar] = last_slice[timevar] + dt
    ds_expanded = xr.concat([prologue, ds, epilogue], dim=timevar)
    return ds_expanded


def interp_and_save_year(ds, var, time, year, outputdir, method='conservative'):
    """ slice year with one day before and after, regrid and save to netcdf """
    start = f'{year-1}-12-31'
    end = f'{year+1}-01-01'
    data = ds.rename({time: 'time'}).sel(time=slice(start,end))
    data = data.chunk({'time': -1}).transpose(*('time', 'lat', 'lon'))
    if method == 'conservative':
        regridded = regrid_conserve(data)
    elif method == 'patch':
        regridded = regrid_patch(data)
    encoding = {'time':{'units': 'days since 1900-01-01T0:00:00',
                        'calendar': 'gregorian'},
                'lon': {'_FillValue': 1e+20},
                'lat': {'_FillValue': 1e+20},
                var: {'_FillValue': 1e+20},
               }
    regridded['lon'].attrs = {'axis': 'X', 'units': 'degrees_east', 
                              'long_name': 'Longitude', 
                              'standard_name': 'longitude'}
    regridded['lat'].attrs = {'axis': 'Y', 'units': 'degrees_north', 
                              'long_name': 'Latitude', 
                              'standard_name': 'latitude'}
    regridded['time'].attrs = {'axis': 'T'}
    regridded.to_netcdf(f'{outputdir}/{var}_ERAinterim_{year}_OM4p5_{method}.nc', 
                        format='NETCDF3_64BIT',
                        encoding=encoding, 
                        unlimited_dims=['time'])
    return None


## Regrid precips and radiative fluxes

precip = xr.open_mfdataset(f'{erainterim_indir}/precip_ERAinterim_*_daily_ROMS.nc', combine='by_coords')
precip_expanded = expand_time_serie(precip, 'rain_time')
precip_expanded = precip_expanded.rename({'rain': 'precip'})
for year in range(firstyear,lastyear+1):
    interp_and_save_year(precip_expanded,'precip','rain_time', year, erainterim_outdir)


snow = xr.open_mfdataset(f'{erainterim_indir}/snow_ERAinterim_*_daily_ROMS.nc', combine='by_coords')
snow = snow.rename({'rain': 'snow'})
snow_expanded = expand_time_serie(snow, 'rain_time')
for year in range(firstyear,lastyear+1):
    interp_and_save_year(snow_expanded,'snow','rain_time', year, erainterim_outdir)


radlw = xr.open_mfdataset(f'{erainterim_indir}/radlw_ERAinterim_*_daily_ROMS.nc', combine='by_coords')
radlw_expanded = expand_time_serie(radlw, 'lrf_time')
for year in range(firstyear,lastyear+1):
    interp_and_save_year(radlw_expanded,'lwrad_down','lrf_time', year, erainterim_outdir)


radsw = xr.open_mfdataset(f'{erainterim_indir}/radsw_ERAinterim_*_daily_ROMS.nc', combine='by_coords')
radsw_expanded = expand_time_serie(radsw, 'srf_time')
for year in range(firstyear,lastyear+1):
    interp_and_save_year(radsw_expanded,'swrad','srf_time', year, erainterim_outdir)


## Regrid turb variables

tair = xr.open_mfdataset(f'{erainterim_indir}/t2_ERAinterim_*_ROMS.nc',
                         combine='by_coords')
tair_expanded = expand_time_serie(tair, 'tair_time')
tair_expanded['Tair'] = tair_expanded['Tair'] + 273.15
for year in range(firstyear,lastyear+1):
    print("work on tair", year)
    interp_and_save_year(tair_expanded,'Tair','tair_time',
                         year, erainterim_outdir,
                         method='patch')


qair = xr.open_mfdataset(f'{erainterim_indir}/q2_ERAinterim_*_ROMS.nc',
                         combine='by_coords')
qair_expanded = expand_time_serie(qair, 'qair_time')
for year in range(firstyear,lastyear+1):
    print("work on qair", year)
    interp_and_save_year(qair_expanded,'Qair','qair_time',
                         year, erainterim_outdir,
                         method='patch')


pair = xr.open_mfdataset(f'{erainterim_indir}/msl_ERAinterim_*_ROMS.nc',
                         combine='by_coords')
pair_expanded = expand_time_serie(pair, 'pair_time')
for year in range(firstyear,lastyear+1):
    print("work on pair", year)
    interp_and_save_year(pair_expanded,'Pair','pair_time',
                         year, erainterim_outdir,
                         method='patch')


uwind = xr.open_mfdataset(f'{erainterim_indir}/u10_ERAinterim_*_ROMS.nc',
                          combine='by_coords')
uwind_expanded = expand_time_serie(uwind, 'wind_time')
for year in range(firstyear,lastyear+1):
    print("work on uwind", year)
    interp_and_save_year(uwind_expanded,'Uwind','wind_time',
                         year, erainterim_outdir,
                         method='patch')


vwind = xr.open_mfdataset(f'{erainterim_indir}/v10_ERAinterim_*_ROMS.nc',
                          combine='by_coords')
vwind_expanded = expand_time_serie(vwind, 'wind_time')
for year in range(firstyear,lastyear+1):
    print("work on vwind", year)
    interp_and_save_year(vwind_expanded,'Vwind','wind_time',
                         year, erainterim_outdir,
                         method='patch')

## rotate the velocities

for year in range(firstyear,lastyear+1):
    print("rotating winds year =", year)
    uraw = xr.open_dataset(f'{erainterim_outdir}/Uwind_ERAinterim_{year}_OM4p5_patch.nc')['Uwind']
    vraw = xr.open_dataset(f'{erainterim_outdir}/Vwind_ERAinterim_{year}_OM4p5_patch.nc')['Vwind']
    urot = uraw * np.cos(om4_grid['angle']) + vraw * np.sin(om4_grid['angle'])
    vrot = vraw * np.cos(om4_grid['angle']) - uraw * np.sin(om4_grid['angle'])
 
    encoding_u = {'time':{'units': 'days since 1900-01-01T0:00:00',
                          'calendar': 'gregorian'},
                  'lon': {'_FillValue': 1e+20},
                  'lat': {'_FillValue': 1e+20},
                  'Uwind': {'_FillValue': 1e+20},
               }
    encoding_v = {'time':{'units': 'days since 1900-01-01T0:00:00',
                          'calendar': 'gregorian'},
                  'lon': {'_FillValue': 1e+20},
                  'lat': {'_FillValue': 1e+20},
                  'Vwind': {'_FillValue': 1e+20},
               }

    UwindR = urot.to_dataset(name='Uwind')
    VwindR = vrot.to_dataset(name='Vwind')

    UwindR.to_netcdf(f'{erainterim_outdir}/UwindR_ERAinterim_{year}_OM4p5_patch.nc', 
                        format='NETCDF3_64BIT',
                        encoding=encoding_u, 
                        unlimited_dims=['time'])
    VwindR.to_netcdf(f'{erainterim_outdir}/VwindR_ERAinterim_{year}_OM4p5_patch.nc', 
                        format='NETCDF3_64BIT',
                        encoding=encoding_v, 
                        unlimited_dims=['time'])
