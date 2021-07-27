#!/usr/bin/env python

# # Making forcing for OM4p5

import xarray as xr
import xesmf
import numpy as np
import warnings
import cftime


gpcp_indir = '/lustre/f2/dev/Raphael.Dussin/forcings/GPCP'
gpcp_outdir = '/lustre/f2/dev/Raphael.Dussin/forcings/ERAinterim_OM4p5'
firstyear=1979
lastyear=2018


# ## Building the grids

# GPCP
gpcp_grid = xr.open_dataset(f'{gpcp_indir}/precip.mon.mean.nc',
                            decode_times=False, 
                            drop_variables=['precip', 'time'])

lon_bnds = np.concatenate([gpcp_grid['lon_bnds'].isel(nv=0).values, 
                           np.array([gpcp_grid['lon_bnds'].isel(nv=1).values[-1]])], axis=0)

lat_bnds = np.concatenate([gpcp_grid['lat_bnds'].isel(nv=0).values, 
                           np.array([gpcp_grid['lat_bnds'].isel(nv=1).values[-1]])], axis=0)

gpcp_grid['lon_b'] = xr.DataArray(data=lon_bnds, dims=('lonp1'))
gpcp_grid['lat_b'] = xr.DataArray(data=lat_bnds, dims=('latp1'))

# OM4p5
dirgrid='/lustre/f2/pdata/gfdl/gfdl_O/datasets/OM4_05/mosaic_ocean.v20180227.unpacked'
om4_supergrid = xr.open_dataset(f'{dirgrid}/ocean_hgrid.nc')

om4_grid = xr.Dataset()
lon = om4_supergrid['x'].values[1::2,1::2].copy()
lat = om4_supergrid['y'].values[1::2,1::2].copy()
lon_b = om4_supergrid['x'].values[0::2,0::2].copy()
lat_b = om4_supergrid['y'].values[0::2,0::2].copy()

om4_grid['lon'] = xr.DataArray(data=lon, dims=('y', 'x'))
om4_grid['lat'] = xr.DataArray(data=lat, dims=('y', 'x'))
om4_grid['lon_b'] = xr.DataArray(data=lon_b, dims=('yp1', 'xp1'))
om4_grid['lat_b'] = xr.DataArray(data=lat_b, dims=('yp1', 'xp1'))


regrid_conserve = xesmf.Regridder(gpcp_grid, om4_grid,
                                  method='conservative',
                                  periodic=True, reuse_weights=True)



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
    ds_expanded = ds_expanded.resample(time="1D").interpolate("linear")
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
                        'calendar': 'gregorian', 'dtype': np.dtype('f8')},
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
    regridded.to_netcdf(f'{outputdir}/{var}_GPCPv23_{year}_OM4p5_{method}.nc', 
                        format='NETCDF3_64BIT',
                        encoding=encoding, 
                        unlimited_dims=['time'])
    return None


## Regrid precips


precip = xr.open_dataset(f'{gpcp_indir}/precip.mon.mean.nc',
                         drop_variables=['lon_bnds', 'lat_bnds'])
precip_expanded = expand_time_serie(precip, 'time')
precip_expanded['precip'] = precip_expanded['precip'] / 86400.
for year in range(firstyear,lastyear+1):
    interp_and_save_year(precip_expanded,'precip','time', year, gpcp_outdir)

