#!/usr/bin/env python

import xarray as xr
import xesmf
import numpy as np
import warnings
import cftime


erainterim_indir = '/lustre/f2/dev/Raphael.Dussin/forcings/ERAinterim/'
erainterim_outdir = '/lustre/f2/dev/Raphael.Dussin/forcings/ERAinterim_padded/'
firstyear=1979
lastyear=2018

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


def save_year(ds, var, time, year, outputdir):
    """ slice year with one day before and after and save to netcdf """
    start = f'{year-1}-12-31'
    end = f'{year+1}-01-01'
    data = ds.rename({time: 'time'}).sel(time=slice(start,end))
    data = data.chunk({'time': -1}).transpose(*('time', 'lat', 'lon'))

    encoding = {'time':{'units': 'days since 1900-01-01T0:00:00',
                        'calendar': 'gregorian'},
                'lon': {'_FillValue': 1e+20},
                'lat': {'_FillValue': 1e+20},
                var: {'_FillValue': 1e+20},
               }
    data['lon'].attrs = {'axis': 'X', 'units': 'degrees_east', 
                         'long_name': 'Longitude', 
                         'standard_name': 'longitude'}
    data['lat'].attrs = {'axis': 'Y', 'units': 'degrees_north', 
                         'long_name': 'Latitude', 
                         'standard_name': 'latitude'}
    data['time'].attrs = {'axis': 'T'}

    data.to_netcdf(f'{outputdir}/{var}_ERAinterim_{year}_padded.nc', 
                        format='NETCDF3_64BIT',
                        encoding=encoding, 
                        unlimited_dims=['time'])
    return None


## pad precips and radiative fluxes

precip = xr.open_mfdataset(f'{erainterim_indir}/precip_ERAinterim_*_daily_ROMS.nc', combine='by_coords')
snow   = xr.open_mfdataset(f'{erainterim_indir}/snow_ERAinterim_*_daily_ROMS.nc', combine='by_coords')
precip_expanded = expand_time_serie(precip, 'rain_time')
snow_expanded   = expand_time_serie(snow, 'rain_time')
# create liquid precip
rain_expanded = precip_expanded - snow_expanded
rain_expanded['rain'] = rain_expanded['rain'].clip(min=0)

snow_expanded = snow_expanded.rename({'rain': 'snow'})

for year in range(firstyear,lastyear+1):
    save_year(rain_expanded,'rain','rain_time', year, erainterim_outdir)
    save_year(snow_expanded,'snow','rain_time', year, erainterim_outdir)


radlw = xr.open_mfdataset(f'{erainterim_indir}/radlw_ERAinterim_*_daily_ROMS.nc', combine='by_coords')
radlw_expanded = expand_time_serie(radlw, 'lrf_time')
for year in range(firstyear,lastyear+1):
    save_year(radlw_expanded,'lwrad_down','lrf_time', year, erainterim_outdir)


radsw = xr.open_mfdataset(f'{erainterim_indir}/radsw_ERAinterim_*_daily_ROMS.nc', combine='by_coords')
radsw_expanded = expand_time_serie(radsw, 'srf_time')
for year in range(firstyear,lastyear+1):
    save_year(radsw_expanded,'swrad','srf_time', year, erainterim_outdir)


## pad turb variables

tair = xr.open_mfdataset(f'{erainterim_indir}/t2_ERAinterim_*_ROMS.nc',
                         combine='by_coords')
tair_expanded = expand_time_serie(tair, 'tair_time')
tair_expanded['Tair'] = tair_expanded['Tair'] + 273.15
for year in range(firstyear,lastyear+1):
    print("work on tair", year)
    save_year(tair_expanded,'Tair','tair_time',
              year, erainterim_outdir)


qair = xr.open_mfdataset(f'{erainterim_indir}/q2_ERAinterim_*_ROMS.nc',
                         combine='by_coords')
qair_expanded = expand_time_serie(qair, 'qair_time')
for year in range(firstyear,lastyear+1):
    print("work on qair", year)
    save_year(qair_expanded,'Qair','qair_time',
              year, erainterim_outdir)


pair = xr.open_mfdataset(f'{erainterim_indir}/msl_ERAinterim_*_ROMS.nc',
                         combine='by_coords')
pair_expanded = expand_time_serie(pair, 'pair_time')
for year in range(firstyear,lastyear+1):
    print("work on pair", year)
    save_year(pair_expanded,'Pair','pair_time',
              year, erainterim_outdir)


uwind = xr.open_mfdataset(f'{erainterim_indir}/u10_ERAinterim_*_ROMS.nc',
                          combine='by_coords')
uwind_expanded = expand_time_serie(uwind, 'wind_time')
for year in range(firstyear,lastyear+1):
    print("work on uwind", year)
    save_year(uwind_expanded,'Uwind','wind_time',
              year, erainterim_outdir)


vwind = xr.open_mfdataset(f'{erainterim_indir}/v10_ERAinterim_*_ROMS.nc',
                          combine='by_coords')
vwind_expanded = expand_time_serie(vwind, 'wind_time')
for year in range(firstyear,lastyear+1):
    print("work on vwind", year)
    save_year(vwind_expanded,'Vwind','wind_time',
              year, erainterim_outdir)

