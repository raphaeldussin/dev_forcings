#!/usr/bin/env python

# not the right thing to do, remove high freq snow to slow moving precip, creates weird variance

import xarray as xr
import numpy as np

fyear=1979
lyear=2018
gpcp_indir = '/lustre/f2/dev/Raphael.Dussin/forcings/ERAinterim_OM4p5'
erainterim_indir = '/lustre/f2/dev/Raphael.Dussin/forcings/ERAinterim_OM4p5'
erainterim_outdir = '/lustre/f2/dev/Raphael.Dussin/forcings/ERAinterim_OM4p5'

def expand_time_serie_snow(ds):
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


def make_monthly_snow_interp(ds, year):
    # make monthly means for current year, before and after
    dsym1 = ds.sel(time=f"{year-1}").groupby(dsym1.time.dt.month).mean(dim='time')
    dsy0 = ds.sel(time=f"{year}").groupby(dsy0.time.dt.month).mean(dim='time')
    dsyp1 = ds.sel(time=f"{year+1}"),groupby(dsyp1.time.dt.month).mean(dim='time')
    # concat
    ds_3y = xr.concat([dsym1,dsy0,dsyp1], dim='time')
    # interp to the day
    ds_interp = ds_3y.resample(time="1D").interpolate("linear").sel(time=f"{year}")
    return ds_interp


dssnow = xr.open_mfdataset(f"{erainterim_indir}/snow_ERAinterim_*.nc")
dssnow_expanded = expand_time_serie_snow(dssnow)

for year in np.arange(fyear, lyear+1):
    dsprecip = xr.open_dataset(f'{gpcp_indir}/precip_GPCPv23_{year}_OM4p5_conservative.nc')
    dssnow = xr.open_dataset(f'{erainterim_indir}/snow_ERAinterim_{year}_OM4p5_conservative.nc')
    dsprecip['time'] =  dssnow['time']
    dsprecip['lon'] =  dssnow['lon']
    dsprecip['lat'] =  dssnow['lat']
    rain = dsprecip['precip'] - dssnow['snow']
    rain = rain.clip(min=0)
    dsrain = xr.Dataset()
    dsrain['rain'] = rain
    encoding = {'time':{'units': 'days since 1900-01-01T0:00:00',
                        'calendar': 'gregorian'},
                'lon': {'_FillValue': 1e+20},
                'lat': {'_FillValue': 1e+20},
                'rain': {'_FillValue': 1e+20},
               }
    dsrain.to_netcdf(f'{gpcp_indir}/rain_GPCPv23_{year}_OM4p5_conservative.nc',
                     format='NETCDF3_64BIT',
                     encoding=encoding,
                     unlimited_dims=['time'])
