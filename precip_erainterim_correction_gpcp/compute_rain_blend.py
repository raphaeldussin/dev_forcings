#!/usr/bin/env python

# this should not be used since GPCP is rainfall only, blend precip = rain

import xarray as xr
import numpy as np

fyear=1979
lyear=2018
erainterim_indir = '/lustre/f2/dev/Raphael.Dussin/forcings/ERAinterim_OM4p5'
erainterim_outdir = '/lustre/f2/dev/Raphael.Dussin/forcings/ERAinterim_OM4p5'


for year in np.arange(fyear, lyear+1):
    dsprecip = xr.open_dataset(f'{erainterim_indir}/precip_Blend_Dussin_{year}_OM4p5_conservative.nc')
    dssnow = xr.open_dataset(f'{erainterim_indir}/snow_ERAinterim_{year}_OM4p5_conservative.nc')
    print(dsprecip)
    print(dssnow)
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
    dsrain.to_netcdf(f'{erainterim_outdir}/rain_Blend_Dussin_{year}_OM4p5_conservative.nc',
                     format='NETCDF3_64BIT',
                     encoding=encoding,
                     unlimited_dims=['time'])
