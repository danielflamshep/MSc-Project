""" Use to load the PBL MiniMPL depths and TCCOON weather data"""
import os
import numpy as np
from netCDF4 import Dataset
from util import to_date_mnday as to_date


def get_pblh():
    path_to_files = r'E:\data\Mixing_Depths'
    files = os.listdir(path_to_files)
    data_dict = {}
    dates = []
    for file in files:
        data = open(os.path.join(path_to_files, file)).readlines()
        date = data[6].split(', ')
        date = "".join(date[:3])
        dates.append(date)

        quality = data[25].split()[80]
        data_list = [data[i].strip("\n").split(', ') for i in range(37, len(data))]
        data_array = np.array(data_list)
        data_array = data_array[~(data_array == 'NA').any(axis=1)]
        data_array = data_array[~(data_array == '    NA').any(axis=1)]
        data_array = data_array[~(data_array == '     NA').any(axis=1)]
        data_array = data_array.astype('float')
        pblh = data_array[:, 2]
        time = data_array[:, :2].mean(axis=1)/60**2  # time in HRS
        mean_diff = np.mean(np.ediff1d(time))
        print('{}| Q {} | obs {} from {:.3f}--{:.3f} | MD {:.3f} '.format(
               date, quality, time.shape[0], time[0], time[-1], mean_diff))

        data_dict[date+'PBLH'] = pblh
        data_dict[date+'HR'] = time
        data_dict[date+'QUAL'] = quality
    data_dict['dates'] = dates
    np.save('pblh', data_dict)


def get_inputs():
    path_file = r'E:\data\ci20120920_20170302.public.nc'

    vars = {'hour': 'HR', 'tout_C': 'T', 'wspd_m_s': 'U',
            'sia_AU': 'SI', 'pout_hPa': 'P', 'hout_RH': 'QV'}
    data = Dataset(path_file)
    d = data.variables['prior_date_index'][:]
    year = data.variables['year'][:]
    day = data.variables['day'][:]
    hour = data.variables['hour'][:]
    day_idx = []
    dates = []
    data_dict = {}

    for i in range(d.shape[0]-1):
        if d[i] != d[i+1]:
            day_idx.append(i+1)
            date = str(year[i]) + to_date(day[i], year[i])
            if len(day_idx) > 2:
                hrs = hour[day_idx[-2]:day_idx[-1]]
                mean_diff = np.mean(np.ediff1d(hrs))
                print('{} -- {} | OBS {} | {:.3f}--{:.3f} | mean diff {:.3f} '.format(
                      date, day[i], hrs.shape[0], hrs[0], hrs[-1], mean_diff))
            dates.append(date)

        if i == d.shape[0]-2:
            date = str(year[i]) + to_date(day[i], year[i])
            hrs = hour[day_idx[-2]:day_idx[-1]]
            mean_diff = np.mean(np.ediff1d(hrs))
            print('{} -- {} | OBS {} | {:.3f}--{:.3f} | mean diff {:.3f} '.format(
                  date, day[i], hrs.shape[0], hrs[0], hrs[-1], mean_diff))
            dates.append(date)

    print(dates[0], dates[-1])
    print(day[0], day[-1])
    print('number of dates', len(dates))
    print('number of day idx', len(day_idx))

    for var in vars.keys():
        d = data.variables[var][:]
        print(var, np.mean(d))
        dsplit = np.split(d, day_idx)
        print(var, len(dsplit))
        var = vars[var]
        for i in range(len(dsplit)):
            data_dict[dates[i]+var] = dsplit[i]
    data_dict['dates'] = dates
    np.save('inputs', data_dict)

if __name__ == "__main__":
    get_inputs()
    get_pblh()