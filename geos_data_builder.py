from __future__ import print_function
import os
import numpy as np
import itertools as t
from netCDF4 import MFDataset


class DataBuilder(object):
    def __init__(self, LAT, LON, levels, years, months,
                 dir_roots=None,  # form [''dir_loc'']
                 dir_save='/home/dflamshe'):
        self.LAT = LAT
        self.LON = LON
        self.lats = np.linspace(9.75, 60., 202)
        self.lons = np.linspace(-130., -60., 225)
        self.idx_LAT = np.argmin(np.abs(LAT - self.lats))
        self.idx_LON = np.argmin(np.abs(LON - self.lons))
        self.years = years
        self.months = months
        self.levels = levels
        self.files = ['A1', 'A3dyn', 'I3']
        self.A1_vars = ['PBLH', 'TS', 'SWGDN']
        self.A3_vars = ['U', 'V']
        self.I3_vars = ['PS', 'T', 'QV']
        if len(self.years) > 1.0 and dir_roots is not None:
            self.dirs = dir_roots * len(self.years)
        elif dir_roots is not None and len(dir_roots) > 1.0:
            self.dirs = dir_roots
        else:
            self.dirs = ['/data/ctm/GEOS_0.25x0.3125_NA.d/GEOS_FP/',
                         '/users/jk/15/dbj/data/ctm/']
        self.dir_save = dir_save
        self.d = {}

    def get_data_dict(self):
        for year, month in t.product(self.years, self.months):
            dir_root = self.dirs[self.years.index(year)]
            file_path = os.path.join(dir_root, year, month)
            os.chdir(file_path)
            for var in self.A1_vars:
                file = 'GEOSFP.' + year + month + '**.A1.025x03125.NA.nc'
                self.d[year + month + var] = \
                    MFDataset(file, aggdim='time').variables[var][1::3, self.idx_LAT, self.idx_LON]
                print(year + month + var, self.d[year + month + var].shape[0])
            for var in self.A3_vars:
                file = 'GEOSFP.' + year + month + '**.A3dyn.025x03125.NA.nc'
                for level in self.levels:
                    self.d[year + month + var + level] = \
                        MFDataset(file, aggdim='time').variables[var][:, int(level), self.idx_LAT, self.idx_LON]
                    print(year + month + var + level, self.d[year + month + var + level].shape[0])
            for var in self.I3_vars:
                file = 'GEOSFP.' + year + month + '**.I3.025x03125.NA.nc'
                if var == 'PS':
                    self.d[year + month + var] = \
                        MFDataset(file, aggdim='time').variables[var][:, self.idx_LAT, self.idx_LON]
                else:
                    for level in self.levels:
                        self.d[year + month + var + level] = \
                            MFDataset(file, aggdim='time').variables[var][:, int(level), self.idx_LAT, self.idx_LON]
                        print(year + month + var + level, self.d[year + month + var + level].shape[0])
        return self.d

    def save_data_dict(self, version):
        if self.d == {}:
            print('git some data first')
        else:
            file_name = ''.join(self.years) + ''.join(self.months) + \
                        'LAT' + str(self.LAT) + 'LON' + str(self.LON) + \
                        'LVL' + self.levels[0] + '-' + self.levels[-1] + 'vers' + str(version)
            np.save(os.path.join(self.dir_save, file_name), self.d)
            print('SAVED that DATA BRO')


if __name__ == '__main__':
    LAT = 44.75
    LON = -80.3125
    LVLS = [str(x) for x in range(5)]
    years = ['2013', '2014']
    months = ['05', '06', '07']
    db = DataBuilder(LAT, LON, LVLS, years, months)
    db.get_data_dict()
    db.save_data_dict(2)
