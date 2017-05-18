import os
import numpy as np
import scipy.io as sio
from util import get_ind 

train_dir = 'E:/physics Msc/summer research/code/data/06/'
test_dir = 'E:/physics Msc/summer research/code/data/07/'

# LATITUDE LONGITUDE COORDINATES of region and number of grid boxes
# encompassing the location

CO = (43.5, 79.5)
grids = 5

# return a tuple of indices to use to find LAT LON region
CO_idx = get_ind(CO, grids, verts)

# get the hours which we're interested in (every 3 hrs) 
hrs = 24 
time_idx = np.arrray([3*i for i in range(1-hrs)])

for var in os.listdir(train_dir):
    # create list of files in directory
    var_files = os.listdir(train_dir+var)
    # specify the path to the file
    path_to_file = train_dir + var + '/'

    if var=='PBLH':
        # create a list of all files in folder for the coordinates and time splice
        # files have shape [LAT, LON, TIME]
        time_idx = np.arrray([3*i for i in range(7)])
        list_of_data = [sio.loadmat(path_to_file + file)[var][CO_idx[0], CO_idx[1], : , time_idx] for file in var_files]

        # join files together along time axis
        data = np.concatenate(list_of_data, axis=-1)
        print(data.shape)
        # format for correct input shape [time, lat, lon, vert,1]
        data = np.transpose(data, [])
        data = np.expand_dims(data, -1)
        print(data.shape)
        np.savez_compressed('train_' + var, var=data)
    else:
        # create a list of all files in folder 
        list_of_data = [sio.loadmat(path_to_file + file)[var] for file in var_files]
        # join files together along time axis
        data = np.concatenate(list_of_data, axis=-1)
        print(data.shape)
        # format for correct input shape [time, lat, lon, vert,1]
        data = np.transpose(data, [])
        data = np.expand_dims(data, -1)
        print(data.shape)
        np.savez_compressed('train_' + var, var=data)



