import os
import numpy as np
import scipy.io as sio

train_dir = 'E:/physics Msc/summer research/code/data/06/U/'
test_dir = 'E:/physics Msc/summer research/code/data/07/'


files = os.listdir(train_dir)
dt = np.dtype('int8')
# create a list of all files in folder
list_of_data = [sio.loadmat(train_dir + file)['U'].astype(dt) for file in files]
# join files together along time axis
data = np.concatenate(list_of_data, axis=-1)
print(data.shape)
# format for correct input shape [time, lat, lon, vert,1]
# data = np.transpose(data, [])
# data = np.expand_dims(data, -1)
print(data.shape)
np.savez_compressed('train_U', var=data)
