from pydataset import data
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


seals = data('seals')

#dupa = seals.get_value(500,'lat')
#empty lists ready to get values
d_long = []
d_lat = []
longitude = []
latitude = []

#iterates through rows of seals dataset an appends deltas values to
#delta_long and delta_lat lists
for index, row in seals.iterrows():

    d_long.append(row["delta_long"])
    d_lat.append(row["delta_lat"])
    longitude.append(row["long"])
    latitude.append(row["lat"])

#calculating mean and standard deviation of deltas
mean_long = np.mean (d_long)
mean_lat = np.mean (d_lat)
standard_deviation_long = np.std (d_long)
standard_deviation_lat = np.std (d_lat)

#until there are not 10k records in a list a loop does its job (10k can be changed to any value)
while len(d_long) <= 10000:

    if len(d_long) % 200 == 0:
        mean_long = np.mean (d_long[ len(d_long) - 1155: ])
        mean_lat = np.mean (d_lat[ len(d_lat) - 1155: ])
        standard_deviation_long = np.std (d_long [ len(d_long) - 1155: ])
        standard_deviation_lat = np.std (d_lat [ len(d_lat) - 1155: ])
    zero_one_lat = np.random.randint(low = 0, high = 2)

    #creates random values from range of mean-sd to mean+sd
    create_dlong = np.random.uniform (low = (mean_long - 1.75*standard_deviation_long),
    high = (mean_long + 1.75*standard_deviation_long))
    create_dlat = np.random.uniform (low = (mean_lat - 1.75*standard_deviation_lat),
    high = (mean_lat + 1.75*standard_deviation_lat))
    #print (zero_one_lat)
    if zero_one_lat == 0:
        create_dlat = create_dlat * -1
    else:
        continue

    d_long.append(create_dlong)
    d_lat.append(create_dlat)

    create_long = longitude[-1] + create_dlong
    create_lat = latitude[-1] + create_dlat

    if create_long > 180:
        create_long = create_long - 360
    elif create_long < -180:
        create_long = create_long + 360

    if create_lat > 90 or create_lat < -90:
        create_lat = latitude[-1] - create_dlat

    longitude.append(create_long)
    latitude.append(create_lat)

#turns 2 lists into 1 list of tuples
long_lat = zip(latitude, longitude)

#convertion from list of tuples to numpy tuples in array
np_latlon = np.array(long_lat)

print (np_latlon)

#x_train = long_lat[:9000]
#y_train = keras.utils.to_categorical(long_lat, 2)
