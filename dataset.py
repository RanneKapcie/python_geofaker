from pydataset import data
import numpy as np
import csv
import keras
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.optimizers import SGD

#using data function from pydataset module to get data under 'seals' variable
seals = data('seals')

#empty lists ready to get values
d_long = []
d_lat = []
longitude = []
latitude = []
time = []
group_l = []
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
id = 1

while len(d_long) < 19155:

#if length of current list is dividable by 200, script calculates new standard devation and mean
#depending on last 1155 records (first 200 is replaced by new 200 and so on)
    if len(d_long) % 200 == 0:
        mean_long = np.mean (d_long[ len(d_long) - 1155: ])
        mean_lat = np.mean (d_lat[ len(d_lat) - 1155: ])
        standard_deviation_long = np.std (d_long [ len(d_long) - 1155: ])
        standard_deviation_lat = np.std (d_lat [ len(d_lat) - 1155: ])

#choosing 0 or 1 randomly
    zero_one_lat = np.random.randint(low = 0, high = 2)

#creates random values from range of mean-sd*1.75 to mean+sd*1.75
    if (id <= 3000) or ((id > 9000) and (id <= 12000)):
        multiplier = 1
        group = 1
    elif (id <= 6000 and id > 3000) or (id > 12000 and id <= 15000):
        multiplier = 1.5
        group = 2
    elif (id <= 9000 and id > 6000) or (id > 15000):
        multiplier = 1.75
        group = 3

    create_dlong = np.random.uniform (low = (mean_long - multiplier*standard_deviation_long),
    high = (mean_long + multiplier*standard_deviation_long))
    create_dlat = np.random.uniform (low = (mean_lat - multiplier*standard_deviation_lat),
    high = (mean_lat + multiplier*standard_deviation_lat))

#now if zero_one_lat which was defined earlier is 0, than the delta is multiplied by -1
#if equals 1, than it stays as before
    if zero_one_lat == 0:
        create_dlat = create_dlat * -1
    else:
        continue

#adding delta longitude and latitude to lists
    d_long.append(create_dlong)
    d_lat.append(create_dlat)

#creating longitude and latitude
    create_long = longitude[-1] + create_dlong
    create_lat = latitude[-1] + create_dlat

#if there are unpossible results like 190 longitude, then it means, that seals have
#crossed the 180 longitude degree and are at -180
    if create_long > 180:
        create_long = create_long - 360
    elif create_long < -180:
        create_long = create_long + 360

#unfortunately, we can not do the same with latitude, because crossing 90 degrees south,
#does not make them 90 degrees north, so if result is over 90 degrees latitude,
#it makes last record - current delta latitude (changes from positive to negative)
    if create_lat > 90 or create_lat < -90:
        create_lat = latitude[-1] - create_dlat

#adds id, longitude and latitude to lists
    longitude.append(create_long)
    latitude.append(create_lat)
    group_l.append(group)

    id += 1 #increment id by 1

#calculating biggest delta in the set. depending on this, time between records will be added
max_delta = 0.7 * (max(d_long) + math.fabs(min(d_long))) + (max(d_lat) + math.fabs(min(d_lat)))
max_delta_1 = max_delta * 1/7
max_delta_2 = max_delta * 2/7
max_delta_3 = max_delta * 3/7
max_delta_4 = max_delta * 4/7
max_delta_5 = max_delta * 5/7
max_delta_6 = max_delta * 6/7

id = 0

#iterating through the records, and adding time to list, depending on delta
for coords in d_long:
    current_delta = math.fabs(d_long[id]) + math.fabs(d_lat[id])
    if current_delta < max_delta_1:
        time.append(1)
    elif (current_delta < max_delta_2) & (current_delta > max_delta_1):
        time.append(2)
    elif (current_delta < max_delta_3) & (current_delta > max_delta_2):
        time.append(3)
    elif (current_delta < max_delta_4) & (current_delta > max_delta_3):
        time.append(4)
    elif (current_delta < max_delta_5) & (current_delta > max_delta_4):
        time.append(5)
    elif (current_delta < max_delta_6) & (current_delta > max_delta_5):
        time.append(6)
    elif current_delta > max_delta_6:
        time.append(7)
    id += 1

#turns 2 lists into 1 list of tuples
long_lat = map (list, zip(latitude, longitude, time, group_l))
long_lat = np.array(long_lat)
long_lat = long_lat [:,np.newaxis]

#slicing the data
x_train = long_lat[:9000]
y_train = long_lat[9000:18000]
x_test = long_lat[:9000]
y_test = long_lat[9000:18000]

#checking shape of inputs
print y_train.shape, y_test.shape, x_train.shape, x_test.shape

#creating model and adding layers to it
model = Sequential()
model.add(Dense(units = 64, input_shape = (1, 4)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=25,
          batch_size=50)
score = model.evaluate(x_test, y_test, batch_size=100)

model.summary()
