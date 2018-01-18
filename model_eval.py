import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.optimizers import SGD
import csv
import numpy as np
import pandas

with open ('data.csv', 'r') as csvfile:
    csv_data = csv.reader(csvfile, delimiter = '\t', quotechar = '|')

array_data = pandas.read_csv('data.csv', sep = ',', header = None, delimiter = '\t')
#array_data.values()
#print np.arange()

x_train = array_data[:9000]
y_train = keras.utils.to_categorical(array_data[9000:10000], num_classes=15000)
x_test = array_data[:9000]
y_test = keras.utils.to_categorical(array_data[9000:10000], num_classes=15000)
y_train = y_train.reshape((-1, 1))

print len(x_train), len(y_train), len(x_test), len(y_test)



#np.reshape(x_train,(6000,3))
#keras blackbox thingstop
model = Sequential()
#model.add(Embedding((9000,4), 10, input_length=9000))
model.add(Dense(units = 50, input_shape = (None ,9000)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
#model.add(Flatten())
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=25,
          batch_size=100)
score = model.evaluate(x_test, y_test, batch_size=100)

model.summary()
#ValueError: Error when checking input: expected dense_1_input to have shape (None, 1) but got array with shape (8000, 2)
