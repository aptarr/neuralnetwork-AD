import numpy
import sys

from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

seq_length = 5
filename = "models/lstm-80-49-0.7064.hdf5"
model = load_model(filename)
model.compile(loss="categorical_crossentropy", optimizer="adam")

start = [ord(c) for c in "get i"]
pattern = start

for i in range(200):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(255)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    #print index
    result = chr(index)
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print "\nDone"