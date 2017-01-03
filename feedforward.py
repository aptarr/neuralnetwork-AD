from keras.models import Sequential, Model
from keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix
import numpy
import pandas
import sys


seed = 66
numpy.random.seed(seed)

ports = {'20': 3, '21': 4, '22': 5, '23': 6, '25': 7, '53': 8, '80': 9, '110': 10, '139': 11, '143': 12, '443': 13, '445': 14}
port_str = sys.argv[1]
port = ports[port_str]

#dataset = numpy.loadtxt("../previous implementation/PAYL/iscx-12jun-inbound.csv", delimiter=",", skiprows=1, usecols=(7:263))
dataset = pandas.read_csv("~/Documents/Dataset/ISCX12/without retransmission/csv-bytefreq/14jun.csv", delimiter=",", skiprows=0)
dataset = dataset.as_matrix()
dataset[:, 15] = dataset[:, 15].astype("float32") / 1500
dataset = dataset[dataset[:,port] == 1,:]

if dataset.size == 0:
    print "No training data on port " + port_str
    exit(-1)

X_train = dataset[:, 15:]
Y_train = dataset[:, 0]

dataset2 = pandas.read_csv("~/Documents/Dataset/ISCX12/without retransmission/csv-bytefreq/13jun.csv", delimiter=",", skiprows=0)
dataset2 = dataset2.as_matrix()
dataset2[:, 15] = dataset2[:, 15].astype("float32") / 1500
dataset2 = dataset2[dataset2[:,port] == 1,:]

if dataset.size == 0:
    print "No testing data on port " + port_str
    exit(-1)

X_test = dataset2[:, 15:]
Y_test = dataset2[:, 0]

model = Sequential()
input_dimension = 257

model.add(Dense(150, input_dim=input_dimension, init="uniform", activation="relu"))
model.add(Dense(50, init="uniform", activation="relu"))
model.add(Dense(20, init="uniform", activation="relu"))
model.add(Dense(1, init="uniform", activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["precision", "recall"])
model.fit(X_train, Y_train, batch_size=100, nb_epoch=10)

#model.train_on_batch(X_train, Y_train)
#model.train_on_batch(X_train2, Y_train2)

#scores_train = model.evaluate(X_train, Y_train)
predictions = model.predict(X_test, verbose=2)
rounded = [round(x) for x in predictions]
scores_test = confusion_matrix(Y_test, rounded)

print scores_test

#print("\nTraining %s: %.2f%%, %s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100, model.metrics_names[2], scores_train[2]*100))
#print("\nTesting %s: %.2f%%, %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100, model.metrics_names[2], scores_test[2]*100))