from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.models import model_from_json
import numpy
import pandas
import sys


def count(mse, mean, stdev):
    if mse > (float(mean) + 2 * float(stdev)):
        return 1
    else:
        return 0


def save_model(filename, model, port):
    model_json = model.to_json()

    with open("models/" + filename + port + ".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("models/" + filename + port + ".h5")
    print("Saved {} to disk".format(filename))


def save_mean_stdev(port, mean, stdev):
    fmean = open("models/" + "mean" + port + ".txt", "w")
    fmean.write("{},{}".format(mean, stdev))
    fmean.close()


def load_model(filename, port):
    json_file = open("models/" + filename + port + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/" + filename + port + ".h5")
    print("Loaded {} to disk".format(filename))
    return loaded_model


def load_mean_stdev(port):
    fmean = open("models/" + "mean" + port + ".txt", "r")
    line = fmean.readline()
    split = line.split(",")
    fmean.close()
    return split[0], split[1]


try:
    seed = 66
    numpy.random.seed(seed)
    ports = {'20': 3, '21': 4, '22': 5, '23': 6, '25': 7, '53': 8, '80': 9, '110': 10, '139': 11, '143': 12, '443': 13,
             '445': 14}
    port_str = sys.argv[2]
    port = ports[sys.argv[2]]

    if sys.argv[1] == "training":
        dataset = pandas.read_csv("~/Documents/Dataset/ISCX12/without retransmission/csv-bytefreq/14jun.csv", delimiter=",", skiprows=0)
        dataset = dataset.as_matrix()
        dataset = dataset[dataset[:,port] == 1,:]
        dataset[:, 15] = dataset[:, 15].astype("float32") / 1500

        X_train = dataset[:, 15:]
        Y_train = dataset[:, 0]

        input_dimension = 257
        hidden_dimension = [200, 100]
        input = Input(shape=(input_dimension,))
        encoded = Dense(hidden_dimension[0], activation="relu") (input)
        encoded = Dropout(0.2) (encoded)
        encoded = Dense(hidden_dimension[1], activation="relu") (encoded)
        encoded = Dropout(0.2) (encoded)
        decoded = Dense(hidden_dimension[0], activation="relu")(encoded)
        decoded = Dropout(0.2) (decoded)
        decoded = Dense(input_dimension, activation="sigmoid") (decoded)
        autoencoder = Model(input=input, output=decoded)
        #encoder = Model(input=input, output=encoded)
        #encoded_input = Input(shape=(hidden_dimension,))
        #decoder_layer = autoencoder.layers[-1]
        #decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        autoencoder.compile(loss="binary_crossentropy", optimizer="adadelta")

        autoencoder.fit(X_train, X_train, batch_size=100, nb_epoch=20, shuffle=True)

        save_model("autoencoder", autoencoder, port_str)
        #save_model("encoder", encoder, port_str)
        #save_model("decoder", decoder, port_str)

        print("Counting mean and stddev...")
        #encoded_packets = encoder.predict(X_train)
        #decoded_packets = decoder.predict(encoded_packets)
        decoded_packets = autoencoder.predict(X_train)

        test = numpy.sqrt(numpy.mean((decoded_packets - X_train) ** 2, axis=1))
        mean = numpy.mean(test)
        stdev = numpy.std(test)
        save_mean_stdev(port_str, mean, stdev)
        print mean, stdev
    elif sys.argv[1] == "testing":
        dataset2 = pandas.read_csv("~/Documents/Dataset/ISCX12/without retransmission/csv-bytefreq/13jun.csv", delimiter=",",
                                   skiprows=0)
        dataset2 = dataset2.as_matrix()
        dataset2 = dataset2[dataset2[:, port] == 1, :]
        dataset2[:, 15] = dataset2[:, 15].astype("float32") / 1500

        X_test = dataset2[:, 15:]
        Y_test = dataset2[:, 0]

        autoencoder = load_model("autoencoder", port_str)
        #encoder = load_model("encoder", port_str)
        #decoder = load_model("decoder", port_str)

        #encoded_packets = encoder.predict(X_test)
        #decoded_packets = decoder.predict(encoded_packets)
        decoded_packets = autoencoder.predict(X_test)

        test = numpy.sqrt(numpy.mean((decoded_packets-X_test)**2, axis=1))
        mean, stdev = load_mean_stdev(port_str)
        count_all = numpy.vectorize(count)
        result = count_all(test, mean, stdev)
        print test
        print mean, stdev
        all = numpy.c_[test, result, Y_test]
        numpy.savetxt("results/result-{}.csv".format(port), all, delimiter=",")

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(0, len(test)):
            if result[i] == 1 and  Y_test[i] == 1:
                tp = tp + 1
            elif result[i] == 1 and Y_test[i] == 0:
                fp = fp + 1
            elif result[i] == 0 and Y_test[i] == 0:
                tn = tn + 1
            elif result[i] == 0 and Y_test[i] == 1:
                fn = fn + 1

        tpr = float(tp) / (tp + fn + 0.0001) * 100
        fpr = float(fp) / (fp + tn + 0.0001) * 100
        tnr = float(tn) / (tn + fp + 0.0001) * 100
        fnr = float(fn) / (tp + fn + 0.0001) * 100

        print tpr, fpr, tnr, fnr, tp, fp, tn, fn

except IndexError as e:
    print(e)
    print("Usage: python autoencoder.py <training|testing> <port>")