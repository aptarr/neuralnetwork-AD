import binascii
import math
import numpy
import os
import sys
import time
sys.path.insert(0, '../aeids-py/')

from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, LSTM, GRU, SimpleRNN
from keras.models import model_from_json
from keras.utils import np_utils
from PcapReaderThread import PcapReaderThread
from tensorflow import Tensor

done=False
prt=None
root_directory = ""
threshold_method = ""


def main(argv):
    global prt
    global done

    try:
        # validate command line arguments
        if sys.argv[1] != "training" and sys.argv[1] != "predicting" and sys.argv[1] != "testing":
            raise IndexError("Phase {} does not exist.".format(sys.argv[1]))
        else:
            phase = sys.argv[1]

        if sys.argv[2] != "rnn" and sys.argv[2] != "lstm" and sys.argv[2] != "gru":
            raise IndexError("Type {} does not exist.".format(sys.argv[2]))
        else:
            type = sys.argv[2]

        if sys.argv[3] != "tcp" and sys.argv[3] != "udp":
            raise IndexError("Protocol {} is not supported.".format(sys.argv[3]))
        else:
            protocol = sys.argv[3]

        if not sys.argv[4].isdigit():
            raise IndexError("Port must be numeric.")
        else:
            port = sys.argv[4]

        if not sys.argv[5].isdigit():
            raise IndexError("Number of hidden layers must be numeric.")
        else:
            hidden_layers = int(sys.argv[5])

        if not sys.argv[6].isdigit():
            raise IndexError("Sequence length must be numeric.")
        else:
            seq_length = int(sys.argv[6])

        try:
            dropout = float(sys.argv[7])
        except ValueError:
            raise IndexError("Dropout must be numeric.")

        read_conf()
        rnnids(phase, sys.argv[8], protocol, port, type, hidden_layers, seq_length, dropout)

    except IndexError as e:
        print(e)
        print("Usage: python rnnids.py <training|predicting|testing> <rnn|lstm|gru> <tcp|udp> <port> <hidden_layers> <seq_length> <dropout> <filename>")
    except KeyboardInterrupt:
        if prt is not None:
            prt.done = True
        done = True


def read_conf():
    global root_directory
    global threshold_method

    fconf = open("rnnids.conf", "r")
    if not fconf:
        print "File rnnids.conf does not exist."
        exit(-1)

    lines = fconf.readlines()
    for line in lines:
        if line.startswith("#"):
            continue
        split = line.split("=", 2)
        print split
        if split[0] == "root_directory":
            root_directory = split[1].strip()
        elif split[0] == "threshold_method":
            threshold_method = split[1].strip()

    fconf.close()


def rnnids(phase = "training", filename = "", protocol="tcp", port="80", type = "rnn", hidden_layers = 2, seq_length = 3, dropout = 0.0):
    if phase == "training":
        rnn_model = init_model(type, hidden_layers, seq_length, dropout)

        rnn_model.fit_generator(byte_seq_generator(filename, protocol, port, seq_length), steps_per_epoch=100000, epochs=10, verbose=1)
        rnn_model.save("models/{}/{}-{}-hl{}-seq{}-do{}.hdf5".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), overwrite=True)
        print "Training model finished. Calculating prediction errors..."

        predict_byte_seq_generator(rnn_model, filename, protocol, port, type, hidden_layers, seq_length, dropout, phase)
        print "Finished"
        done = True
    elif phase == "predicting":
        rnn_model = load_rnn_model(type, hidden_layers, seq_length, dropout, protocol, port)
        predict_byte_seq_generator(rnn_model, filename, protocol, port, type, hidden_layers, seq_length, dropout, phase)
        done = True
    elif phase == "testing":
        rnn_model = load_rnn_model(type, hidden_layers, seq_length, dropout, protocol, port)
        predict_byte_seq_generator(rnn_model, filename, protocol, port, type, hidden_layers, seq_length, dropout, phase)
        done = True


def init_model(type, hidden_layers, seq_length, dropout):
    rnn_model = Sequential()
    for i in range(0, hidden_layers-1):
        if type == "rnn":
            rnn_model.add(SimpleRNN(units=256, input_shape=(seq_length,1), return_sequences=True))
        elif type == "lstm":
            rnn_model.add(LSTM(units=256, input_shape=(seq_length, 1), return_sequences=True))
        elif type == "gru":
            rnn_model.add(GRU(units=256, input_shape=(seq_length, 1), return_sequences=True))

        rnn_model.add(Dropout(dropout))

    if type == "rnn":
        rnn_model.add(SimpleRNN(units=256, input_shape=(seq_length, 1)))
    elif type == "lstm":
        rnn_model.add(LSTM(units=256, input_shape=(seq_length, 1)))
    elif type == "gru":
        rnn_model.add(GRU(units=256, input_shape=(seq_length, 1)))

    rnn_model.add(Dense(256, activation="softmax"))
    rnn_model.compile(optimizer="adam", loss="categorical_crossentropy")

    return rnn_model


def load_rnn_model(type, hidden_layers, seq_length, dropout, protocol, port, filename):
    rnn_model = load_model("models/{}/{}-{}-hl{}-seq{}-do{}.hdf5".format(filename/ type, protocol+port, hidden_layers, seq_length, dropout))
    return rnn_model


def load_threshold(type, hidden_layers, seq_length, dropout, protocol, port, threshold_model, filename):
    if threshold_model == "mean":
        fmean = open("models/{}/mean-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "r")
    elif threshold_model == "median":
        fmean = open("models/{}/median-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "r")
    elif threshold_model == "zscore":
        fmean = open("models/{}/mad-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "r")
    line = fmean.readline()
    split = line.split(",")
    fmean.close()
    return split[0], split[1]


def byte_seq_generator(filename, protocol, port, seq_length):
    global prt
    global root_directory

    print root_directory + filename
    prt = PcapReaderThread(root_directory + filename, protocol, port)
    prt.start()

    while not done:
        while not prt.done or prt.has_ready_message():
            if not prt.has_ready_message():
                time.sleep(0.0001)
                continue
            else:
                buffered_packet = prt.pop_connection()
                if buffered_packet is None:
                    time.sleep(0.0001)
                    continue
                if buffered_packet.get_payload_length() > 0:
                    payload = [ord(c) for c in buffered_packet.get_payload()]

                    for i in range(0, len(payload) - seq_length, 1):
                        seq_in = payload[i:i + seq_length]
                        seq_out = payload[i + seq_length]
                        X = numpy.reshape(seq_in, (1, seq_length, 1))
                        X = X / float(255)
                        Y = np_utils.to_categorical(seq_out, num_classes=256)

                        yield X, Y

        prt.reset_read_status()


def predict_byte_seq_generator(rnn_model, filename, protocol, port, type, hidden_layers, seq_length, dropout, phase="training"):
    global prt
    global threshold_method

    if prt is None:
        prt = PcapReaderThread(root_directory + filename, protocol, port)
        prt.start()
    else:
        prt.reset_read_status()
        prt.delete_read_connections = True

    errors_list = []
    counter = 0
    print "predict"

    if phase == "testing":
        t1, t2 = load_threshold(type, hidden_layers, seq_length, dropout, protocol, port, threshold_method, filename)
        check_directory(filename, "results")
        fresult = open("results/{}/result-{}-{}-hl{}-seq{}-do{}-{}.csv".format(filename, type, protocol+port, hidden_layers, seq_length, dropout, filename), "w")
        if not fresult:
            raise Exception("Could not create file")

    # for i in range(0,10):
    while not prt.done or prt.has_ready_message():
        if not prt.has_ready_message():
            time.sleep(0.0001)
        else:
            buffered_packet = prt.pop_connection()
            if buffered_packet is None:
                continue

            payload_length = buffered_packet.get_payload_length()
            if payload_length == 0:
                continue

            payload = [ord(c) for c in buffered_packet.get_payload()]
            x_batch = []
            y_batch = []
            for i in range(0, len(payload) - seq_length, 1):
                seq_in = payload[i:i + seq_length]
                seq_out = payload[i + seq_length]
                x = numpy.reshape(seq_in, (1, seq_length, 1))
                x = x / float(255)

                if len(x_batch) == 0:
                    x_batch = x
                    y_batch = seq_out
                else:
                    x_batch = numpy.r_[x_batch, x]
                    y_batch = numpy.r_[y_batch, seq_out]

            sys.stdout.write("\rCalculating {} connection. Len: {}".format(counter+1, len(y_batch)))
            sys.stdout.flush()
            if len(y_batch) < 350:
                prediction = rnn_model.predict_on_batch(x_batch)
                predicted_y = numpy.argmax(prediction, axis=1)
            else:
                predicted_y = []
                for i in range(0, len(y_batch), 350):
                    prediction = rnn_model.predict_on_batch(x_batch[i:i+350])
                    predicted_y = numpy.r_[predicted_y, (numpy.argmax(prediction, axis=1))]

            anomaly_score = 0

            for i in range(0, len(y_batch)):
                if y_batch[i] != predicted_y[i]:
                    anomaly_score += 1

            prediction_error = float(anomaly_score) / float(payload_length)
            if phase == "training" or phase == "predicting":
                errors_list.append(prediction_error)
            elif phase == "testing":
                decision = decide(prediction_error, threshold_method, t1, t2)
                fresult.write("{},{},{}\n".format(buffered_packet.id, prediction_error, decision))

            counter += 1
            # for i in range(0,seq_length):
            #     print chr(payload[i]),
            #
            # for i in range(0,len(predicted_y)):
            #     print chr(predicted_y[i]),

            # sys.stdout.write("\rCalculated {} connections.".format(counter))
            # sys.stdout.flush()

    errors_list = numpy.reshape(errors_list, (1, len(errors_list)))
    if phase == "training" or phase == "predicting":
        if threshold_method == "mean":
            save_mean_stdev(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename)
        elif threshold_method == "median":
            save_q3_iqr(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename)
        elif threshold_method == "zscore":
            save_median_mad(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename)
    elif phase == "testing":
         fresult.close()


def save_mean_stdev(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename):
    mean = numpy.mean(errors_list)
    stdev = numpy.std(errors_list)
    check_directory(filename, "models")
    fmean = open("models/{}/mean-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "w")
    fmean.write("{},{}".format(mean, stdev))
    fmean.close()


def save_q3_iqr(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename):
    qs = numpy.percentile(errors_list, [100, 75, 50, 25, 0])
    iqr = qs[1] - qs[3]
    MC = ((qs[0]-qs[2])-(qs[2]-qs[4]))/(qs[0]-qs[4])
    if MC >= 0:
        constant = 3
    else:
        constant = 4
    iqrplusMC = 1.5 * math.pow(math.e, constant * MC) * iqr
    print "IQR: {}\nMC: {}\nConstant: {}".format(iqr, MC, constant)
    check_directory(filename, "models")
    fmean = open("models/{}/median-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "w")
    fmean.write("{},{}".format(qs[2], iqrplusMC))
    fmean.close()


def save_median_mad(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename):
    median = numpy.median(errors_list)
    mad = numpy.median([numpy.abs(error - median) for error in errors_list])
    check_directory(filename, "models")
    fmean = open("models/{}/mad-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "w")
    fmean.write("{},{}".format(median, mad))
    fmean.close()


def decide(mse, threshold_method, t1, t2):
    if threshold_method == "mean":
        if mse > (float(t1) + 2 * float(t2)):
            return 1
        else:
            return 0
    elif threshold_method == "median":
        if mse > (float(t1) + float(t2)):
            return 1
        else:
            return 0
    elif threshold_method == "zscore":
        zscore = 0.6745 * (mse - float(t1)) / float(t2)
        if zscore > 3.5:
            return 1
        else:
            return 0


def check_directory(filename, root = "models"):
    if not os.isdir("./{}/{}".format(root, filename)):
        os.mkdir("./{}/{}".format(root, filename))


if __name__ == '__main__':
	main(sys.argv)