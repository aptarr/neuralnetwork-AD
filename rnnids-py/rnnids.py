import binascii
import math
import numpy
import os
import sys
import time
import traceback
sys.path.insert(0, '../aeids-py/')

from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, LSTM, GRU, SimpleRNN, Embedding
from keras.models import model_from_json
from keras.utils import np_utils
from PcapReaderThread import PcapReaderThread
from StreamReaderThread import StreamReaderThread
from tensorflow import Tensor

done=False
prt=None
conf = {}
MAX_SEQ_LEN = 3960


def main(argv):
    global prt
    global done

    try:
        # validate command line arguments
        if sys.argv[1] != "training" and sys.argv[1] != "predicting" and sys.argv[1] != "testing" and sys.argv[1] != "counting":
            raise IndexError("Phase {} does not exist.".format(sys.argv[1]))
        elif sys.argv[1] == "counting":
            if sys.argv[2] != "tcp" and sys.argv[2] != "udp":
                raise IndexError("Protocol {} is not supported.".format(sys.argv[3]))
            else:
                protocol = sys.argv[2]

            if not sys.argv[3].isdigit():
                raise IndexError("Port must be numeric.")
            else:
                port = sys.argv[3]

            if not sys.argv[4].isdigit():
                raise IndexError("Sequence length must be numeric.")
            else:
                seq_length = int(sys.argv[4])

            read_conf()
            count_byte_seq_generator(sys.argv[5], protocol, port, seq_length)
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

            if phase == "training" and not sys.argv[9].isdigit():
                raise IndexError("Batch size must be numeric.")
            elif phase == "training":
                batch_size = int(sys.argv[9])

            read_conf()

            if phase == "testing":
                rnnids(phase, sys.argv[8], protocol, port, type, hidden_layers, seq_length, dropout, testing_filename=sys.argv[9])
            elif phase == "training":
                rnnids(phase, sys.argv[8], protocol, port, type, hidden_layers, seq_length, dropout, batch_size=batch_size)
            else:
                rnnids(phase, sys.argv[8], protocol, port, type, hidden_layers, seq_length, dropout)



    except IndexError as e:
        print(e)
	traceback.print_exc()
        print("Usage: python rnnids.py <training|predicting|testing> <rnn|lstm|gru> <tcp|udp> <port> <hidden_layers> "
              "<seq_length> <dropout> <training filename> [training batch size] [testing filename]\n"
              "or \n"
              "python rnnids.py counting <tcp|udp> <port> <seq_length> <training_filename>")
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

    conf["root_directory"] = []
    conf["training_filename"] = {"default-80-5": 100000}
    lines = fconf.readlines()
    for line in lines:
        if line.startswith("#"):
            continue
        split = line.split("=", 2)
        print split
        if split[0] == "root_directory":
            conf["root_directory"].append(split[1].strip())
        elif split[0] == "training_filename":
            tmp = split[1].split(":")
            conf["training_filename"]["{}-{}-{}".format(tmp[0], tmp[1], tmp[2])] = int(tmp[3])

    fconf.close()


def rnnids(phase = "training", filename = "", protocol="tcp", port="80", type = "rnn", hidden_layers = 2, seq_length = 3, dropout = 0.0, testing_filename = "", batch_size = 1):
    global done

    if phase == "training":
        numpy.random.seed(666)
        rnn_model = init_model(type, hidden_layers, seq_length, dropout)

        if "{}-{}-{}".format(filename, port, seq_length) in conf["training_filename"]:
            steps_per_epoch = conf["training_filename"]["{}-{}-{}".format(filename, port, seq_length)] / batch_size
        else:
            steps_per_epoch = conf["training_filename"]["default-80-5"] / batch_size

        print("Steps per epoch: {}".format(steps_per_epoch))

        rnn_model.fit_generator(byte_seq_generator(filename, protocol, port, seq_length, batch_size), steps_per_epoch=steps_per_epoch, epochs=10, verbose=1)
        check_directory(filename, "models")
        rnn_model.save("models/{}/{}-{}-hl{}-seq{}-do{}.hdf5".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), overwrite=True)
        print "Training model finished. Calculating prediction errors..."

        predict_byte_seq_generator(rnn_model, filename, protocol, port, type, hidden_layers, seq_length, dropout, phase)
        print "Finished"
        done = True
    elif phase == "predicting":
        rnn_model = load_rnn_model(type, hidden_layers, seq_length, dropout, protocol, port, filename)
        predict_byte_seq_generator(rnn_model, filename, protocol, port, type, hidden_layers, seq_length, dropout, phase)
        done = True
    elif phase == "testing":
        rnn_model = load_rnn_model(type, hidden_layers, seq_length, dropout, protocol, port, filename)
        predict_byte_seq_generator(rnn_model, filename, protocol, port, type, hidden_layers, seq_length, dropout, phase, testing_filename)
        done = True


def init_model(type, hidden_layers, seq_length, dropout):
    embed_len = 32
    rnn_len = 32
    rnn_model = Sequential()
    rnn_model.add(Embedding(256, embed_len, input_length=seq_length))
    for i in range(0, hidden_layers-1):
        if type == "rnn":
            rnn_model.add(SimpleRNN(units=rnn_len, input_shape=(seq_length,embed_len), return_sequences=True))
        elif type == "lstm":
            rnn_model.add(LSTM(units=rnn_len, input_shape=(seq_length, embed_len), return_sequences=True))
        elif type == "gru":
            rnn_model.add(GRU(units=rnn_len, input_shape=(seq_length, embed_len), return_sequences=True))

        rnn_model.add(Dropout(dropout))

    if type == "rnn":
        rnn_model.add(SimpleRNN(units=rnn_len, input_shape=(seq_length, embed_len)))
    elif type == "lstm":
        rnn_model.add(LSTM(units=rnn_len, input_shape=(seq_length, embed_len)))
    elif type == "gru":
        rnn_model.add(GRU(units=rnn_len, input_shape=(seq_length, embed_len)))

    rnn_model.add(Dense(256, activation="softmax"))
    rnn_model.compile(optimizer="adam", loss="categorical_crossentropy")

    return rnn_model


def load_rnn_model(type, hidden_layers, seq_length, dropout, protocol, port, filename):
    rnn_model = load_model("models/{}/{}-{}-hl{}-seq{}-do{}.hdf5".format(filename, type, protocol+port, hidden_layers, seq_length, dropout))
    return rnn_model


def load_threshold(type, hidden_layers, seq_length, dropout, protocol, port, filename):
    t1 = []
    t2 = []

    fmean = open("models/{}/mean-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "r")
    for line in fmean.readlines():
        split = line.split(",")
        t1.append(float(split[0]))
        t2.append(float(split[1]))
    fmean.close()

    fmean = open("models/{}/median-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "r")
    for line in fmean.readlines():
        split = line.split(",")
        t1.append(float(split[0]))
        t2.append(float(split[1]))
    fmean.close()

    fmean = open("models/{}/zscore-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "r")
    for line in fmean.readlines():
        split = line.split(",")
        t1.append(float(split[0]))
        t2.append(float(split[1]))
    fmean.close()

    return t1, t2


def byte_seq_generator(filename, protocol, port, seq_length, batch_size):
    global prt
    global root_directory

    prt = StreamReaderThread(get_pcap_file_fullpath(filename), protocol, port)
    prt.start()
    counter = 0

    while not done:
        while not prt.done or prt.has_ready_message():
            if not prt.has_ready_message():
                prt.wait_for_data()
                continue
            else:
                buffered_packet = prt.pop_connection()
                if buffered_packet is None:
                    prt.wait_for_data()
                    continue
                if buffered_packet.get_payload_length("server") > 0:
                    payload = [ord(c) for c in buffered_packet.get_payload("server")[:MAX_SEQ_LEN]]
                    #payload.insert(0, -1)  # mark as beginning of payloads

                    for i in range(0, len(payload) - seq_length, 1):
                        seq_in = payload[i:i + seq_length]
                        seq_out = payload[i + seq_length]
                        #X = numpy.reshape(seq_in, (1, seq_length, 1))
                        X = numpy.reshape(seq_in, (1, seq_length))
                        #X = X / float(255)
                        Y = np_utils.to_categorical(seq_out, num_classes=256)

                        if i == 0 or i % batch_size == 1:
                            dataX = X
                            dataY = Y
                        else:
                            dataX = numpy.r_["0,2", dataX, X]
                            dataY = numpy.r_["0,2", dataY, Y]

                        counter += 1
                        if i % batch_size == 0:
                            #print(dataX, dataY)
                            yield dataX, dataY

                    yield dataX, dataY

        # print "Total sequences: {}".format(counter)
        prt.reset_read_status()


def predict_byte_seq_generator(rnn_model, filename, protocol, port, type, hidden_layers, seq_length, dropout, phase="training", testing_filename = ""):
    global prt

    if prt is None:
        if phase == "testing":
            prt = StreamReaderThread(get_pcap_file_fullpath(testing_filename), protocol, port)
        else:
            prt = StreamReaderThread(get_pcap_file_fullpath(filename), protocol, port)
        prt.start()
    else:
        prt.reset_read_status()
        prt.delete_read_connections = True

    errors_list = [[],[]]
    counter = 0
    print "predict"

    if phase == "testing":
        t1, t2 = load_threshold(type, hidden_layers, seq_length, dropout, protocol, port, filename)
        check_directory(filename, "results")
        fresult = open("results/{}/result-{}-{}-hl{}-seq{}-do{}-{}.csv".format(filename, type, protocol+port, hidden_layers, seq_length, dropout, testing_filename), "w")
        if not fresult:
            raise Exception("Could not create file")

    # for i in range(0,100):
    while not prt.done or prt.has_ready_message():
        if not prt.has_ready_message():
            prt.wait_for_data()
        else:
            buffered_packet = prt.pop_connection()
            if buffered_packet is None:
                continue

            payload_length = buffered_packet.get_payload_length("server")
            if payload_length <= seq_length:
                continue

            payload = [ord(c) for c in buffered_packet.get_payload("server")[:MAX_SEQ_LEN]]
            payload_length = len(payload)
            #payload.insert(0, -1) # mark as beginning of payloads
            x_batch = []
            y_batch = []
            for i in range(0, len(payload) - seq_length, 1):
                seq_in = payload[i:i + seq_length]
                seq_out = payload[i + seq_length]
                x = numpy.reshape(seq_in, (1, seq_length))
                #x = numpy.reshape(seq_in, (1, seq_length, 1))
                #x = x / float(255)

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

            binary_anomaly_score = 0
            floating_anomaly_score = 0

            for i in range(0, len(y_batch)):
                if y_batch[i] != predicted_y[i]:
                    binary_anomaly_score += 1
                floating_anomaly_score += (y_batch[i] - predicted_y[i]) ** 2

            binary_prediction_error = float(binary_anomaly_score) / float(payload_length)
            floating_prediction_error = floating_anomaly_score / float(len(y_batch))

            if phase == "training" or phase == "predicting":
                errors_list[0].append(binary_prediction_error)
                errors_list[1].append(floating_prediction_error)
            elif phase == "testing":
                decision = decide([binary_prediction_error, floating_prediction_error], t1, t2)
                fresult.write("{},{},{},{},{},{},{},{},{}\n".format(buffered_packet.id, binary_prediction_error, decision[0], decision[1], decision[2],
                                                                    floating_prediction_error, decision[3], decision[4], decision[5]))

            counter += 1
            # for i in range(0,seq_length):
            #     print chr(payload[i]),
            #
            # for i in range(0,len(predicted_y)):
            #     print chr(predicted_y[i]),

            # sys.stdout.write("\rCalculated {} connections.".format(counter))
            # sys.stdout.flush()

    errors_list = numpy.reshape(errors_list, (2, len(errors_list[0])))
    if phase == "training" or phase == "predicting":
        save_mean_stdev(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename)
        save_q3_iqr(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename)
        save_median_mad(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename)
    elif phase == "testing":
        fresult.close()


def count_byte_seq_generator(filename, protocol, port, seq_length):
    global prt
    global root_directory

    prt = StreamReaderThread(get_pcap_file_fullpath(filename), protocol, port)
    prt.start()
    prt.delete_read_connections = True
    counter = 0
    stream_counter = 0

    while not prt.done or prt.has_ready_message():
        if not prt.has_ready_message():
            prt.wait_for_data()
            continue
        else:
            buffered_packet = prt.pop_connection()
            if buffered_packet is None:
                prt.wait_for_data()
                continue

            payload_length = buffered_packet.get_payload_length("server")
            if payload_length > MAX_SEQ_LEN:
                payload_length = MAX_SEQ_LEN
            # payload = buffered_packet.get_payload("server")
            # payload = "#" + payload  # mark as beginning of payloads
            # print(payload)
            # x = 0
            # for i in range(0, len(payload) - seq_length, 1):
            #     seq_in = payload[i:i + seq_length]
            #     seq_out = payload[i + seq_length]
            #     print(seq_in)
            #     print(seq_out)
            #     x += 1

            if payload_length > 0:
                stream_counter += 1
                counter += (payload_length - seq_length) + 1
                sys.stdout.write("\r{} streams, {} sequences.".format(stream_counter, counter))
                sys.stdout.flush()

    print "Total streams: {}. Total sequences: {}".format(stream_counter, counter)


def save_mean_stdev(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename):
    mean = numpy.mean(errors_list[0])
    stdev = numpy.std(errors_list[0])
    check_directory(filename, "models")
    fmean = open("models/{}/mean-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "w")
    fmean.write("{},{}\n".format(mean, stdev))
    mean = numpy.mean(errors_list[1])
    stdev = numpy.std(errors_list[1])
    fmean.write("{},{}".format(mean, stdev))
    fmean.close()


def save_q3_iqr(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename):
    qs = numpy.percentile(errors_list[0], [100, 75, 50, 25, 0])
    iqr = qs[1] - qs[3]
    MC = ((qs[0]-qs[2])-(qs[2]-qs[4]))/(qs[0]-qs[4])
    if MC >= 0:
        constant = 3
    else:
        constant = 4
    iqrplusMC = 1.5 * math.pow(math.e, constant * MC) * iqr
    check_directory(filename, "models")
    fmean = open("models/{}/median-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "w")
    fmean.write("{},{}\n".format(qs[2], iqrplusMC))

    qs = numpy.percentile(errors_list[1], [100, 75, 50, 25, 0])
    iqr = qs[1] - qs[3]
    MC = ((qs[0] - qs[2]) - (qs[2] - qs[4])) / (qs[0] - qs[4])
    if MC >= 0:
        constant = 3
    else:
        constant = 4
    iqrplusMC = 1.5 * math.pow(math.e, constant * MC) * iqr
    fmean.write("{},{}".format(qs[2], iqrplusMC))
    fmean.close()


def save_median_mad(type, protocol, port, hidden_layers, seq_length, dropout, errors_list, filename):
    median = numpy.median(errors_list)
    mad = numpy.median([numpy.abs(error - median) for error in errors_list])
    check_directory(filename, "models")
    fmean = open("models/{}/zscore-{}-{}-hl{}-seq{}-do{}.txt".format(filename, type, protocol+port, hidden_layers, seq_length, dropout), "w")
    fmean.write("{},{}\n".format(median, mad))

    median = numpy.median(errors_list)
    mad = numpy.median([numpy.abs(error - median) for error in errors_list])
    fmean.write("{},{}".format(median, mad))
    fmean.close()


def decide(mse, t1, t2):
    decision = []

    for i in range(0, 2):
        # mean threshold
        if mse[i] > (float(t1[i]) + 2 * float(t2[i])):
            decision.append(1)
        else:
            decision.append(0)

        # skewed median threshold
        if mse[i] > (float(t1[i+2]) + float(t2[i+2])):
            decision.append(1)
        else:
            decision.append(0)

        # zscore threshold
        zscore = 0.6745 * (mse[i] - float(t1[i+4]) / float(t2[i+4]))
        if zscore > 3.5 or zscore < -3.5:
            decision.append(1)
        else:
            decision.append(0)

    return decision


def check_directory(filename, root = "models"):
    if not os.path.isdir("./{}/{}".format(root, filename)):
        os.mkdir("./{}/{}".format(root, filename))


def get_pcap_file_fullpath(filename):
    global conf
    for i in range(0, len(conf["root_directory"])):
        if os.path.isfile(conf["root_directory"][i] + filename):
            return conf["root_directory"][i] + filename


if __name__ == '__main__':
	main(sys.argv)
