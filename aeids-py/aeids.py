from BufferedPackets import BufferedPackets
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.models import model_from_json
from PcapReaderThread import PcapReaderThread
from tensorflow import Tensor

import binascii
import math
import numpy
import os
import sys
import thread
import time
import traceback

import csv

# def main(argv):
#     try:
#         root_directory = "/home/baskoro/Documents/Dataset/ISCX12/without retransmission/"
#         filename = root_directory + sys.argv[1]
#         prt = PcapReaderThread(filename)
#         prt.run()
#
#         while not prt.done:
#             print "sleeping"
#             time.sleep(1)
#
#         while prt.has_ready_message():
#             bp = prt.pop_connection()
#             print bp.get_payload()
#
#         print "DIE YOU!!!"
#
#     except IndexError:
#         print "Usage : python aeids.py filename [training|testing]"
#     except KeyboardInterrupt:
#         print "Good bye to you my trusted friend"
# root_directory = "/home/baskoro/Documents/Dataset/ISCX12/without retransmission/"
# root_directory = "/home/baskoro/Documents/Dataset/HTTP-Attack-Dataset/morphed-shellcode-attacks/"
# root_directory = "/home/baskoro/Documents/Dataset/HTTP-Attack-Dataset/shellcode-attacks/"
tensorboard_log_enabled = False
backend = "tensorflow"
done = False
prt = None
conf = {}
activation_functions = ["elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear", "softmax"]

# possible values: mean, median, zscore
threshold = "median"


def main(argv):
    try:
        # validate command line arguments
        if sys.argv[1] != "training" and sys.argv[1] != "predicting" and sys.argv[1] != "testing":
            raise IndexError("Phase {} does not exist.".format(sys.argv[1]))
        else:
            phase = sys.argv[1]

        if sys.argv[2] != "tcp" and sys.argv[2] != "udp":
            raise IndexError("Protocol {} is not supported.".format(sys.argv[3]))
        else:
            protocol = sys.argv[2]

        if not sys.argv[3].isdigit():
            raise IndexError("Port must be numeric.")
        else:
            port = sys.argv[3]

        # must be in form of comma separated, representing half of the layers (e.g. 200,100 means there are 3 layers,
        # with 200, 100, and 200 neurons respectively)
        try:
            hidden_layers = sys.argv[4].split(",")
            for neurons in hidden_layers:
                if not neurons.isdigit():
                    raise IndexError("Hidden layers must be comma separated numeric values")
        except ValueError:
            raise IndexError("Hidden layers must be comma separated numeric values")

        if sys.argv[5] not in activation_functions:
            raise IndexError("Activation function must be one of the following list")
        else:
            activation_function = sys.argv[5]

        try:
            dropout = float(sys.argv[6])
        except ValueError:
            raise IndexError("Dropout must be numeric.")

        filename = argv[7]
        if phase == "testing":
            aeids(phase, filename, protocol, port, hidden_layers, activation_function, dropout, sys.argv[8])
        else:
            aeids(phase, filename, protocol, port, hidden_layers, activation_function, dropout)
    except IndexError as e:
        print("Usage: python aeids.py <training|predicting|testing> <tcp|udp> <port> <hidden_layers> <activation_function> <dropout> <training filename> [testing filename]")
        print traceback.print_exc()
        exit(0)
    except KeyboardInterrupt:
        print "Interrupted"
        if prt is not None:
            prt.done = True
    except BaseException as e:
        print traceback.print_exc()
        if prt is not None:
            prt.done = True


def aeids(phase = "training", filename = "", protocol="tcp", port="80", hidden_layers = [200,100], activation_function = "relu", dropout = 0.0, testing_filename = ""):
    global done
    read_conf()

    if phase == "training":
        numpy.random.seed(666)

        autoencoder = init_model(hidden_layers, activation_function, dropout)

        if tensorboard_log_enabled and backend == "tensorflow":
            tensorboard_callback = TensorBoard(log_dir="./logs", batch_size=10000, write_graph=True, write_grads=True,
                                               histogram_freq=1)
            autoencoder.fit_generator(byte_freq_generator(filename, protocol, port), steps_per_epoch=100,
                                      epochs=100, verbose=1, callbacks=[tensorboard_callback])
            check_directory(filename, "models")
            autoencoder.save("models/{}/aeids-with-log-{}-hl{}-af{}-do{}.hdf5".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout), overwrite=True)
        else:
            autoencoder.fit_generator(byte_freq_generator(filename, protocol, port), steps_per_epoch=12000,
                                      epochs=10, verbose=1)
            check_directory(filename, "models")
            autoencoder.save("models/{}/aeids-{}-hl{}-af{}-do{}.hdf5".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout), overwrite=True)

        print "Training autoencoder finished. Calculating threshold..."
        predict_byte_freq_generator(autoencoder, filename, protocol, port, hidden_layers, activation_function, dropout, phase)
        done = True
        print "\nFinished."
    elif phase == "predicting":
        autoencoder = load_autoencoder(filename, protocol, port, hidden_layers, activation_function, dropout)
        predict_byte_freq_generator(autoencoder, filename, protocol, port, hidden_layers, activation_function, dropout, phase)
        done = True
        print "\nFinished."
    elif phase == "testing":
        autoencoder = load_autoencoder(filename, protocol, port, hidden_layers, activation_function, dropout)
        predict_byte_freq_generator(autoencoder, filename, protocol, port, hidden_layers, activation_function, dropout, phase, testing_filename)
        print "\nFinished."
    else:
        raise IndexError


def read_conf():
    global conf

    fconf = open("aeids.conf", "r")
    if not fconf:
        print "File aeids.conf does not exist."
        exit(-1)

    conf["root_directory"] = []
    lines = fconf.readlines()
    for line in lines:
        if line.startswith("#"):
            continue
        split = line.split("=", 2)
        print split
        if split[0] == "root_directory":
            conf["root_directory"].append(split[1].strip())

    fconf.close()


def init_model(hidden_layers = [200, 100], activation_function ="relu", dropout = 0):
    input_dimension = 256
    input = Input(shape=(input_dimension,))

    for i in range(0, len(hidden_layers)):
        if i == 0:
            encoded = Dense(int(hidden_layers[i]), activation=activation_function)(input)
        else:
            encoded = Dense(int(hidden_layers[i]), activation=activation_function)(encoded)

        encoded = Dropout(dropout)(encoded)

    for i in range(len(hidden_layers) - 1, -1, -1):
        if i == len(hidden_layers) - 1:
            decoded = Dense(int(hidden_layers[i]), activation=activation_function)(encoded)
        else:
            decoded = Dense(int(hidden_layers[i]), activation=activation_function)(decoded)

        decoded = Dropout(0.2)(decoded)

    if len(hidden_layers) == 1:
        decoded = Dense(input_dimension, activation="sigmoid")(encoded)
    else:
        decoded = Dense(input_dimension, activation="sigmoid")(decoded)
    autoencoder = Model(outputs=decoded, inputs=input)
    autoencoder.compile(loss="binary_crossentropy", optimizer="adadelta")

    return autoencoder


def load_autoencoder(filename, protocol, port, hidden_layers, activation_function, dropout):
    autoencoder = load_model("models/{}/aeids-{}-hl{}-af{}-do{}.hdf5".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout))
    return autoencoder


def byte_freq_generator(filename, protocol, port):
    global prt
    global conf
    prt = PcapReaderThread(get_pcap_file_fullpath(filename), protocol, port)
    prt.start()

    while not done:
        while not prt.done or prt.has_ready_message():
            if not prt.has_ready_message():
                time.sleep(0.0001)
                continue
            else:
                buffered_packets = prt.pop_connection()
                if buffered_packets is None:
                    time.sleep(0.0001)
                    continue
                if buffered_packets.get_payload_length() > 0:
                    byte_frequency = buffered_packets.get_byte_frequency()
                    dataX = numpy.reshape(byte_frequency, (1, 256))
                    yield dataX, dataX

        prt.reset_read_status()


def predict_byte_freq_generator(autoencoder, filename, protocol, port, hidden_layers, activation_function, dropout, phase="training", testing_filename = ""):
    global prt
    global threshold

    if prt is None:
        if phase == "testing":
            prt = PcapReaderThread(get_pcap_file_fullpath(testing_filename), protocol, port)
        else:
            prt = PcapReaderThread(get_pcap_file_fullpath(filename), protocol, port)
        prt.start()
    else:
        prt.reset_read_status()
        prt.delete_read_connections = True

    errors_list = []
    counter = 0
    print "predict"

    if phase == "testing":
        t1, t2 = load_threshold(filename, protocol, port, hidden_layers, activation_function, dropout)
        check_directory(filename, "results")
        fresult = open("results/{}/result-{}-hl{}-af{}-do{}-{}.csv".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout, testing_filename), "w")
        if fresult is None:
            raise Exception("Could not create file")

    # ftemp = open("results/data.txt", "wb")
    # fcsv = open("results/data.csv", "wb")
    # a = csv.writer(fcsv, quoting=csv.QUOTE_ALL)
    # time.sleep(2)
    i_counter = 0
    # for i in range(0,10):
    while not prt.done or prt.has_ready_message():
        if not prt.has_ready_message():
            time.sleep(0.0001)
        else:
            buffered_packets = prt.pop_connection()
            if buffered_packets is None:
                continue
            if buffered_packets.get_payload_length() == 0:
                continue

            i_counter += 1
            #print "{}-{}: {}".format(i_counter, buffered_packets.id, buffered_packets.get_payload()[:100])
            byte_frequency = buffered_packets.get_byte_frequency()
            # ftemp.write(buffered_packets.get_payload())
            # a.writerow(byte_frequency)
            data_x = numpy.reshape(byte_frequency, (1, 256))
            decoded_x = autoencoder.predict(data_x)
            # a.writerow(decoded_x[0])

            # fcsv.close()
            error = numpy.mean((decoded_x - data_x) ** 2, axis=1)
            # ftemp.write("\r\n\r\n{}".format(error))
            # ftemp.close()
            if phase == "training":
                errors_list.append(error)
            elif phase == "testing":
                decision = decide(error[0], t1, t2)
                fresult.write("{},{},{},{},{}\n".format(buffered_packets.id, error[0], decision[0], decision[1], decision[2]))

            counter += 1
            sys.stdout.write("\rCalculated {} connections.".format(counter))
            sys.stdout.flush()

    errors_list = numpy.reshape(errors_list, (1, len(errors_list)))
    if phase == "training" or phase == "predicting":
        save_mean_stdev(filename, protocol, port, hidden_layers, activation_function, dropout, errors_list)
        save_q3_iqr(filename, protocol, port, hidden_layers, activation_function, dropout, errors_list)
        save_median_mad(filename, protocol, port, hidden_layers, activation_function, dropout, errors_list)
    elif phase == "testing":
        fresult.close()


def save_mean_stdev(filename, protocol, port, hidden_layers, activation_function, dropout, errors_list):
    mean = numpy.mean(errors_list)
    stdev = numpy.std(errors_list)
    fmean = open("models/{}/mean-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout), "w")
    fmean.write("{},{}".format(mean, stdev))
    fmean.close()


def save_q3_iqr(filename, protocol, port, hidden_layers, activation_function, dropout, errors_list):
    qs = numpy.percentile(errors_list, [100, 75, 50, 25, 0])
    iqr = qs[1] - qs[3]
    MC = ((qs[0]-qs[2])-(qs[2]-qs[4]))/(qs[0]-qs[4])
    if MC >= 0:
        constant = 3
    else:
        constant = 4
    iqrplusMC = 1.5 * math.pow(math.e, constant * MC) * iqr
    print "IQR: {}\nMC: {}\nConstant: {}".format(iqr, MC, constant)
    fmean = open("models/{}/median-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout), "w")
    fmean.write("{},{}".format(qs[2], iqrplusMC))
    fmean.close()


def save_median_mad(filename, protocol, port, hidden_layers, activation_function, dropout, errors_list):
    median = numpy.median(errors_list)
    mad = numpy.median([numpy.abs(error - median) for error in errors_list])

    fmean = open("models/{}/zscore-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout), "w")
    fmean.write("{},{}".format(median, mad))
    fmean.close()


def load_threshold(filename, protocol, port, hidden_layers, activation_function, dropout):
    t1 = []
    t2 = []

    fmean = open(
        "models/{}/mean-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout), "r")
    line = fmean.readline()
    split = line.split(",")
    t1.append(split[0])
    t2.append(split[1])
    fmean.close()

    fmean = open(
        "models/{}/median-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout), "r")
    line = fmean.readline()
    split = line.split(",")
    t1.append(split[0])
    t2.append(split[1])
    fmean.close()

    fmean = open(
        "models/{}/zscore-{}-hl{}-af{}-do{}.txt".format(filename, protocol + port, ",".join(hidden_layers), activation_function, dropout), "r")
    line = fmean.readline()
    split = line.split(",")
    t1.append(split[0])
    t2.append(split[1])
    fmean.close()

    return t1, t2


def get_threshold(threshold_method, t1, t2):
    if threshold_method == "mean":
        return (float(t1[0]) + 2 * float(t2[0]))
    elif threshold_method == "median":
        return (float(t1[1]) + float(t2[1]))
    elif threshold_method == "zscore":
        return 3.5


def decide(mse, t1, t2):
    decision = []

    if mse > (float(t1[0]) + 2 * float(t2[0])):
        decision.append(1)
    else:
        decision.append(0)

    if mse > (float(t1[1]) + float(t2[1])):
        decision.append(1)
    else:
        decision.append(0)

    zscore = 0.6745 * (mse - float(t1[2])) / float(t2[2])
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
