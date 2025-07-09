import binascii
import math
import numpy
import glob
import os
import sys
import time
import traceback
import collections
import warnings
import pydot
import graphviz
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

sys.path.insert(0, "../aeids-py/")

from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import (
    Dense,
    Input,
    Dropout,
    LSTM,
    GRU,
    SimpleRNN,
    Embedding,
    Bidirectional,
)
from keras.models import model_from_json
from keras.utils import to_categorical
from PcapReaderThread import PcapReaderThread
from StreamReaderThread import StreamReaderThread
from tensorflow import Tensor
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from scipy.stats import ks_2samp


done = False
prt = None
conf = {}
MAX_SEQ_LEN = 3960


def main(argv):
    global prt
    global done

    try:
        # validate command line arguments
        if (
            sys.argv[1] != "training"
            and sys.argv[1] != "predicting"
            and sys.argv[1] != "testing"
            and sys.argv[1] != "counting"
            and sys.argv[1] != "retraining"
        ):
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

            if (
                sys.argv[2] != "rnn"
                and sys.argv[2] != "lstm"
                and sys.argv[2] != "gru"
                and sys.argv[2] != "bi-lstm"
            ):
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

            if not sys.argv[8].isdigit():
                raise IndexError("Vocabularies length must be numeric.")
            else:
                oov_numbers = int(sys.argv[8])

            if phase == "training" and not sys.argv[10].isdigit():
                raise IndexError("Batch size must be numeric.")
            elif phase == "training":
                batch_size = int(sys.argv[10])

            read_conf()
            print(sys.argv[9])

            if phase == "testing":
                rnnids(
                    phase,
                    sys.argv[9],
                    protocol,
                    port,
                    type,
                    hidden_layers,
                    seq_length,
                    dropout,
                    oov_numbers=oov_numbers,
                    testing_filename=sys.argv[10],
                )
            elif phase == "training":
                rnnids(
                    phase,
                    sys.argv[9],
                    protocol,
                    port,
                    type,
                    hidden_layers,
                    seq_length,
                    dropout,
                    batch_size=batch_size,
                    oov_numbers=oov_numbers,
                )
            elif phase == "retraining":
                rnnids(
                    phase,
                    sys.argv[9],
                    protocol,
                    port,
                    type,
                    hidden_layers,
                    seq_length,
                    dropout,
                    batch_size=128,
                    oov_numbers=oov_numbers,
                    testing_filename=sys.argv[10],
                )
            else:
                rnnids(
                    phase,
                    sys.argv[9],
                    protocol,
                    port,
                    type,
                    hidden_layers,
                    seq_length,
                    dropout,
                    oov_numbers=oov_numbers,
                )

    except IndexError as e:
        print(e)
        # traceback.print_exc()
        print(
            "Usage: python rnnids.py <training|predicting|testing> <<rnn|lstm|gru|bi-lstm> <tcp|udp> <port> <hidden_layers> "
            "<seq_length> <dropout> <oov numbers> <training filename> [training batch size] [testing filename]\n"
            "or \n"
            "python rnnids.py counting <tcp|udp> <port> <seq_length> <training_filename>"
        )
    except KeyboardInterrupt:
        if prt is not None:
            prt.done = True
        done = True


class Tokenizer(object):
    def __init__(self, num_words=None, oov_token=None, **kwargs):
        if "nb_words" in kwargs:
            warnings.warn(
                "The `nb_words` argument in `Tokenizer` has been renamed `num_words`."
            )
            num_words = kwargs.pop("nb_words")
        document_count = kwargs.pop("document_count", 0)
        if kwargs:
            raise TypeError("Unrecognized keyword arguments: " + str(kwargs))
        self.word_counts = collections.OrderedDict()
        self.word_docs = collections.defaultdict(int)
        self.num_words = num_words
        self.oov_token = oov_token
        self.document_count = document_count
        self.index_docs = collections.defaultdict(int)
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        for text in texts:
            if text in self.word_counts:
                self.word_counts[text] += 1
            else:
                self.word_counts[text] = 1
        for text in set(texts):
            if text in self.word_docs:
                self.word_docs[text] += 1
            else:
                self.word_docs[text] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        self.word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences(self, texts):
        return self.texts_to_sequences_generator(texts)

    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        vect = []
        for text in texts:
            i = self.word_index.get(text)
            if i is not None:
                if num_words and i >= num_words:
                    if oov_token_index is not None:
                        vect.append(oov_token_index)
                else:
                    vect.append(i)
            elif self.oov_token is not None:
                vect.append(oov_token_index)
        yield vect


def read_conf():
    global root_directory
    global threshold_method

    fconf = open("rnnids.conf", "r")
    if not fconf:
        print("File rnnids.conf does not exist.")
        exit(-1)

    conf["root_directory"] = []
    conf["training_filename"] = {"default-80-5": 100000}
    lines = fconf.readlines()
    for line in lines:
        if line.startswith("#"):
            continue
        split = line.split("=", 2)
        print(split)
        if split[0] == "root_directory":
            conf["root_directory"].append(split[1].strip())
        elif split[0] == "training_filename":
            tmp = split[1].split(":")
            conf["training_filename"]["{}-{}-{}".format(tmp[0], tmp[1], tmp[2])] = int(
                tmp[3]
            )

    fconf.close()


def rnnids(
    phase="training",
    filename="",
    protocol="tcp",
    port="80",
    type="rnn",
    hidden_layers=2,
    seq_length=3,
    dropout=0.0,
    testing_filename="",
    batch_size=1,
    oov_numbers=1000,
):
    global done

    if phase == "training":
        print(filename)
        numpy.random.seed(666)
        rnn_model = init_model(type, hidden_layers, seq_length, dropout, oov_numbers)
        if "{}-{}-{}".format(filename, port, seq_length) in conf["training_filename"]:
            steps_per_epoch = (
                conf["training_filename"]["{}-{}-{}".format(filename, port, seq_length)]
                / batch_size
            )
            steps_per_epoch = int(steps_per_epoch)
        else:
            steps_per_epoch = conf["training_filename"]["default-80-5"] / batch_size
            steps_per_epoch = int(steps_per_epoch)
        tokenizer = Tokenizer(oov_numbers, oov_token="<OOV")
        tokenizer.fit_on_texts(fitting_text(phase, filename, seq_length))
        # print(tokenizer.word_index.get((50, 48, 32)))
        # print(tokenizer.word_index.get((48, 32, 100)))
        # print(tokenizer.word_index.get((32, 100, 101)))
        # print(f'Ini dia kuncinya = {tokenizer.index_word.get(202)}')

        # sequences = tokenizer.texts_to_sequences(byte_seq_generator(filename, seq_length))
        print("Steps per epoch: {}".format(steps_per_epoch))
        # for dataX, dataY in preprocessing_text(tokenizer, filename, seq_length, batch_size):
        #     print(dataX, dataY)

        history = rnn_model.fit(
            preprocessing_text(
                phase, tokenizer, filename, seq_length, batch_size, oov_numbers
            ),
            steps_per_epoch=steps_per_epoch,
            epochs=40,
            verbose=1,
        )
        plt.plot(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
            ],
            # [
            #     1,
            #     2,
            #     3,
            #     4,
            #     5,
            #     6,
            #     7,
            #     8,
            #     9,
            #     10,
            #     11,
            #     12,
            #     13,
            #     14,
            #     15,
            #     16,
            #     17,
            #     18,
            #     19,
            #     20,
            #     21,
            #     22,
            #     23,
            #     24,
            #     25,
            #     26,
            #     27,
            #     28,
            #     29,
            #     30,
            # ],
            history.history["loss"],
            label="Training Loss",
        )
        plt.title("Training Loss for HTTP")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("squares - HTTP.png")

        # plt.show()
        check_directory(filename, "models/vector/")
        with open(
            "models/vector/{}/{}-{}-hl{}-seq{}-do{}-oov{}.pickle".format(
                filename,
                type,
                protocol + port,
                hidden_layers,
                seq_length,
                dropout,
                oov_numbers,
            ),
            "wb",
        ) as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        rnn_model.save(
            "models/vector/{}/{}-{}-hl{}-seq{}-do{}-oov{}.hdf5".format(
                filename,
                type,
                protocol + port,
                hidden_layers,
                seq_length,
                dropout,
                oov_numbers,
            ),
            overwrite=True,
        )
        print("Training model finished. Calculating prediction errors...")

        predict_byte_seq_generator(
            rnn_model,
            filename,
            protocol,
            port,
            type,
            hidden_layers,
            seq_length,
            dropout,
            tokenizer,
            oov_numbers,
            phase,
        )
        print("Finished")
        done = True
    elif phase == "retraining":
        tf.config.run_functions_eagerly(True)
        tf.compat.v1.enable_eager_execution()

        files = glob.glob(f"errors/training_errors_{filename}_run*.csv")

        if len(files) == 0:
            error_old = pd.read_csv(f"errors/training_errors_{filename}.csv")
        elif len(files) == 1:
            print(f"masuk {files[0]}")
            error_old = pd.read_csv(files[0])
        else:
            print("masuk1")
            files.sort()  # assuming _run0, _run1, ... will sort correctly
            error_old = pd.read_csv(files[-1])

        error_new = pd.read_csv(
            "/home/apta/neuralnetwork-AD/rnn-apta/errors/testing_errors_{}.csv".format(
                testing_filename,
            )
        )

        print(filename)
        print(testing_filename)

        column_2_values1 = error_old["floating_prediction_error"]
        column_2_values2 = error_new["floating_prediction_error"]
        # column_2_values1 = error_old['binary_prediction_error']
        # column_2_values2 = error_new['binary_prediction_error']

        drift, result = detect_drift(column_2_values2, column_2_values1)

        print("Drift Detected:", drift)
        print(f"p-value = {result} (threshold = 0.05)")

        if drift:
            print(filename)
            numpy.random.seed(666)
            # rnn_model = init_model(
            #     type, hidden_layers, seq_length, dropout, oov_numbers
            # )

            rnn_model = load_rnn_model(
                type,
                hidden_layers,
                seq_length,
                dropout,
                protocol,
                port,
                filename,
                oov_numbers,
            )

            rnn_model.compile(
                loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
            )

            if (
                "{}-{}-{}".format(filename, port, seq_length)
                in conf["training_filename"]
            ):
                steps_per_epoch = (
                    conf["training_filename"][
                        "{}-{}-{}".format(filename, port, seq_length)
                    ]
                    / batch_size
                )
                steps_per_epoch = int(steps_per_epoch)  # update GPU
            else:
                steps_per_epoch = conf["training_filename"]["default-80-5"] / batch_size
                steps_per_epoch = int(steps_per_epoch)  # update GPU
            # tokenizer = Tokenizer(oov_numbers, oov_token="<OOV")

            tokenizer = load_tokenizer(
                type,
                hidden_layers,
                seq_length,
                dropout,
                protocol,
                port,
                filename,
                oov_numbers,
            )

            print("cek retraining")
            new_data = fitting_text(
                phase,
                testing_filename,
                seq_length,
            )
            print(testing_filename)
            tokenizer.fit_on_texts(new_data)

            rnn_model.summary()

            # print(tokenizer.word_index.get((50, 48, 32)))
            # print(tokenizer.word_index.get((48, 32, 100)))
            # print(tokenizer.word_index.get((32, 100, 101)))
            # print(f'Ini dia kuncinya = {tokenizer.index_word.get(202)}')

            # sequences = tokenizer.texts_to_sequences(byte_seq_generator(filename, seq_length))
            print("Steps per epoch: {}".format(steps_per_epoch))
            # for dataX, dataY in preprocessing_text(tokenizer, filename, seq_length, batch_size):
            #     print(dataX, dataY)

            history = rnn_model.fit(  # Update GPU fit_generator()
                preprocessing_text(
                    phase,
                    tokenizer,
                    testing_filename,
                    seq_length,
                    batch_size,
                    oov_numbers,
                ),
                steps_per_epoch=steps_per_epoch,
                epochs=20,
                verbose=1,
            )
            plt.plot(
                # [1, 2],
                # [
                #     1,
                #     2,
                #     3,
                #     4,
                #     5,
                #     6,
                #     7,
                #     8,
                #     9,
                #     10,
                #     11,
                #     12,
                #     13,
                #     14,
                #     15,
                #     16,
                #     17,
                #     18,
                #     19,
                #     20,
                #     21,
                #     22,
                #     23,
                #     24,
                #     25,
                #     26,
                #     27,
                #     28,
                #     29,
                #     30,
                # ],
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                ],
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                # range(1, len(history.history["loss"]) + 1),
                history.history["loss"],
                label="Training Loss",
            )
            plt.title("Training Loss for HTTP LSTM")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig(f"squares - {phase} - HTTP LSTM.png")

            model_dir = "models/vector/" + filename
            existing_models = glob.glob(model_dir)
            retraining_index = len(existing_models)
            folder_name = filename + f"_run{retraining_index}"

            # plt.show() #update GPU
            check_directory(folder_name, "models/vector")
            with open(
                "models/vector/{}/{}-{}-hl{}-seq{}-do{}-oov{}.pickle".format(
                    folder_name,
                    type,
                    protocol + port,
                    hidden_layers,
                    seq_length,
                    dropout,
                    oov_numbers,
                ),
                "wb",
            ) as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            rnn_model.save(
                "models/vector/{}/{}-{}-hl{}-seq{}-do{}-oov{}.hdf5".format(
                    folder_name,
                    type,
                    protocol + port,
                    hidden_layers,
                    seq_length,
                    dropout,
                    oov_numbers,
                ),
                overwrite=True,
            )
            print("Training model finished. Calculating prediction errors...")

            predict_byte_seq_generator(
                rnn_model,
                folder_name,
                protocol,
                port,
                type,
                hidden_layers,
                seq_length,
                dropout,
                tokenizer,
                oov_numbers,
                phase,
                testing_filename,
            )
            print("Finished")
            done = True
    elif phase == "predicting":
        rnn_model = load_rnn_model(
            type,
            hidden_layers,
            seq_length,
            dropout,
            protocol,
            port,
            filename,
            oov_numbers,
        )
        tokenizer = load_tokenizer(
            type,
            hidden_layers,
            seq_length,
            dropout,
            protocol,
            port,
            filename,
            oov_numbers,
        )
        predict_byte_seq_generator(
            rnn_model,
            filename,
            protocol,
            port,
            type,
            hidden_layers,
            seq_length,
            dropout,
            tokenizer,
            oov_numbers,
            phase,
        )
        done = True
    elif phase == "testing":
        rnn_model = load_rnn_model(
            type,
            hidden_layers,
            seq_length,
            dropout,
            protocol,
            port,
            filename,
            oov_numbers,
        )
        tokenizer = load_tokenizer(
            type,
            hidden_layers,
            seq_length,
            dropout,
            protocol,
            port,
            filename,
            oov_numbers,
        )
        predict_byte_seq_generator(
            rnn_model,
            filename,
            protocol,
            port,
            type,
            hidden_layers,
            seq_length,
            dropout,
            tokenizer,
            oov_numbers,
            phase,
            testing_filename,
        )
        done = True


def init_model(type, hidden_layers, seq_length, dropout, oov_numbers):
    embed_len = 32
    rnn_len = 32
    rnn_model = Sequential()
    rnn_model.add(Embedding(oov_numbers, embed_len, input_length=seq_length))
    for i in range(0, hidden_layers - 1):
        if type == "rnn":
            rnn_model.add(
                SimpleRNN(
                    units=rnn_len,
                    input_shape=(seq_length, embed_len),
                    return_sequences=True,
                )
            )
        elif type == "lstm":
            rnn_model.add(
                LSTM(
                    units=rnn_len,
                    input_shape=(seq_length, embed_len),
                    return_sequences=True,
                )
            )
        elif type == "bi-lstm":
            rnn_model.add(
                Bidirectional(
                    LSTM(
                        units=rnn_len,
                        input_shape=(seq_length, embed_len),
                        return_sequences=True,
                    )
                )
            )
        elif type == "gru":
            rnn_model.add(
                GRU(
                    units=rnn_len,
                    input_shape=(seq_length, embed_len),
                    return_sequences=True,
                )
            )

        rnn_model.add(Dropout(dropout))

    if type == "rnn":
        rnn_model.add(SimpleRNN(units=rnn_len, input_shape=(seq_length, embed_len)))
    elif type == "lstm":
        rnn_model.add(LSTM(units=rnn_len, input_shape=(seq_length, embed_len)))
    elif type == "bi-lstm":
        rnn_model.add(
            Bidirectional(LSTM(units=rnn_len, input_shape=(seq_length, embed_len)))
        )
    elif type == "gru":
        rnn_model.add(GRU(units=rnn_len, input_shape=(seq_length, embed_len)))

    rnn_model.add(Dense(oov_numbers, activation="softmax"))
    rnn_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    rnn_model.build(input_shape=(None, seq_length))
    rnn_model.summary()
    plot_model(
        rnn_model,
        to_file="model_plot_lstm.png",
        show_shapes=True,
        show_layer_names=True,
    )

    return rnn_model


class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)


def load_rnn_model(
    type, hidden_layers, seq_length, dropout, protocol, port, filename, oov_numbers
):
    start_time = time.time()
    rnn_model = load_model(
        "models/vector/{}/{}-{}-hl{}-seq{}-do{}-oov{}.hdf5".format(
            filename,
            type,
            protocol + port,
            hidden_layers,
            seq_length,
            dropout,
            oov_numbers,
        ),
        custom_objects={"CustomLSTM": CustomLSTM},
        compile=False,
    )
    print("Model Load time: {:.6} seconds".format(time.time() - start_time))
    return rnn_model


def load_tokenizer(
    type, hidden_layers, seq_length, dropout, protocol, port, filename, oov_numbers
):
    with open(
        "models/vector/{}/{}-{}-hl{}-seq{}-do{}-oov{}.pickle".format(
            filename,
            type,
            protocol + port,
            hidden_layers,
            seq_length,
            dropout,
            oov_numbers,
        ),
        "rb",
    ) as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def load_threshold(
    type, hidden_layers, seq_length, dropout, protocol, port, filename, oov_numbers
):
    t1 = []
    t2 = []

    fmean = open(
        "models/vector/{}/mean-{}-{}-hl{}-seq{}-do{}-oov{}.txt".format(
            filename,
            type,
            protocol + port,
            hidden_layers,
            seq_length,
            dropout,
            oov_numbers,
        ),
        "r",
    )
    for line in fmean.readlines():
        split = line.split(",")
        t1.append(float(split[0]))
        t2.append(float(split[1]))
    fmean.close()

    fmean = open(
        "models/vector/{}/median-{}-{}-hl{}-seq{}-do{}-oov{}.txt".format(
            filename,
            type,
            protocol + port,
            hidden_layers,
            seq_length,
            dropout,
            oov_numbers,
        ),
        "r",
    )
    for line in fmean.readlines():
        split = line.split(",")
        t1.append(float(split[0]))
        t2.append(float(split[1]))
    fmean.close()

    fmean = open(
        "models/vector/{}/zscore-{}-{}-hl{}-seq{}-do{}-oov{}.txt".format(
            filename,
            type,
            protocol + port,
            hidden_layers,
            seq_length,
            dropout,
            oov_numbers,
        ),
        "r",
    )
    for line in fmean.readlines():
        split = line.split(",")
        t1.append(float(split[0]))
        t2.append(float(split[1]))
    fmean.close()

    return t1, t2


def fitting_text(
    phase,
    filename,
    seq_length,
):
    check_directory("texts_cleaned")
    ftext = open("texts_cleaned/{}.txt".format(filename), "r")
    if not ftext:
        print("File rnnids.conf does not exist.")
        exit(-1)

    lines = ftext.readlines()
    counter = 0
    length = 0
    for line in lines:
        if phase == "retraining":
            id = line.split("; ")[0]
            line = line.split("; ")[1]
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.replace("\n", "")
        test = line.split(", ")
        # if counter == 0:
        #     test = [int(x) for x in test]
        # else:
        #     test = [int(x) for x in test]
        #     test = remainder + test
        # remainder = test[-(seq_length-1):]
        test = [int(x) for x in test]
        test = tuple(test)
        # print(f'main seq: {test}')
        for i in range(0, len(test) + 1 - seq_length, 1):
            seq_in = test[i : i + seq_length]
            # batch_seq.append(seq_in)
            # length += len(batch_seq)
            yield seq_in
        # counter += 1
        # tokenizer.fit_on_texts(batch_seq)
        # sys.stdout.write("\rCalculated {} connections, Length: {}.".format(counter, length))
        # sys.stdout.flush()


def preprocessing_text(phase, tokenizer, filename, seq_length, batch_size, oov_numbers):
    check_directory("texts_cleaned")
    while not done:
        ftext = open("texts_cleaned/{}.txt".format(filename), "r")
        if not ftext:
            print("File rnnids.conf does not exist.")
            exit(-1)

        counter_main = 0
        lines = ftext.readlines()
        for line in lines:
            if phase == "retraining":
                id = line.split("; ")[0]
                line = line.split("; ")[1]
            batch_seq = []
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("\n", "")
            test = line.split(", ")
            # if counter == 0:
            #     test = [int(x) for x in test]
            # else:
            #     test = [int(x) for x in test]
            #     test = remainder + test
            # remainder = test[-(seq_length-1):]
            test = [int(x) for x in test]
            test = tuple(test)
            # print(f'main seq: {test}')
            for i in range(0, len(test) + 1 - seq_length, 1):
                seq_in = test[i : i + seq_length]
                batch_seq.append(seq_in)
                # print(seq_in)
            # if counter_main == 0:
            #     print(batch_seq)
            items = tokenizer.texts_to_sequences(batch_seq)
            for item in items:
                counter = 0
                for i in range(0, len(item) - seq_length, 1):
                    seq_in = item[i : i + seq_length]
                    seq_out = item[i + seq_length]
                    X = numpy.reshape(seq_in, (1, seq_length))
                    Y = to_categorical(seq_out, num_classes=oov_numbers)
                    if counter == 0:
                        dataX = X
                        dataY = Y
                    else:
                        dataX = numpy.r_["0,2", dataX, X]
                        dataY = numpy.r_["0,2", dataY, Y]
                    counter += 1
                    if dataX.shape[0] % batch_size == 0:
                        counter = 0
                        yield dataX, dataY.reshape(-1, oov_numbers)
                        # yield dataX, dataY
                if dataX.shape[0] != 0:
                    yield dataX, dataY.reshape(-1, oov_numbers)
            counter_main += 1


def predict_byte_seq_generator(
    rnn_model,
    filename,
    protocol,
    port,
    type,
    hidden_layers,
    seq_length,
    dropout,
    tokenizer,
    oov_numbers,
    phase="training",
    testing_filename="",
):
    if phase == "testing" or phase == "retraining":
        file = testing_filename
    else:
        file = filename

    errors_list = [[], []]
    counter_main = 0
    print("predict")

    if phase == "testing":
        t1, t2 = load_threshold(
            type,
            hidden_layers,
            seq_length,
            dropout,
            protocol,
            port,
            filename,
            oov_numbers,
        )
        check_directory(filename, "results/vector")
        fresult = open(
            "results/vector/{}/result-{}-{}-hl{}-seq{}-do{}-oov{}-{}.csv".format(
                filename,
                type,
                protocol + port,
                hidden_layers,
                seq_length,
                dropout,
                oov_numbers,
                testing_filename,
            ),
            "w",
        )
        if not fresult:
            raise Exception("Could not create file")

    check_directory("texts_cleaned")
    ftext = open("texts_cleaned/{}.txt".format(file), "r")
    if not ftext:
        print("File rnnids.conf does not exist.")
        exit(-1)

    if phase == "retraining":
        pattern = os.path.join("errors", f"training_errors_{filename}_run*.csv")
        existing_files = glob.glob(pattern)

        # If there are no existing files, start from run1
        if existing_files:
            # Extract the last run number and increment it
            last_run = max(
                [int(f.split("_run")[-1].split(".")[0]) for f in existing_files],
                default=0,
            )
            retraining_index = last_run + 1
        else:
            retraining_index = 1

        # Check if '_run' is already part of the filename to avoid duplication
        if "_run" not in filename:
            # Create the new error log path with the appropriate run number
            error_log_path = (
                f"errors/training_errors_{filename}_run{retraining_index}.csv"
            )
        else:
            # If '_run' is already in filename, just use it as it is
            error_log_path = f"errors/training_errors_{filename}.csv"

        # Create the new error log path with the appropriate run number
        error_log_path = f"errors/training_errors_{filename}_run{retraining_index}.csv"
    elif phase == "training":
        error_log_path = f"errors/training_errors_{filename}.csv"
    else:  # testing
        error_log_path = f"errors/testing_errors_{testing_filename}.csv"

    check_directory("errors")
    error_log_file = open(error_log_path, "w")
    error_log_file.write(
        "sequence_index,binary_prediction_error,floating_prediction_error\n"
    )

    lines = ftext.readlines()
    for line in lines:
        batch_seq = []
        if phase == "testing" or phase == "retraining":
            id = line.split("; ")[0]
            line = line.split("; ")[1]
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.replace("\n", "")
        test = line.split(", ")
        # if counter == 0:
        #     test = [int(x) for x in test]
        # else:
        #     test = [int(x) for x in test]
        #     test = remainder + test
        # remainder = test[-(seq_length-1):]
        test = [int(x) for x in test]
        test = tuple(test)
        # print(f'main seq: {test}')
        for i in range(0, len(test) + 1 - seq_length, 1):
            seq_in = test[i : i + seq_length]
            batch_seq.append(seq_in)
            # print(seq_in)
        # if counter_main == 0:
        #     print(batch_seq)
        items = tokenizer.texts_to_sequences(batch_seq)
        for item in items:
            x_batch = []
            y_batch = []
            for i in range(0, len(item) - seq_length, 1):
                seq_in = item[i : i + seq_length]
                seq_out = item[i + seq_length]
                X = numpy.reshape(seq_in, (1, seq_length))
                # print(X[0])
                if len(x_batch) == 0:
                    # print(X)
                    x_batch = X
                    y_batch = [seq_out]
                else:
                    # print(y_batch)
                    x_batch = numpy.r_[x_batch, X]
                    y_batch = numpy.r_[y_batch, seq_out]

            if len(x_batch) > 0:
                prediction = rnn_model.predict_on_batch(x_batch)
                predicted_y = numpy.argmax(prediction, axis=1)
                # print(y_batch)
                # print(predicted_y)

                binary_anomaly_score = 0
                floating_anomaly_score = 0

                for i in range(0, len(y_batch)):
                    if y_batch[i] != predicted_y[i]:
                        binary_anomaly_score += 1
                    floating_anomaly_score += (y_batch[i] - predicted_y[i]) ** 2

                binary_prediction_error = float(binary_anomaly_score) / float(len(item))
                floating_prediction_error = floating_anomaly_score / float(len(y_batch))

                if error_log_file:
                    error_log_file.write(
                        f"{counter_main},{binary_prediction_error},{floating_prediction_error}\n"
                    )

                if (
                    phase == "training"
                    or phase == "predicting"
                    or phase == "retraining"
                ):
                    errors_list[0].append(binary_prediction_error)
                    errors_list[1].append(floating_prediction_error)

                elif phase == "testing":
                    decision = decide(
                        [binary_prediction_error, floating_prediction_error], t1, t2
                    )
                    fresult.write(
                        "{},{},{},{},{},{},{},{},{}\n".format(
                            id,
                            binary_prediction_error,
                            decision[0],
                            decision[1],
                            decision[2],
                            floating_prediction_error,
                            decision[3],
                            decision[4],
                            decision[5],
                        )
                    )

            else:
                continue
            counter_main += 1

    #         # for i in range(0,seq_length):
    #         #     print chr(payload[i]),
    #         #
    #         # for i in range(0,len(predicted_y)):
    #         #     print chr(predicted_y[i]),

    #         # sys.stdout.write("\rCalculated {} connections.".format(counter))
    #         # sys.stdout.flush()

    print(errors_list[0])
    errors_list = numpy.reshape(errors_list, (2, len(errors_list[0])))
    print(errors_list)
    if phase == "training" or phase == "predicting" or phase == "retraining":
        save_mean_stdev(
            type,
            protocol,
            port,
            hidden_layers,
            seq_length,
            dropout,
            errors_list,
            filename,
            oov_numbers,
        )
        save_q3_iqr(
            type,
            protocol,
            port,
            hidden_layers,
            seq_length,
            dropout,
            errors_list,
            filename,
            oov_numbers,
        )
        save_median_mad(
            type,
            protocol,
            port,
            hidden_layers,
            seq_length,
            dropout,
            errors_list,
            filename,
            oov_numbers,
        )
    elif phase == "testing":
        fresult.close()


def count_byte_seq_generator(filename, protocol, port, seq_length):
    global prt
    global root_directory
    print(filename)
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
                sys.stdout.write(
                    "\r{} streams, {} sequences.".format(stream_counter, counter)
                )
                sys.stdout.flush()

    print("Total streams: {}. Total sequences: {}".format(stream_counter, counter))
    # print ("Maxium payload length: {}. Maximum number of sequences: {}".format(max_payload, max_sequences))


def save_mean_stdev(
    type,
    protocol,
    port,
    hidden_layers,
    seq_length,
    dropout,
    errors_list,
    filename,
    oov_numbers,
):
    mean = numpy.mean(errors_list[0])
    stdev = numpy.std(errors_list[0])
    check_directory(filename, "models/vector")
    fmean = open(
        "models/vector/{}/mean-{}-{}-hl{}-seq{}-do{}-oov{}.txt".format(
            filename,
            type,
            protocol + port,
            hidden_layers,
            seq_length,
            dropout,
            oov_numbers,
        ),
        "w",
    )
    fmean.write("{},{}\n".format(mean, stdev))
    mean = numpy.mean(errors_list[1])
    stdev = numpy.std(errors_list[1])
    fmean.write("{},{}".format(mean, stdev))
    fmean.close()


def save_q3_iqr(
    type,
    protocol,
    port,
    hidden_layers,
    seq_length,
    dropout,
    errors_list,
    filename,
    oov_numbers,
):
    qs = numpy.percentile(errors_list[0], [100, 75, 50, 25, 0])
    iqr = qs[1] - qs[3]
    print(qs)
    MC = ((qs[0] - qs[2]) - (qs[2] - qs[4])) / (qs[0] - qs[4])
    if MC >= 0:
        constant = 3
    else:
        constant = 4

    iqrplusMC = 1.5 * math.pow(math.e, constant * MC) * iqr
    check_directory(filename, "models/vector")
    fmean = open(
        "models/vector/{}/median-{}-{}-hl{}-seq{}-do{}-oov{}.txt".format(
            filename,
            type,
            protocol + port,
            hidden_layers,
            seq_length,
            dropout,
            oov_numbers,
        ),
        "w",
    )
    fmean.write("{},{}\n".format(qs[3], qs[1]))

    qs = numpy.percentile(errors_list[1], [100, 75, 50, 25, 0])
    iqr = qs[1] - qs[3]
    MC = ((qs[0] - qs[2]) - (qs[2] - qs[4])) / (qs[0] - qs[4])
    if MC >= 0:
        constant = 3
    else:
        constant = 4
    iqrplusMC = 1.5 * math.pow(math.e, constant * MC) * iqr
    fmean.write("{},{}".format(qs[3], qs[1]))
    fmean.close()


def save_median_mad(
    type,
    protocol,
    port,
    hidden_layers,
    seq_length,
    dropout,
    errors_list,
    filename,
    oov_numbers,
):
    median = numpy.median(errors_list[0])
    mad = numpy.median([numpy.abs(error - median) for error in errors_list[0]])
    check_directory(filename, "models/vector")
    fmean = open(
        "models/vector/{}/zscore-{}-{}-hl{}-seq{}-do{}-oov{}.txt".format(
            filename,
            type,
            protocol + port,
            hidden_layers,
            seq_length,
            dropout,
            oov_numbers,
        ),
        "w",
    )
    fmean.write("{},{}\n".format(median, mad))

    median = numpy.median(errors_list[1])
    mad = numpy.median([numpy.abs(error - median) for error in errors_list[1]])
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
        iqr = float(t2[i + 2]) - float(t1[i + 2])
        upper_bound = float(t2[i + 2]) + 1.5 * iqr
        lower_bound = float(t1[i + 2]) - 1.5 * iqr
        if mse[i] < lower_bound or mse[i] > upper_bound:
            decision.append(1)
        else:
            decision.append(0)

        # zscore threshold
        if t2[i + 4] == 0:
            t2[i + 4] = 0.000001
        zscore = 0.6745 * (mse[i] - float(t1[i + 4])) / float(t2[i + 4])
        if zscore > 3.5 or zscore < -3.5:
            decision.append(1)
        else:
            decision.append(0)

    return decision


def detect_drift(train_errors, test_errors, alpha=0.05):
    result = {}

    stat, p_value = ks_2samp(train_errors, test_errors)
    result["statistic"] = stat
    result["p_value"] = p_value
    drift_detected = p_value < alpha

    return drift_detected, result


def check_directory(filename, root="/home/apta/"):
    if not os.path.isdir("{}/{}".format(root, filename)):
        os.mkdir("{}/{}".format(root, filename))


def get_pcap_file_fullpath(filename):
    global conf
    for i in range(0, len(conf["root_directory"])):
        if os.path.isfile(conf["root_directory"][i] + filename):
            return conf["root_directory"][i] + filename


if __name__ == "__main__":
    main(sys.argv)
