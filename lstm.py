import pcapy
import sys
import numpy

from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from collections import Counter
from impacket import ImpactDecoder, ImpactPacket

ports = [20, 21, 22, 23, 25, 53, 80, 110, 139, 143, 443, 445]
batch_size = 100
data_input = {20: [], 21: [], 22: [], 23: [], 25: [], 53: [], 80: [], 110: [], 139: [], 143: [], 443: [], 445: []}
counter = {20: 0, 21: 0, 22: 0, 23: 0, 25: 0, 53: 0, 80: 0, 110: 0, 139: 0, 143: 0, 443: 0, 445: 0}
dataX = []
dataY = []
is_counting = False

def main(argv):
    try:
        if is_counting == False:
            seq_length = int(argv[2])
            model = Sequential()
            model.add(LSTM(256, input_shape=(seq_length, 1), return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(256, return_sequences=True))
            model.add(Dropout(0.3))
            #model.add(LSTM(256, return_sequences=True))
            #model.add(Dropout(0.3))
            model.add(LSTM(256))
            model.add(Dropout(0.3))
            model.add(Dense(256, activation="softmax"))
            model.compile(optimizer="adam", loss="categorical_crossentropy")

            #filepath = "models/lstm-" + str(d_port) + "-{epoch:02d}-{loss:.4f}.hdf5"
            #checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True)
            #callback_list = [checkpoint]
            print "Training model...\nSequence length : {}".format(seq_length)
            model.fit_generator(yield_read_dataset(argv[1], seq_length, int(argv[3])), samples_per_epoch=3000, nb_epoch=20, verbose=1)

            #model.fit(dataX, dataY, nb_epoch=50, batch_size=128, callbacks=callback_list)
            # model.train_on_batch(dataX, dataY)
            model.save("models/lstm-{}.hdf5".format(seq_length), overwrite=True)
            print "Saved"
            # counter[d_port] = 0
            # del data_input[d_port]
            # data_input[d_port] = []
            # print "1 batch"
        else:
            port = int(argv[3])
            seq_length = int(argv[2])
            read_dataset(argv[1], seq_length, port)
            print("Total : " + str(counter[int(argv[3])]))
    except IndexError:
        print "Usage : python lstm.py <pcap_filename> <sequence length> <port>"


def yield_read_dataset(filename, seq_length, port):
    #dataset_dir = "/home/baskoro/Documents/Dataset/ISCX12/without retransmission/"
    dataset_dir = "/home/baskoro/Documents/Dataset/Irene/"
    print(dataset_dir + filename)
    cap = pcapy.open_offline(dataset_dir + filename)
    #for i in range(5000):
    while(True):
        (header, packet) = cap.next()
        if not header:
            break
        payload = parse_packet(header, packet, port, seq_length)

        if payload is None:
            continue

        for i in range(0, len(payload) - seq_length, 1):
            seq_in = payload[i:i + seq_length]
            seq_out = payload[i + seq_length]
            dataX.append(seq_in)
            dataY.append(seq_out)

        n_pattern = len(dataX)

        while len(dataX) >= batch_size:
            X = dataX[0:batch_size]
            del dataX[0:batch_size]
            Y = dataY[0:batch_size]
            del dataY[0:batch_size]
            X = numpy.reshape(X, (batch_size, seq_length, 1))
            X = X / float(255)
            print "X : ", X.shape[0], X.shape[1], X.shape[2]
            print "Y : ", len(Y)
            Y = np_utils.to_categorical(Y, nb_classes=256)
            print "Y : ", Y.shape[0], Y.shape[1]
            print "Length of X" + str(len(dataX))

            if X is not None and Y is not None:
                yield X, Y
            else:
                break


def read_dataset(filename, seq_length, port):
    # dataset_dir = "/home/baskoro/Documents/Dataset/ISCX12/without retransmission/"
    dataset_dir = "/home/baskoro/Documents/Dataset/Irene/"
    print(dataset_dir + filename)
    cap = pcapy.open_offline(dataset_dir + filename)
    #for i in range(5000):
    while(True):
        (header, packet) = cap.next()
        if not header:
            break
        payload = parse_packet(header, packet, port, seq_length)

        if payload is None:
            continue

        for i in range(0, len(payload) - seq_length, 1):
            seq_in = payload[i:i + seq_length]
            seq_out = payload[i + seq_length]
            dataX.append(seq_in)
            dataY.append(seq_out)

        n_pattern = len(dataX)

        print n_pattern
        counter[port] += n_pattern
        del dataX[:]
        del dataY[:]

    #packet_generator(data_input, port)


def parse_packet(header, packet, port, seq_length):
    decoder = ImpactDecoder.EthDecoder()
    ether = decoder.decode(packet)

    #print str(ether.get_ether_type()) + " " + str(ImpactPacket.IP.ethertype)

    if ether.get_ether_type() == ImpactPacket.IP.ethertype:
        iphdr = ether.child()
        transporthdr = iphdr.child()

        s_addr = iphdr.get_ip_src()
        d_addr = iphdr.get_ip_dst()

        if isinstance(transporthdr, ImpactPacket.TCP):
            s_port = transporthdr.get_th_sport()
            d_port = transporthdr.get_th_dport()
            seq_num = transporthdr.get_th_seq()
            d_length = len(transporthdr.get_data_as_string())
            protocol = "tcp_ip"
            v_protocol = "1,0"
        elif isinstance(transporthdr, ImpactPacket.UDP):
            s_port = transporthdr.get_uh_sport()
            d_port = transporthdr.get_uh_dport()
            seq_num = 0
            d_length = transporthdr.get_uh_ulen()
            protocol = "udp_ip"
            v_protocol = "0,1"
        else:
            return

        if d_length <= seq_length:
            return

        if (d_port != port) and (s_port != port):
            return

        payload = str(transporthdr.get_data_as_string()).lower()
        ascii_payload = [ord(c) for c in payload]
        return ascii_payload
        #data_input[d_port].append(ascii_payload)
        #counter[d_port] += 1
        #print "{} - {}:{} -> {}:{}".format(counter[d_port], s_addr, s_port, d_addr, d_port)

        #if len(data_input[d_port]) == batch_size:
        #    train(d_port, data_input[d_port])


def packet_generator(payloads, d_port):
    dataX = []
    dataY = []

    for payload in payloads[d_port]:
        for i in range(0, len(payload) - seq_length, 1):
            seq_in = payload[i:i+seq_length]
            seq_out = payload[i+seq_length]
            dataX.append(seq_in)
            dataY.append(seq_out)

    n_pattern = len(dataX)
    dataX = numpy.reshape(dataX, (n_pattern, seq_length))
    dataX = dataX / float(255)
    print "X : ", dataX.shape[1], dataX.shape[2]
    print "Y : ", len(dataY)
    dataY = np_utils.to_categorical(dataY)
    print "Y : ", dataY.shape[0], dataY.shape[1]
    print "1 batch is gonna be used"
    counter[d_port] += 1
    payloads[d_port] = []
    yield dataX, dataY


if __name__ == '__main__':
	main(sys.argv)
