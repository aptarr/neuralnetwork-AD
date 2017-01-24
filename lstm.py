import pcapy
import sys
import numpy

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from collections import Counter
from impacket import ImpactDecoder, ImpactPacket

ports = [20, 21, 22, 23, 25, 53, 80, 110, 139, 143, 443, 445]
seq_length = 20
batch_size = 32
data_input = {20: [], 21: [], 22: [], 23: [], 25: [], 53: [], 80: [], 110: [], 139: [], 143: [], 443: [], 445: []}
counter = {20: 0, 21: 0, 22: 0, 23: 0, 25: 0, 53: 0, 80: 0, 110: 0, 139: 0, 143: 0, 443: 0, 445: 0}


def main(argv):
    read_dataset(argv[1], "training")


def read_dataset(filename, mode):
    dataset_dir = "/home/baskoro/Documents/Dataset/ISCX12/without retransmission/"
    cap = pcapy.open_offline(dataset_dir + filename)

    while(1):
        (header, packet) = cap.next()
        if not header:
            break
        parse_packet(header, packet)

    #lstm_train(data_input, 80)


def parse_packet(header, packet):
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

        if d_port != 80:
            return

        payload = str(transporthdr.get_data_as_string()).lower()
        ascii_payload = [ord(c) for c in payload]
        data_input[d_port].append(ascii_payload)
        counter[d_port] += 1

        if counter[d_port] == batch_size:
            lstm_train(data_input, d_port)

        #if len(data_input[d_port]) == batch_size:
        #    train(d_port, data_input[d_port])


def lstm_train(payloads, d_port):
    dataX = []
    dataY = []

    for payload in payloads[d_port]:
        for i in range(0, len(payload) - seq_length, 1):
            seq_in = payload[i:i+seq_length]
            seq_out = payload[i+seq_length]
            dataX.append(seq_in)
            dataY.append(seq_out)

    n_pattern = len(dataX)
    dataX = numpy.reshape(dataX, (n_pattern, seq_length, 1))
    dataX = dataX / float(255)
    print "X : ", dataX.shape[1], dataX.shape[2]
    print "Y : ", len(dataY)
    dataY = np_utils.to_categorical(dataY)
    print "Y : ", dataY.shape[0], dataY.shape[1]
    print "test"
    model = Sequential()
    model.add(LSTM(256, input_shape=(dataX.shape[1], dataX.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(dataY.shape[1], activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    filepath = "models/lstm-" + str(d_port) + "-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True)
    callback_list = [checkpoint]

    #model.fit(dataX, dataY, nb_epoch=50, batch_size=128, callbacks=callback_list)
    model.train_on_batch(dataX, dataY)
    model.save("models/lstm.hdf5", overwrite=True)
    counter[d_port] = 0
    del data_input[d_port]
    data_input[d_port] = []
    print "1 batch"


if __name__ == '__main__':
	main(sys.argv)