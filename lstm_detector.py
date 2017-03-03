import numpy
import pcapy
import sys
import time

from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from impacket import ImpactDecoder, ImpactPacket


seq_length = 20
threshold = (0.209352331307 + 2 * 0.154211673538)

def main(argv):
    try:
        model = init_model()
        read_dataset(argv[1], int(argv[2]), model)
    except IndexError:
        print "Usage : python.py lstm_detector dataset_filename port"


def init_model():
    filename = "models/lstm.hdf5"
    model = load_model(filename)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def read_dataset(filename, port, model):
    dataset_dir = "/home/baskoro/Documents/Dataset/ISCX12/without retransmission/"
    cap = pcapy.open_offline(dataset_dir + filename)
    anomaly_scores = []
    detection_decisions = []
    fresult = open("results/result-lstm-{}.csv".format(port), "w")

    #for i in range(2000):
    while(True):
        (header, packet) = cap.next()
        if not header:
            break
        ascii_payload = parse_packet(header, packet, port, fresult)

        if ascii_payload is not None:
            length = len(ascii_payload)
            anomaly_score = detect(model, ascii_payload) / float(length)
            if anomaly_score > threshold:
                fresult.write("{},1\n".format(anomaly_score))
                #detection_decisions.append(1)
            else:
                fresult.write("{},0\n".format(anomaly_score))
                #detection_decisions.append(0)

            anomaly_scores.append(anomaly_score)
            print anomaly_score

    #print packets
    #print detection_decisions
    #mean = numpy.mean(anomaly_scores)
    #stdev = numpy.std(anomaly_scores)
    #print mean, stdev
    #numpy.savetxt("results/result-lstm-{}.csv".format(port), packets, delimiter=",")
    fresult.close()


def parse_packet(header, packet, port, fresult):
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
            return None

        if d_length <= seq_length:
            return None

        if d_port != port:
            return None

        fresult.write("{}, {}, {}, {}, {}, {}, {},".format(s_addr, s_port, d_addr, d_port, protocol, seq_num, d_length))
        payload = str(transporthdr.get_data_as_string()).lower()
        ascii_payload = [ord(c) for c in payload]

        return ascii_payload


def detect(model, ascii_payload):
    x_batch = []
    y_batch = []
    anomaly_score = 0
    counter = 0
    #print 2
    for i in range(0, len(ascii_payload) - seq_length):
        x = ascii_payload[i:i+seq_length]
        y = ascii_payload[i+seq_length]
        x = numpy.reshape(x, (1, seq_length, 1))
        x = x / float(255)
        counter += 1

        if len(x_batch) == 0:
            x_batch = x
            y_batch = y
        else:
            x_batch = numpy.r_[x_batch, x]
            y_batch = numpy.r_[y_batch, y]

    #start = time.time()
    prediction = model.predict_on_batch(x_batch)
    #end = time.time()
    #print ((end-start) * 1000)
    predicted_y = numpy.argmax(prediction, axis=1)
    #print prediction.shape[0], prediction.shape[1]
    #print predicted_y

    for i in range(0, len(y_batch)):
        #print y_batch[i], predicted_y[i]
        if y_batch[i] != predicted_y[i]:
            anomaly_score += 1

    return anomaly_score


if __name__ == '__main__':
	main(sys.argv)