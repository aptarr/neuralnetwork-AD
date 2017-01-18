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
seq_length = 50
batch_size = 128
data_input = {20: [], 21: [], 22: [], 23: [], 25: [], 53: [], 80: [], 110: [], 139: [], 143: [], 443: [], 445: []}

def read_dataset(filename, mode):
    dataset_dir = "/home/baskoro/Documents/Dataset/ISCX12/without retransmission/"
    cap = pcapy.open_offline(dataset_dir + filename)

    while (1):
        (header, packet) = cap.next()
        if not header:
            break
        parse_packet(header, packet)

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

        if d_length <= seq_length:
            return

        if d_port not in ports:
            return

        payload = str(transporthdr.get_data_as_string()).lower()
        ascii_payload = [ord(c) for c in payload]
        data_input[d_port].append(ascii_payload)
        #if len(data_input[d_port]) == batch_size:
        #    train(d_port, data_input[d_port])

def lstm_train(input, d_port):

    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))